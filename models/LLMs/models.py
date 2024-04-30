from json import load
import logging
from re import T
import time
import os, os.path as osp
from transformers import AutoConfig, AutoTokenizer, BertModel
import torch
import numpy as np
import pickle
import faiss
from tqdm import tqdm
from src.load_data import load_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def load_retriever(model_path, pooling="average", random_init=False):
    cfg = load_hf(AutoConfig, model_path)
    tokenizer = load_hf(AutoTokenizer, model_path)
    retriever = load_hf(Contriever, model_path)

    return retriever, tokenizer


def load_hf(obj_class, model_name):
    try:
        obj = obj_class.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        logger.warning(f"Failed to load model from local files: {e}")
        obj = obj_class.from_pretrained(model_name, local_files_only=False)
    return obj


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):
        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class Retriever:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def index_encoded_data(self, index, embedding_dir, indexing_batch_size):
        logger.info(f"Start data indexing ...")
        allids = []
        allembeddings = np.array([])
        embedding_types = ["summary", "title", "hyperedge"]
        embedding_files = [
            os.path.join(embedding_dir, f"embeddings_{etype}.pickle")
            for etype in embedding_types
        ]
        self.mapping_type2dbids = {}
        for fidx, file_path in enumerate(embedding_files):
            logger.info(f"Loading embeddings from {file_path}")
            data = load_pickle(file_path)
            ids, embeddings = data["ids"], data["embeddings"]

            self.mapping_type2dbids[embedding_types[fidx]] = ids
            allids.extend(ids)
            allembeddings = (
                np.vstack((allembeddings, embeddings))
                if allembeddings.size
                else embeddings
            )
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size
                )
        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

        self.mapping_tid2eids = load_pickle(osp.join(embedding_dir, "tid2eids.pickle"))
        assert sum([len(value) for value in self.mapping_tid2eids.values()]) == len(
            self.mapping_type2dbids["hyperedge"]
        ), "Hyperedge embeddings are not properly loaded"
        logger.info(f"Data indexing completed")

    def load_passages(self, passages_dir):
        passage_types = ["summary", "title", "hyperedge"]
        passages_files = [
            os.path.join(passages_dir, f"plaintext_{etype}.pickle")
            for etype in passage_types
        ]
        passages = {}
        count = 0
        for fidx, file_path in enumerate(passages_files):
            logger.info(f"Loading passages from {file_path}")
            data = load_pickle(file_path)
            for id, passage in data.items():
                passages[id] = passage
            count += len(data.keys())
        assert count == len(passages.keys()), "Passages are not properly loaded"
        return passages

    def embed_queries(self, args, queries):
        embeddings, batch_questions = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.LLMs_lowercase:
                    q = q.lower()
                if args.LLMs_normalize_text:
                    pass
                batch_questions.append(q)
                if (
                    len(batch_questions) == args.LLMs_question_batch_size
                    or k == len(queries) - 1
                ):
                    encoded_batch = self.tokenizer(
                        batch_questions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.LLMs_question_maxlength,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    embeddings.append(self.model(**encoded_batch, normalize=True))
                    batch_questions = []
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.cpu().numpy()

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def search_document_batch(self, queries, top_n=30):
        questions_embedding = self.embed_queries(self.args, queries)

        index_id_to_db_ids = self.index.get_id_mapping()

        ### Step 1: Search with titles:
        title_ids = self.mapping_type2dbids["title"]
        title_index_ids = [index_id_to_db_ids[id] for id in title_ids]

        top_title_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            title_index_ids,
            self.args.LLMs_num_docs,
        )

        top_title_ids = [
            top_title_ids_and_score[0][:top_n]
            for top_title_ids_and_score in top_title_ids_and_scores
        ]

        top_tids_from_title = [
            [int(id.split("_")[-1]) for id in top_title_id]
            for top_title_id in top_title_ids
        ]

        ### Step 2: Search with summaries:
        summary_ids = self.mapping_type2dbids["summary"]
        summary_index_ids = [index_id_to_db_ids[id] for id in summary_ids]

        top_summary_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            summary_index_ids,
            self.args.LLMs_num_docs,
        )

        top_summary_ids = [
            top_summary_ids_and_score[0][:top_n]
            for top_summary_ids_and_score in top_summary_ids_and_scores
        ]

        top_tids_from_summary = [
            [int(id.split("_")[-1]) for id in top_summary_id]
            for top_summary_id in top_summary_ids
        ]

        top_tids = [
            list(
                f"table_{id}"
                for id in set(tids_from_title).intersection(set(tids_from_summary))
            )
            for tids_from_title, tids_from_summary in zip(
                top_tids_from_title, top_tids_from_summary
            )
        ]

        ### Step 3: Search with hyperedges:
        res = []
        for tids in top_tids:
            hyperedge_index_ids = [
                index_id_to_db_ids[eid]
                for tid in tids
                for eid in self.mapping_tid2eids[tid]
            ]

            top_hyperedge_ids_and_scores = self.index.targeted_search_knn(
                questions_embedding,
                hyperedge_index_ids,
                self.args.LLMs_num_docs,
            )

            top_hyperedge_ids = top_hyperedge_ids_and_scores[0][0][:top_n]
            res.append(
                {"hyperedge": {id: self.passages[id] for id in top_hyperedge_ids}}
            )

        return res

    def search_document(self, query, top_n=30):
        questions_embedding = self.embed_queries(self.args, [query])

        index_id_to_db_ids = self.index.get_id_mapping()
        ### Step 1: Search with titles:
        title_ids = self.mapping_type2dbids["title"]
        title_index_ids = [index_id_to_db_ids[id] for id in title_ids]

        top_title_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            title_index_ids,
            self.args.LLMs_num_docs,
        )

        top_title_ids = top_title_ids_and_scores[0][0][:top_n]

        top_tids_from_titles = [int(id.split("_")[-1]) for id in top_title_ids]

        ### Step 2: Search with summaries:
        summary_ids = self.mapping_type2dbids["summary"]
        summary_index_ids = [index_id_to_db_ids[id] for id in summary_ids]

        top_summary_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            summary_index_ids,
            self.args.LLMs_num_docs,
        )

        top_summary_ids = top_summary_ids_and_scores[0][0][:top_n]

        top_tids_from_summary = [int(id.split("_")[-1]) for id in top_summary_ids]

        # TODO how to combine those two lists?

        top_tids = list(
            f"table_{id}"
            for id in set(top_tids_from_titles).intersection(set(top_tids_from_summary))
        )

        ### Step 3: Search with hyperedges:
        hyperedge_index_ids = [
            index_id_to_db_ids[eid]
            for tid in top_tids
            for eid in self.mapping_tid2eids[tid]
        ]

        top_hyperedge_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            hyperedge_index_ids,
            self.args.LLMs_num_docs,
        )

        top_hyperedge_ids = top_hyperedge_ids_and_scores[0][0][:top_n]

        # top_ids_and_scores = self.index.search_knn(
        #     questions_embedding, self.args.LLMs_num_docs
        # )
        # return self.add_passages(self.passages, top_ids_and_scores)[:top_n]
        return {
            "title": {id: self.passages[id] for id in top_title_ids},
            "summary": {id: self.passages[id] for id in top_summary_ids},
            "hyperedge": {id: self.passages[id] for id in top_hyperedge_ids},
        }

    def search_table(self, query, top_n=30):
        questions_embedding = self.embed_queries(self.args, [query])

        index_id_to_db_ids = self.index.get_id_mapping()
        ### Step 1: Search with titles:
        title_ids = self.mapping_type2dbids["title"]
        title_index_ids = [index_id_to_db_ids[id] for id in title_ids]

        top_title_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            title_index_ids,
            self.args.LLMs_num_docs,
        )

        top_title_ids = top_title_ids_and_scores[0][0][:top_n]

        top_tids_from_titles = [id for id in top_title_ids]

        ### Step 2: Search with summaries:
        summary_ids = self.mapping_type2dbids["summary"]
        summary_index_ids = [index_id_to_db_ids[id] for id in summary_ids]

        top_summary_ids_and_scores = self.index.targeted_search_knn(
            questions_embedding,
            summary_index_ids,
            self.args.LLMs_num_docs,
        )

        top_summary_ids = top_summary_ids_and_scores[0][0][:top_n]

        top_tids_from_summary = [id for id in top_summary_ids]

        top_tids = list(
            f"table_{id}"
            for id in set(top_tids_from_titles).intersection(set(top_tids_from_summary))
        )

        return {"title": top_tids_from_summary, "summary": top_tids_from_titles}

    def add_passages(self, passages, top_passages_and_scores):
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def setup_retriever(self):
        logger.info(f"Load model from {self.args.LLMs_retriever_model_name}")
        self.model, self.tokenizer = load_retriever(self.args.LLMs_retriever_model_name)

        if self.args.LLMs_retriever_inclue_tages:
            node_token_names = ["[NODE]", "[/NODE]"]
            row_token_names = ["[ROW]", "[/ROW]"]
            col_token_names = ["[COL]", "[/COL]"]
            title_token_names = ["[TITLE]", "[/TITLE]"]
            summary_token_names = ["[SUMMARY]", "[/SUMMARY]"]
            null_token_name = "[NULL]"

            self.tokenizer.add_special_tokens(
                special_tokens_dict={
                    "additional_special_tokens": node_token_names
                    + row_token_names
                    + col_token_names
                    + title_token_names
                    + summary_token_names
                    + [null_token_name]
                }
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(device)
        # self.tokenizer = self.tokenizer.to(device)

        if not self.args.LLMs_no_fp16:
            self.model = self.model.half()

        self.index = Indexer(
            self.args.LLMs_projection_size,
            self.args.LLMs_n_subquantizers,
            self.args.LLMs_n_bits,
        )

        input_paths = self.args.LLMs_retriever_input_path

        GNNs_dir = osp.join(input_paths, "GNNs")
        pretrain_dir = osp.join(input_paths, "pretrain")
        index_dir = osp.join(input_paths, "retriever")
        index_path = osp.join(index_dir, "index.faiss")

        if (
            self.args.LLMs_save_or_load_index
            and os.path.exists(index_path)
            and (not self.args.LLMs_reload_index)
        ):
            self.index.deserialize_from(index_dir)
        else:
            print(f"Indexing passages from file {input_paths}")
            start_time = time.time()
            self.index_encoded_data(
                self.index,
                # GNNs_dir,
                pretrain_dir,
                self.args.LLMs_indexing_batch_size,
            )
            print(f"Indexing time: {time.time() - start_time:.1f}s")
            if self.args.LLMs_save_or_load_index:
                self.index.serialize(index_dir)

        # titleid2eid_path = osp.join(GNNs_dir, "titleid2eid.pickle")
        # self.titleid2eid = load_pickle(titleid2eid_path)
        # summaryid2eid_path = osp.join(GNNs_dir, "summaryid2eid.pickle")
        # self.summaryid2eid = load_pickle(summaryid2eid_path)

        print("Loading passages")

        passages_dir = osp.join(input_paths, "pretrain")
        self.passages = self.load_passages(passages_dir)

        logger.info("Passages have been loaded")


class Indexer(object):
    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(
                vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        # self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype("float32")
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f"Total data indexed {len(self.index_id_to_db_id)}")

    def search_knn(
        self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048
    ):
        query_vectors = query_vectors.astype("float32")
        result = []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        # for k in tqdm(range(nbatch)):
        for k in range(nbatch):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                for query_top_idxs in indexes
            ]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def targeted_search_knn(
        self, query_vectors, ranges, top_docs, index_batch_size=2048
    ):
        query_vectors = query_vectors.astype("float32")
        result = []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        # for k in tqdm(range(nbatch)):
        for k in range(nbatch):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            id_selector = faiss.IDSelectorArray(ranges)
            params = faiss.SearchParametersIVF(sel=id_selector)
            scores, indexes = self.index.search(q, top_docs, params=params)
            # convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                for query_top_idxs in indexes
            ]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        print(f"Serializing index to {index_file}, meta data to {meta_file}")

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        print(f"Loading index from {index_file}, meta data from {meta_file}")

        self.index = faiss.read_index(index_file)
        print(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids):
        # new_ids = np.array(db_ids, dtype=np.int64)
        # self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)

    def get_id_mapping(self):
        return {
            db_id: index_id for index_id, db_id in enumerate(self.index_id_to_db_id)
        }


# if __name__ == "__main__":
#     # path = "/media/mtybilly/My Passport1/Program/Baselines/self-rag/retrieval_lm/psgs_w100.tsv"
#     # import csv

#     # passages = []
#     # with open(path) as f:
#     #     reader = csv.reader(f, delimiter="\t")
#     #     for k, row in enumerate(reader):
#     # if not row[0] == "id":
#     #     ex = {"id": row[0], "title": row[2], "text": row[1]}
#     #     passages.append(ex)
#     args = parse_args()
#     retreiver = Retreiver(args)
#     retriever.setup_retriever()
#     print("")
