import os
import os.path as osp
import time
from collections import defaultdict
import faiss
import numpy as np
import logging
import pickle
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    BertModel,
)
from vllm import LLM, SamplingParams
from arguments import parse_args
from experiment_retrieve_table import load_retriever
from models.LLMs.models import Retriever
from src.load_data import load_pickle, load_data
from src.prompt import generate_prompt
from src.preprocess import add_special_token
from src.load_data import load_json, save_json
from src.normalize_text import normalize
from tqdm import tqdm
from src.preprocess import process_questions
import re
import torch

from src.preprocess import (
    generate_unique_tables,
    generate_hyperedges_plaintext_from_tables,
    generate_node_plaintext_from_tables,
    generate_plaintext_from_table,
    construct_hypergraph,
)
from src.preprocess import (
    START_ROW_TAG,
    END_ROW_TAG,
    START_COL_TAG,
    END_COL_TAG,
    START_NODE_TAG,
    END_NODE_TAG,
    START_TITLE_TAG,
    END_TITLE_TAG,
    START_SUMMARY_TAG,
    END_SUMMARY_TAG,
    NULL_TAG,
)
from models.GNNs.allset import Encoder


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
        embedding_files = [os.path.join(embedding_dir, f"tables.pickle")]
        for fidx, file_path in enumerate(embedding_files):
            logger.info(f"Loading embeddings from {file_path}")
            data = load_pickle(file_path)
            ids, embeddings = data["ids"], data["embedding"]

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

        logger.info(f"Data indexing completed")

    def load_passages(self, passages_dir):
        passages_files = [os.path.join(passages_dir, f"tables.pickle")]
        passages = {}
        count = 0
        for fidx, file_path in enumerate(passages_files):
            logger.info(f"Loading passages from {file_path}")
            data = load_pickle(file_path)
            ids, plaintexts = data["ids"], data["plaintext"]
            for id, passage in zip(ids, plaintexts):
                passages[id] = passage
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

    def search_table(self, query, top_n=30):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        top_ids_and_scores = self.index.search_knn(questions_embedding, top_n)

        return top_ids_and_scores[0][0][:top_n]

    def setup_retriever(self):
        logger.info(f"Load model from {self.args.LLMs_retriever_model_name}")
        self.model, self.tokenizer = load_retriever(self.args.LLMs_retriever_model_name)

        if self.args.LLMs_retriever_include_tags:
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

        input_path = self.args.LLMs_retriever_input_path

        index_dir = osp.join(input_path, "retriever")
        index_path = osp.join(index_dir, "index.faiss")

        if (
            self.args.LLMs_save_or_load_index
            and os.path.exists(index_path)
            and (not self.args.LLMs_reload_index)
        ):
            self.index.deserialize_from(index_dir)
        else:
            logger.info(f"Indexing passages from file {input_path}")
            start_time = time.time()
            self.index_encoded_data(
                self.index,
                # GNNs_dir,
                input_path,
                self.args.LLMs_indexing_batch_size,
            )
            print(f"Indexing time: {time.time() - start_time:.1f}s")
            if self.args.LLMs_save_or_load_index:
                self.index.serialize(index_dir)

        print("Loading passages")

        self.passages = self.load_passages(input_path)
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
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
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


def main(args):
    # ************************* Load Data *************************

    data = load_data(args)

    # ************************* Preprocess Data *************************

    unique_tables, qas, tname2tid, qid2tid, qid2qname = generate_unique_tables(data)

    tables = {}

    for t_idx, (key, table) in enumerate(unique_tables.items()):
        raw_str = load_pickle(
            osp.join(
                args.raw_data_dir, args.dname, "summary", f"summary_{t_idx}.pickle"
            )
        )
        raw_split = raw_str.split("\n")
        assert len(raw_split) == 4
        table["title"] = raw_split[1][14:-2]
        table["summary"] = raw_split[2][16:-1]
        tables[tname2tid[table["name"]]] = table

    if args.LLMs_rephrase_questions:
        rephrase_questions = process_questions(args)
        for qid, question in rephrase_questions.items():
            qas[qid]["question"] = question

    _, tokenizer = load_retriever(args.LLMs_retriever_model_name)
    HGNN = Encoder(args)
    model = HGNN.embed_layer.load_state_dict(osp.join(args.GNNs_dir, "model.pt"))


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    args.LLMs_rephrase_question = True
    args.save_output = False
    args.LLMs_question_batch_size = 64
    main(args)
    print("")

    # args = parse_args()
    # main(args)
    # print("")
