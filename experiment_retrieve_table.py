import os
import os.path as osp
import time
import logging
from transformers import AutoTokenizer, AutoConfig, BertModel, AutoModel
import faiss
import pickle
import numpy as np

# from vllm import LLM, SamplingParams
from arguments import parse_args
from src.prompt import generate_prompt
from src.preprocess import add_special_token
from src.load_data import load_json, save_json, save_pickle, load_pickle
from src.normalize_text import normalize
from tqdm import tqdm
import re
import torch
from torch_scatter import scatter
from pretrain_embedding import generate_embeddings
from src.preprocess import (
    add_special_token,
    generate_plaintext_from_table,
    process_questions,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def load_model(model_path):
    if model_path == "facebook/contriever-msmarco":
        return load_retriever(model_path)
    else:
        tokenzier = load_hf(AutoTokenizer, model_path)
        model = load_hf(AutoModel, model_path)
    return model, tokenzier


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
        self.encoded_data = []
        for fidx, file_path in enumerate(embedding_files):
            logger.info(f"Loading embeddings from {file_path}")
            data = load_pickle(file_path)
            ids, embeddings = data["ids"], data["embedding"]
            self.encoded_data.extend(embeddings)
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
            data["plaintext"] = ["" for _ in data["ids"]]
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

        embeddings = embeddings.cpu().numpy()
        # embeddings = np.concatenate([embeddings, embeddings], axis=-1)
        return embeddings
        # return embeddings.cpu().numpy()

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        # embeddings_to_add = np.concatenate(
        #     [embeddings_to_add, embeddings_to_add], axis=-1
        # )
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def search_table(self, query, top_n=30):
        if isinstance(query, list):
            questions_embedding = self.embed_queries(self.args, query)
        else:
            questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        top_ids_and_scores = self.index.search_knn(questions_embedding, top_n)

        if isinstance(query, list):
            res = [v[0][:top_n] for v in top_ids_and_scores]
        else:
            res = top_ids_and_scores[0][0][:top_n]
        return res

    def setup_retriever(self):
        logger.info(f"Load model from {self.args.LLMs_retriever_model_name}")
        self.model, self.tokenizer = load_model(self.args.LLMs_retriever_model_name)

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


def run_with_table(args):
    ### Step 1: Load data

    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    raw_dir = osp.join(args.raw_data_dir, args.dname)
    rephrase_files = [
        file.split(".")[0] for file in os.listdir(osp.join(raw_dir, "rephrase"))
    ]

    if args.save_output:
        save_model_name = args.LLMs_generator_model_name.split("/")[-1]
        curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

        output_dir = osp.join(
            args.root_dir,
            "output",
            "LLMs",
            f"{args.task_mode}_{save_model_name}_{curr_time}",
        )
        os.makedirs(output_dir, exist_ok=True)

    data = load_pickle(osp.join(pretrain_dir, "plaintext_tables.pickle"))
    tables, qas = data["tables"], data["qas"]
    mappings = load_pickle(osp.join(pretrain_dir, "mapping.pickle"))
    qid2tid = mappings["qid2tid"]

    retriever = Retriever(args)
    retriever.setup_retriever()
    # res = retriever.search_document(query1)

    # tokenzier = add_special_token(tokenzier)
    torch.cuda.empty_cache()
    # model = LLM(
    #     model=args.LLMs_generator_model_name,
    #     download_dir=args.LLMs_dir,
    #     dtype=args.LLMs_dtype,
    #     tensor_parallel_size=args.LLMs_world_size,
    # )
    model = None
    # model.set_tokenizer(tokenzier)

    num_questions = len(qas.items())
    correct = 0
    prompts = []
    questions, ground_truths = [], []
    for idx, (qid, qa) in tqdm(enumerate(qas.items())):
        question, ground_truth = qa["question"], qa["answer"][0]

        questions.append(question)
        ground_truths.append(ground_truth)
        table = tables[qid2tid[qid]]

        evidences = (
            "| "
            + " |".join(
                [normalize(head).replace("\n", " ") for head in table["header"]]
            )
            + "\n"
        )
        for row in table["rows"]:
            evidences += (
                "| "
                + " |".join(normalize(element).replace("\n", " ") for element in row)
                + "\n"
            )

        prompt = generate_prompt(args, question, evidences)
        prompts.append(prompt)
        if len(prompts) == args.LLMs_question_batch_size or idx == num_questions - 1:
            # sampling_params = SamplingParams(
            #     temperature=0.0, top_p=1.0, max_tokens=100, stop=["</answer>"]
            # )
            sampling_params = None
            response = model.generate(prompts, sampling_params)
            predictions = [normalize(result.outputs[0].text) for result in response]
            if args.save_output:
                save_json(
                    [
                        {
                            "prompt": prompt,
                            "question": question,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                        }
                        for prompt, question, ground_truth, prediction in zip(
                            prompts, questions, ground_truths, predictions
                        )
                    ],
                    osp.join(output_dir, f"{idx}.json"),
                )
            print(f"{prompts[0]}\n Response: {predictions[0]}\n GT:{ground_truths[0]}")
            prompts = []
            ground_truths = []
            questions = []
        if idx % 1000 == 0:
            print(f"Finished {idx}/{num_questions} questions")
    print("Accuracy: ", correct / num_questions)


def run(args, qas, model, retriever):
    predictions = {}

    qids = list(qas.keys())
    questions = [qas[qid]["question"] for qid in qids]
    # documents = retriever.search_table(passages, 15)

    # for qid, document in zip(qids, documents):
    #     predictions[qid] = document

    for qid, question in zip(qids, questions):
        document = retriever.search_table(question, 15)
        predictions[qid] = document

    return predictions


def main(args):
    ### Step 1: Load data

    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    output_dir = args.LLMs_retriever_input_path

    data = load_pickle(osp.join(pretrain_dir, "plaintext_data.pickle"))
    tables, qas = data["tables"], data["qas"]

    if args.LLMs_rephrase_question:
        rephrase_questions = process_questions(args)
        for qid, question in rephrase_questions.items():
            qas[qid]["question"] = question

    retriever = Retriever(args)
    retriever.setup_retriever()

    predictions = run(args, qas, None, retriever)

    save_json(predictions, osp.join(output_dir, "predictions.json"))


def evaluate_retriever(args):
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")

    output_dir = args.LLMs_retriever_input_path

    predictions = load_json(osp.join(output_dir, "predictions.json"))

    data = load_pickle(osp.join(pretrain_dir, "plaintext_data.pickle"))

    mappings = data["mappings"]
    qid2tid = mappings["qid2tid"]

    num_questions = len(qid2tid.keys())

    res = {1: 0, 3: 0, 5: 0, 10: 0}

    for qid, prediction in predictions.items():
        gt = qid2tid[qid].split("_")[-1]
        if isinstance(prediction, list):
            prefix_res = [name.split("_")[-1] for name in prediction]
            for k in res.keys():
                if gt in prefix_res[:k]:
                    res[k] += 1

    res = {k: round(100 * v / num_questions, 2) for k, v in res.items()}

    save_json(
        res,
        osp.join(output_dir, "result.json"),
    )
    print(output_dir.split("/")[-1], res)
    return res


def LLMs_generate_table_embeddings(args):
    model, tokenizer = load_model(args.LLMs_retriever_model_name)
    if args.LLMs_pretrain_include_tags:
        tokenizer = add_special_token(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    load_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "plaintext_data.pickle"
    )
    data = load_pickle(osp.join(load_path))
    tables, qas = data["tables"], data["qas"]
    plaintext_tables = {}
    for idx, (tid, table) in enumerate(tables.items()):
        plaintext = generate_plaintext_from_table(table, args)
        plaintext_tables[tid] = plaintext
    tids, embedding_tables = generate_embeddings(
        args, plaintext_tables, model, tokenizer
    )

    if args.save_output:
        #  curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        save_model_name = args.LLMs_retriever_model_name.split("/")[-1]
        output_dir = osp.join(
            args.root_dir,
            "output",
            "LLMs",
            # f"{args.LLMs_table_plaintext_format}_{save_model_name}_{curr_time}",
            f"oq_{args.LLMs_table_plaintext_format}_{save_model_name}"
            if not args.LLMs_rephrase_question
            else f"{args.LLMs_table_plaintext_format}_{save_model_name}",
        )
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    save_pickle(
        {
            "ids": tids,
            "embedding": embedding_tables,
            "plaintext": [plaintext_tables[tid] for tid in tids],
        },
        osp.join(output_dir, "tables.pickle"),
    )
    save_json(
        {"plaintext": plaintext_tables[tids[0]]},
        osp.join(output_dir, "example.json"),
    )
    return output_dir


def GNNs_generate_table_embeddings(args):
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    summary_embedding_path = osp.join(pretrain_dir, "embeddings_summary.pickle")
    title_emnbedding_path = osp.join(pretrain_dir, "embeddings_title.pickle")
    mapping_path = osp.join(pretrain_dir, "mapping.pickle")

    summaries = load_pickle(summary_embedding_path)
    titles = load_pickle(title_emnbedding_path)

    title_ids, title_embeddings = (
        titles["ids"],
        titles["embeddings"],
    )
    summary_ids, summary_embeddings = (
        summaries["ids"],
        summaries["embeddings"],
    )

    mappings = load_pickle(mapping_path)
    nid2tid, eid2tid = mappings["nid2tid"], mappings["eid2tid"]

    node_embedding_path = osp.join(GNNs_dir, "embeddings_node.pickle")
    hyperedge_embedding_path = osp.join(GNNs_dir, "embeddings_hyperedge.pickle")

    nodes = load_pickle(node_embedding_path)
    hyperedges = load_pickle(hyperedge_embedding_path)

    nids, node_embeddings = nodes["ids"], nodes["embeddings"]
    eids, hyperedge_embeddings = hyperedges["ids"], hyperedges["embeddings"]

    hyperedge_embeddings = torch.FloatTensor(hyperedge_embeddings)

    if args.GNNs_table_embedding_format == "all":
        edge_index = np.stack(
            [
                np.arange(hyperedge_embeddings.shape[0]),
                [int(tid.split("_")[-1]) for tid in eid2tid.values()],
            ],
            0,
        )

        edge_index = torch.LongTensor(edge_index)

        table_embeddings = scatter(
            hyperedge_embeddings[edge_index[0]],
            edge_index[1],
            dim=0,
            reduce="mean",
        )
        table_embeddings = table_embeddings.numpy()

    elif args.GNNs_table_embedding_format == "row":
        mask = [int(eid.split("_")[-1]) for eid in eids if eid.startswith("row")]
        row_ids = [eid for eid in eids if eid.startswith("row")]

        hyperedge_embeddings = hyperedge_embeddings[mask]
        edge_index = np.stack(
            [
                np.arange(hyperedge_embeddings.shape[0]),
                [int(eid2tid[eid].split("_")[-1]) for eid in row_ids],
            ],
            0,
        )
        edge_index = torch.LongTensor(edge_index)

        table_embeddings = scatter(
            hyperedge_embeddings[edge_index[0]],
            edge_index[1],
            dim=0,
            reduce="mean",
        )
        table_embeddings = table_embeddings.numpy()

    elif args.GNNs_table_embedding_format == "col":
        mask = [int(eid.split("_")[-1]) for eid in eids if eid.startswith("col")]
        col_ids = [eid for eid in eids if eid.startswith("col")]

        hyperedge_embeddings = hyperedge_embeddings[mask]
        edge_index = np.stack(
            [
                np.arange(hyperedge_embeddings.shape[0]),
                [int(eid2tid[eid].split("_")[-1]) for eid in col_ids],
            ],
            0,
        )
        edge_index = torch.LongTensor(edge_index)

        table_embeddings = scatter(
            hyperedge_embeddings[edge_index[0]],
            edge_index[1],
            dim=0,
            reduce="mean",
        )
        table_embeddings = table_embeddings.numpy()

    elif args.GNNs_table_embedding_format == "node":
        node_embeddings = torch.FloatTensor(node_embeddings)
        edge_index = np.stack(
            [
                np.arange(node_embeddings.shape[0]),
                [int(tid.split("_")[-1]) for tid in nid2tid.values()],
            ],
            0,
        )

        edge_index = torch.LongTensor(edge_index)

        table_embeddings = scatter(
            node_embeddings[edge_index[0]],
            edge_index[1],
            dim=0,
            reduce="mean",
        )
        table_embeddings = table_embeddings.numpy()

    if args.save_output:
        #  curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        save_model_name = args.LLMs_retriever_model_name.split("/")[-1]
        output_dir = osp.join(
            args.root_dir,
            "output",
            "GNNs",
            # f"{args.LLMs_table_plaintext_format}_{save_model_name}_{curr_time}",
            f"{args.GNNs_table_embedding_format}",
        )
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    save_pickle(
        {
            "ids": [f"table_{idx}" for idx in range(table_embeddings.shape[0])],
            "embedding": table_embeddings,
            "plaintext": "",
        },
        osp.join(output_dir, "tables.pickle"),
    )
    return output_dir


def combination_embeddings(args):
    LLMs_table_embedding_dir = osp.join(
        args.root_dir,
        "output",
        f"oq_{args.LLMs_table_plaintext_format}_{args.LLMs_retriever_model_name.split('/')[-1]}"
        if not args.LLMs_rephrase_question
        else f"{args.LLMs_table_plaintext_format}_{args.LLMs_retriever_model_name.split('/')[-1]}",
    )

    GNNs_table_embedding_dir = osp.join(
        args.root_dir, "output", f"GNNs_{args.GNNs_table_embedding_format}"
    )

    LLMs_data = load_pickle(osp.join(LLMs_table_embedding_dir, "tables.pickle"))
    GNNs_data = load_pickle(osp.join(GNNs_table_embedding_dir, "tables.pickle"))

    LLMs_table_embeddings = LLMs_data["embedding"]
    GNNs_table_embeddings = GNNs_data["embedding"]

    LLMs_table_embeddings = np.array(LLMs_table_embeddings)
    GNNs_table_embeddings = np.array(GNNs_table_embeddings)

    table_embeddings = LLMs_table_embeddings + GNNs_table_embeddings

    output_dir = osp.join(
        args.root_dir,
        "output",
        f"oq_{args.LLMs_table_plaintext_format}_{args.GNNs_table_embedding_format}"
        if not args.LLMs_rephrase_question
        else f"{args.LLMs_table_plaintext_format}_{args.GNNs_table_embedding_format}",
    )
    os.makedirs(output_dir, exist_ok=True)
    save_pickle(
        {
            "ids": [f"table_{idx}" for idx in range(table_embeddings.shape[0])],
            "embedding": table_embeddings,
            "plaintext": "",
        },
        osp.join(output_dir, "tables.pickle"),
    )
    return output_dir


if __name__ == "__main__":
    args = parse_args()
    args.save_output = True

    args.LLMs_rephrase_question = True
    args.LLMs_save_or_load_index = False
    args.LLMs_pretrain_include_tags = True
    args.LLMs_projection_size = 768
    # for table_format in ["md", "dict", "html", "sentence"]:
    #     for sub in ["", "_summary", "_summary_title", "_title"]:
    #         table_plaintext_format = table_format + sub
    #         args.LLMs_table_plaintext_format = table_plaintext_format

    #         output_dir = LLMs_generate_table_embeddings(args)
    #         args.LLMs_retriever_input_path = output_dir
    #         main(args)
    #         evaluate_retriever(args)

    # for table_format in ["md", "dict", "html", "sentence"]:
    #     for sub in ["", "_summary", "_summary_title", "_title"]:
    #         table_plaintext_format = table_format + sub
    #         args.LLMs_table_plaintext_format = table_plaintext_format
    #         for GNNs_table_embedding_format in ["all", "row", "col", "node"]:
    #             args.GNNs_table_embedding_format = GNNs_table_embedding_format
    #             # output_dir = GNNs_generate_table_embeddings(args)
    #             output_dir = combination_embeddings(args)
    #             args.LLMs_retriever_input_path = output_dir
    #             main(args)
    #             evaluate_retriever(args)

    # save_model_name = args.LLMs_pretrain_model.split("/")[-1]
    # args.LLMs_retriever_input_path = osp.join(
    #     args.root_dir, "output", "LLMs", f"dict_summary_title_{save_model_name}"
    # )
    # main(args)

    args.LLMs_retriever_model_name = "google-bert/bert-base-uncased"
    args.LLMs_table_plaintext_format = "dict_summary_title"
    args.save_output = True
    output_dir = LLMs_generate_table_embeddings(args)
    args.LLMs_retriever_input_path = output_dir

    main(args)
    evaluate_retriever(args)
