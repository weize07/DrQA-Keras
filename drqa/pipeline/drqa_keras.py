from keras.models import load_model
from multiprocessing import Pool as ProcessPool

def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

class DrQA_K(object):
    def __init__(self, reader_model, ranker_config, db_config, bert_path):
        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing document reader...')
        self.reader = load_model(reader_model)

        logger.info('Initializing bert embedder...')
        self.bert_embedder = BERTEmbedder(bert_path)

        logger.info('Initializing document retrivers')
        tok_class = DEFAULTS['tokenizer']
        tok_opts = set()
        db_config = db_config or {}
        db_class = db_config.get('class', DEFAULTS['db'])
        db_opts = db_config.get('options', {})
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )

    def process(self, query, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        all_docids, all_doc_scores = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)
        (doc_embeddings, query_embedding) = self.bert_embedder.embed(doc_texts, query)
        doc_best_span = []
        for doc in doc_embeddings:
            # process doc one by one, to find answer span in each document
            doc_predicts = []
            for te in doc:
                res = self.reader_model.predict(te + query_embedding)
                doc_predicts.append(res)

            start_end = [-1, -1, -1] # start, end, probability
            for i in range(len(doc_predicts)):
                predict = doc_predicts[i]
                start_prob = predict[0]
                for j in range(i + 1, min(len(doc_predicts), i + 15)):
                    if start_prob * doc_predicts[j][1] > start_end[2]:
                        start_end = [i, j, start_prob * doc_predicts[j][1]]
            doc_best_span.append(start_end)
        return doc_best_span

            


        
        