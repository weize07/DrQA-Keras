import json

def load_dataset(input_file):
    """Read a list of `InputExample`s from an input file."""
    dataset = []
    with open(input_file) as f:
        data = json.load(f)['data']
        for article in data:
            dataset.extend(article['paragraphs'])
    return dataset

def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]

def process_dataset(dataset, bert_path, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    bert_embedder = BERTEmbedder(bert_path)

    for ctx_and_qas in dataset:
        context = ctx_and_qas['context']
        qas = ctx_and_qas['qas']
        (ctx_bert_features, ctx_raw_features) = bert_embedder.embed_document(context)
        for qa in qas:
            question = qa['question']
            (q_bert_features, q_raw_features) = bert_embedder.embed_question(question)
    


if __name__ == '__main__':
    bert_path = '/Users/weize/Workspace/VENV-3.6/workspace/bert/uncased_L-12_H-768_A-12'
    dataset = read_dataset('../../data/datasets/SQuAD-v1.1-train.json')
    process_dataset(dataset, bert_path)
    # print(dataset[0])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
    # parser.add_argument('out_dir', type=str, help='Path to output file dir')
    # parser.add_argument('bert_path', type=str, help='Path to output file dir')
    # parser.add_argument('--split', type=str, help='Filename for train/dev split',
    #                     default='SQuAD-v1.1-train')
    # # parser.add_argument('--workers', type=int, default=None)
    # # parser.add_argument('--tokenizer', type=str, default='corenlp')
    # args = parser.parse_args()

    # t0 = time.time()

    # in_file = os.path.join(args.data_dir, args.split + '.json')
    # print('Loading dataset %s' % in_file, file=sys.stderr)
    # dataset = load_dataset(in_file)

    # out_file = os.path.join(
    #     args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
    # )
    # print('Will write to file %s' % out_file, file=sys.stderr)
    # with open(out_file, 'w') as f:
    #     for ex in process_dataset(dataset, args.tokenizer, args.workers):
    #         f.write(json.dumps(ex) + '\n')
    # print('Total time: %.4f (s)' % (time.time() - t0))