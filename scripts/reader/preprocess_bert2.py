import json
import os
import argparse

from drqa.reader import BERTEmbedder

def load_dataset(input_file):
    """Read a list of `InputExample`s from an input file."""
    dataset = []
    with open(input_file) as f:
        data = json.load(f)['data']
        for article in data:
            dataset.extend(article['paragraphs'])
            break
    return dataset

def find_answer(ctx_token_ids, ans_token_ids):
    """Match token offsets with the char begin/end offsets of the answer."""
    # TODO: use KMP to improve performance
    ans_index = 0
    ctx_start = 0
    ctx_index = 0
    while ctx_start < len(ctx_token_ids) - len(ans_token_ids):
        if ans_token_ids[ans_index] == ctx_token_ids[ctx_index]:
            ans_index += 1
            ctx_index += 1
            if ans_index == len(ans_token_ids):
                return [ctx_start, ctx_index]
        else:
            ctx_start += 1
            ctx_index = ctx_start
            answer_index = 0
    return None


def process_dataset(dataset, bert_path, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    bert_embedder = BERTEmbedder(bert_path)

    processed = {'contexts': {}, 'qas': {}}
    cid = 1
    for ctx_and_qas in dataset:
        context = ctx_and_qas['context']
        qas = ctx_and_qas['qas']
        (ctx_bert_predict, ctx_raw_features) = bert_embedder.embed_document(context)
        ctx_bert_features = None
        for result in ctx_bert_predict:
            ctx_bert_features = result['layer_output_0']
        ctx_token_ids = ctx_raw_features[0].input_ids
        ctx_tokens = ctx_raw_features[0].tokens
        ctx = {'bert_features': ctx_bert_features, 'cid': cid, 'text': context}
        print('----------')
        for qa in qas:
            question = qa['question']
            (q_bert_predict, q_raw_features) = bert_embedder.embed_question(question)
            q_bert_features = None
            for result in q_bert_predict:
                q_bert_features = result['layer_output_0']
            answer_token_ids = bert_embedder.convert_txt_to_token_ids(qa['answers'][0]['text'])
            answer_offsets = find_answer(ctx_token_ids, answer_token_ids)
            if answer_offsets is not None:
                print(ctx_tokens[answer_offsets[0]:answer_offsets[1]])
            exit()
        cid += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory')
    # parser.add_argument('out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--bert_path', type=str, help='Path to output file dir')
    # parser.add_argument('--split', type=str, help='Filename for train/dev split',
    #                     default='SQuAD-v1.1-train')
    # # parser.add_argument('--workers', type=int, default=None)
    # # parser.add_argument('--tokenizer', type=str, default='corenlp')
    args = parser.parse_args()
    # bert_path = '/Users/weize/Workspace/VENV-3.6/workspace/bert/uncased_L-12_H-768_A-12'
    dataset = load_dataset(os.path.join(args.data_dir, 'SQuAD-v1.1-train.json'))
    print('SQuAD-v1.1 dataset length: ', len(dataset))
    process_dataset(dataset, args.bert_path)

    