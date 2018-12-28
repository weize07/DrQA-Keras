import argparse
import os
import json

from drqa.reader import utils, vector, config, data
ROOT_DIR = '/Users/weize/Workspace/VENV-3.6/workspace/DrQA-Keras'

def prepare_data(args):
    bert_que_features = {}
    with open(args.bert_question_feature_file) as f:
        for line in f:
            jo = json.loads(line)
            # use last layer of bert as doc_features
            bert_que_features[jo['id']] = jo['features']['layers'][0]
    bert_doc_features = {}
    with open(args.bert_question_feature_file) as f:
        for line in f:
            jo = json.loads(line)
            bert_doc_features[jo['id']] = jo['features']['layers'][0]
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    for train_ex in train_exs:
        train_ex['question'] = bert_que_features[train_ex['id']][0] # which is the encoding of "[CLS]" from bert
        train_ex['document'] = bert_doc_features[train_ex['id']]
        del train_ex['offsets']
        del train_ex['lemma']
        del train_ex['qlemma']
        del train_ex['pos']
        del train_ex['ner']

    return train_exs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    args.train_file = os.path.join(ROOT_DIR, 'data/datasets/SQuAD-v1.1-train-processed-corenlp.txt')
    args.bert_doc_feature_file = os.path.join(ROOT_DIR, 'data/datasets/doc_output.jsonl')
    args.bert_question_feature_file = os.path.join(ROOT_DIR, 'data/datasets/que_output.jsonl')
    args.uncased_question = True
    args.uncased_doc = True
    train_data = prepare_data(args)
    print(train_data[0])

