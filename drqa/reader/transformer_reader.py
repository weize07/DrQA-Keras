import argparse
import os
import json

from drqa.reader import utils, vector, config, data
from keras.models import Model
from keras.layers import Input, Dense
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

def build_start_model(ctx_dim, ctx_length, which_side):
    """
    softmax(CTX * Ws * Q), Ws/e: matrix for start/end injection 
    which_side: start/end
    """
    inputs_ctx = Input(shape=(ctx_dim, ctx_length))
    inputs_que = Input(shape=(ctx_dim, 1))

    ctx_multi_mat = Dense(ctx_dim, activation='linear')(inputs_ctx)
    ctx_mat_que = merge([ctx_multi_mat, inputs_que], output_shape=ctx_length, mode='mul')

    output = Dense(ctx_dim, activation='softmax', name=('%s_probs' % which_side))(ctx_mat_que)
    model = Model(input=[inputs_ctx, inputs_que], output=output)
    return model

def train_network(train_exs):
    X = []
    Y = []
    for train_ex in train_exs:
        for i in range(len(train_ex['document'])):
            x = train_ex['document'][i] + train_ex['question']
            if train['answer'][0] == i:
                y1 = [1]
            else:
                y1 = [0]
            if train['answer'][1] == i:
                y2 = [1]
            else:
                y2 = [0]
            X.append(x)
            Y.append([y1, y2])
    
    input_dim = len(X[0])
    a = Input(shape=(input_dim,))
    b1 = Dense(input_dim)(a)
    b2 = Dense(input_dim)(a)
    model = Model(inputs=a, outputs=[b1, b2])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, Y)

    model.save()

def load_model(model_path):
    a = Input(shape=(input_dim,))
    b1 = Dense(input_dim)(a)
    b2 = Dense(input_dim)(a)
    model = Model(inputs=a, outputs=[b1, b2])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.load()
    return model

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
    train_exs = prepare_data(args)
    print(train_exs[0])
    train_network(train_exs)



