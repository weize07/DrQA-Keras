import argparse
import os
import json

from hdf5_util import *
from drqa.reader import utils, vector, config, data, DataGenerator
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, merge

def build_model(ctx_dim, ctx_length, which_side):
    """
    softmax(CTX * Ws * Q), Ws/e: matrix for start/end injection 
    which_side: start/end
    """
    inputs_ctx = Input(shape=(ctx_dim, ctx_length))
    inputs_que = Input(shape=(ctx_dim, 1))

    # ctx_multi_mat = Dense(ctx_dim, activation='linear')(inputs_ctx)
    que_multi_mat = Dense(ctx_dim, activation='linear')(inputs_ctx)
    # ctx_mat_que = merge([ctx_multi_mat, inputs_que], output_shape=ctx_length, mode='mul')
    ctx_mat_que = merge([inputs_ctx, que_multi_mat], output_shape=ctx_length, mode='mul')

    output = Dense(ctx_dim, activation='softmax', name=('%s_probs' % which_side))(ctx_mat_que)
    model = Model(input=[inputs_ctx, inputs_que], output=output)
    return model

def train_network(dataset):
    start_model = build_model(768, 512, 'start')
    # end_model = build_model(768, 512, 'end')
    X = []
    Y = []
    # for 

    start_model.fit(X, Y)

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
    parser.add_argument('--dataset', type=str, help='Path to dataset file')
    args = parser.parse_args()
    dataset = load_dict_from_hdf5(args.dataset)
    X = []
    Y = []
    for qid in dataset['qas']:
        cid = dataset['qas'][qid]['cid']
        print(dataset['qas'][qid]['bert_features'][0].shape)
        print(dataset['contexts'][cid]['bert_features'].shape)
        x = np.array([dataset['contexts'][cid]['bert_features'], dataset['qas'][qid]['bert_features'][0]])
        print(x.shape)
        print(x)
        X.append(x)
        Y.append(to_categorical(dataset['qas'][qid]['answer_offsets'][0], num_classes=512))
        break
    start_model = build_model(768, 512, 'start')
    start_model.fit(X, Y)
    # Datasets
    # partition = # IDs
    # labels = # Labels
    # Parameters
    params = {'dim': (32,32,32),
          'batch_size': 32,
          'n_classes': 512,
          'n_channels': 1,
          'shuffle': False}

    # Generators
    # training_generator = DataGenerator(dataset, **params, 'start')
    # validation_generator = DataGenerator(partition['validation'], labels, **params)
    # print(dataset)
    # train_network(dataset)



