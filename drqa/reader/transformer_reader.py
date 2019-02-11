import argparse
import os
import json

from hdf5_util import *
from drqa.reader import utils, vector, config, data, DataGenerator
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dot, Layer
from keras import regularizers, constraints, initializers, optimizers
import keras.backend as K 
import keras

# class Attention(Layer):
#     '''
#     compute the 'attention' scores for question over ctx, 
#     which is a_i = exp(c_i*W*q) / sum_over_j(exp(c_j*W*q))
#     W is a weight matrix to be learned.
#     '''
#     def __init__(self, ctx_dim, ctx_length, **kwargs):
#         self.ctx_dim = ctx_dim
#         self.ctx_length = ctx_length
#         self.init = initializers.get('glorot_uniform')
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # input_shape: [ctx, question]; ctx: vectors of tokens, question: vector of question
#         self.W = self.add_weight((self.ctx_dim, self.ctx_dim,),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=regularizers.l2(0.01),
#                                  constraint=constraints.max_norm(2.))
#         super(Attention, self).build(input_shape)

#     def call(self, x, mask=None):
#         print(x.shape)
#         tmp = K.dot(K.transpose(x[0]), self.W)
#         tmp = K.dot(tmp, x[1])
#         a = K.exp(tmp)
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#         return a

#     def compute_output_shape(self, input_shape):
#         return self.ctx_dim

# 定义callback类
class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_batch_end(self, batch, logs={}): # batch 为index, logs为当前batch的日志acc, loss...
        self.losses.append(logs.get('loss')) 
        return


def build_model(ctx_dim, ctx_length, which_side):
    """
    softmax(CTX * Ws * Q), Ws/e: matrix for start/end injection 
    which_side: start/end
    """
    # inputs = Input(shape=(2,))
    inputs_ctx = Input(shape=(ctx_dim, ctx_length))
    inputs_que = Input(shape=(ctx_dim, ))

    # # ctx_multi_mat = Dense(ctx_dim, activation='linear')(inputs_ctx)
    que_multi_mat = Dense(ctx_dim, input_dim=ctx_dim, activation='linear', use_bias=False)(inputs_que)
    # print(inputs_ctx.shape)
    # print(inputs_que.shape)
    # print(que_multi_mat.shape)
    # # ctx_mat_que = merge([ctx_multi_mat, inputs_que], output_shape=ctx_length, mode='mul')
    print(inputs_ctx.shape)
    print(que_multi_mat.shape)
    ctx_mat_que = Dot(0)([inputs_ctx, que_multi_mat])
    print(ctx_mat_que.shape)

    output = Dense(ctx_length, input_dim=ctx_length, activation='softmax', name=('%s_probs' % which_side))(ctx_mat_que)
    # print(output.shape)
    # output = Attention(ctx_dim, ctx_length, name=('%s_probs' % which_side))(inputs)
    # model = Model(input=inputs, output=output)
    model = Model(inputs=[inputs_ctx, inputs_que], output=output)
    # model = Model(input=que_multi_mat, output=output)
    # opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.0, nesterov=False)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# def train_network(dataset):
#     start_model = build_model(768, 512, 'start')
#     # end_model = build_model(768, 512, 'end')
#     X = []
#     Y = []
#     # for 

#     start_model.fit(X, Y)

#     model.save()

# def load_model(model_path):
#     a = Input(shape=(input_dim,))
#     b1 = Dense(input_dim)(a)
#     b2 = Dense(input_dim)(a)
#     model = Model(inputs=a, outputs=[b1, b2])
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
#     model.load()
#     return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, help='Path to dataset file')
    args = parser.parse_args()
    dataset = load_dict_from_hdf5(args.dataset)
    X1 = []
    X2 = []
    Y = []
    for qid in dataset['qas']:
        cid = dataset['qas'][qid]['cid']
        # print(dataset['qas'][qid]['bert_features'][0].shape)
        # print(dataset['contexts'][cid]['bert_features'].shape)
        # x = np.array([dataset['contexts'][cid]['bert_features'], dataset['qas'][qid]['bert_features'][0]])
        x1 = dataset['contexts'][cid]['bert_features'].transpose()
        x2 = dataset['qas'][qid]['bert_features'][0].reshape([768,])
        # x1 = K.cast(x1, dtype='float16')
        # x2 = K.cast(x2, dtype='float16')
        # test = K.dot(x1, x2)
        # print(test)
        # exit()

        # print(x1.shape)
        # print(x2.shape)
        # print(x)
        # print(x1)
        # print(x2)
        X1.append(x1)
        X2.append(x2)
        Y.append(to_categorical(dataset['qas'][qid]['answer_offsets'][0], num_classes=512))
        # break
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    start_model = build_model(768, 512, 'start')
    hist = start_model.fit([X1, X2], Y, verbose=1, epochs=100)
    # cb.losses
    print(Y)
    y_prob = start_model.predict([X1, X2])
    y_classes = y_prob.argmax(axis=-1)
    print(y_classes)
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



