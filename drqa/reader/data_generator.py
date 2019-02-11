import numpy as np
import keras

''' 
Modified from
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=512, shuffle=True, side='start'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list(dataset['qas'].keys())
        self.dataset = dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.side = side
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, qid in enumerate(list_IDs_temp):
            # Store sample
            cid = dataset['qas'][qid]['cid']
            # X[i,] = np.array([dataset['contexts'][cid]['bert_features'], dataset['qas'][qid]['bert_features'][0]])
            X[i,] = ([dataset['contexts'][cid]['bert_features'], dataset['qas'][qid]['bert_features'][0]])

            # Store class
            if self.side == 'start':
                y[i] = dataset['qas'][qid]['answer_offsets'][0]
            else:
                y[i] = dataset['qas'][qid]['answer_offsets'][1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
