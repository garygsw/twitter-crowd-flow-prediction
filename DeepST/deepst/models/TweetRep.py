from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer
from keras.layers import Reshape, Lambda
import numpy as np


class TweetRep(Layer):
    def __init__(self, vocab_size, embedding_size, map_height, map_width, len_seq,
                 seq_size, sum_type='simple', initial_weights=None,
                 train_embeddings=True, reduce_index_dims=False, **kwargs):
        super(TweetRep, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initial_weights = initial_weights
        self.train_embeddings = train_embeddings
        self.len_seq = len_seq
        self.seq_size = seq_size
        self.map_height = map_height
        self.map_width = map_width
        self.reduce_index_dims = reduce_index_dims
        self.sum_type = sum_type

    def build(self, input_shape):
        if self.initial_weights is None:
            self.initial_weights = np.random.random((self.vocab_size + 1,
                                                     self.embedding_size))
            self.initial_weights[0] = [0] * self.embedding_size
        self.embeddings = K.variable(self.initial_weights)
        if self.train_embeddings:
            self.trainable_weights = [self.embeddings]
        else:
            self.trainable_weights = []

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')          # convert index to integers
        W = K.gather(self.embeddings, x)    # get vectors at index of embeddings
        # W: (?*l*h*w, n, k)
        W = K.reshape(W, (-1,
                          self.len_seq,
                          self.map_height,
                          self.map_width,
                          self.seq_size,
                          self.embedding_size))
        # W: (?, l, h, w, n, k)
        if self.sum_type == 'weighted':
            V = K.sum(W, axis=-2, keepdims=True)   # sum vectors for all words per grid
            # V: (?, l, h, w, 1, k)
            b = W * V
            # b: (?, l, h, w, n, k)
            b = K.sum(b, axis=-1)
            # b: (?, l, h, w, n)
            c = softmax(b, axis=-1)  # to make sum of all weights to sum up to 1
            # c: (?, l, h, w, n)
            c = K.expand_dims(c, axis=-1)
            # c: (?, l, h, w, n, 1)
            V = W * c
            # W: (?, l, h, w, n, k)
            V = K.sum(V, axis=-2)  # weighted sum
            # W: (?, l, h, w, k)
        elif self.sum_type == 'simple':
            V = K.sum(W, axis=-2, keepdims=False)   # sum vectors for all words per grid
            # V: (?, l, h, w, k)
        else:
            raise Exception('invalid sum type ' + str(self.sum_type))
        if not self.reduce_index_dims:
            inverted_output = K.permute_dimensions(V, (0, 1, 4, 2, 3))
            # inverted_output: (?, l, k, h, w)
            output = K.reshape(inverted_output, (-1,
                                                 self.embedding_size * self.len_seq,
                                                 self.map_height,
                                                 self.map_width))
            # output: (?, l*k, h, w)
        else:
            inverted_output = K.permute_dimensions(V, (0, 2, 3, 1, 4))
            # inverted_output: (?, h, w, l, k)
            output = K.reshape(inverted_output, (-1,
                                                 self.map_height,
                                                 self.map_width,
                                                 self.embedding_size * self.len_seq))
            # output: (?, h, w, l*k)
        return output

    def compute_output_shape(self, input_shape):
        if not self.reduce_index_dims:
            return (input_shape[0],) + (self.embedding_size * self.len_seq,
                self.map_height, self.map_width)
        else:
            return (input_shape[0],) + (self.map_height, self.map_width,
                self.embedding_size * self.len_seq)
