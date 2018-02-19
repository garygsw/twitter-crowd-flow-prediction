from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer
from keras.layers import Reshape, Lambda
import numpy as np


class TweetRep(Layer):
    def __init__(self, vocab_size, embedding_size, map_height, map_width, len_seq, seq_size,
                 initial_weights=None, train_embeddings=True, **kwargs):
        super(TweetRep, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initial_weights = initial_weights
        self.train_embeddings = train_embeddings
        self.len_seq = len_seq
        self.seq_size = seq_size
        self.map_height = map_height
        self.map_width = map_width
        # self.batch_size = batch_size

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
        #print 'x shape:', K.shape(x)
        W = K.gather(self.embeddings, x)    # get vectors at index of embeddings
        #print 'W shape:', K.shape(W)
        # W: (?, l*h*w, n, k)
        # shape = K.shape(W)
        # print 'w shape:', shape
        # print shape[0]
        # print shape[0].value
        W = K.reshape(W, (-1,
                          self.len_seq,
                          self.map_height,
                          self.map_width,
                          self.seq_size,
                          self.embedding_size))
        #print('input shape: ', W.shape)
        #print('requested shape: ', (self.len_seq, self.map_height, self.map_width, self.seq_size, self.embedding_size))
        #print('requested shape size: ', self.len_seq * self.map_height * self.map_width * self.seq_size * self.embedding_size)
        # W = Reshape((self.len_seq,
        #              self.map_height,
        #              self.map_width,
        #              self.seq_size,
        #              self.embedding_size))(W)
        #print 'W (reshaped) shape:', K.shape(W)
        # W: (?, l, h, w, n, k)
        V = K.sum(W, axis=-2, keepdims=True)              # sum vectors for all words per grid
        #print 'V shape:', K.shape(V)
        # V: (?, l, h, w, k)
        #V = K.expand_dims(V, axis=-2)
        #V = K.repeat_elements(V, rep=1, axis=-1)
        b = W * V
        #print 'b shape:', K.shape(b)
        # b: (?, l, h, w, n, k)
        b = K.sum(b, axis=-1)
        #print 'b (summed) shape:', K.shape(b)
        # b: (?, l, h, w, n)

        c = softmax(b, axis=-1)  # to make sum of all weights to sum up to 1
        #print 'c shape:', K.shape(c)
        # c: (?, l, h, w, n)
        c = K.expand_dims(c, axis=-1)
        #print 'c (expanded) shape:', K.shape(c)
        #c = K.repeat_elements(c, rep=1, axis=-1)
        weighted_S = W * c
        #print 'weighted_S shape:', K.shape(weighted_S)
        # weighted_S: (?, l, h, w, n, k)
        weighted_S = K.sum(weighted_S, axis=-2)  # weighted sum
        # weighted_S: (?, l, h, w, k)
        #print 'weighted_S shape (summed):', K.shape(weighted_S)
        inverted_output = K.permute_dimensions(weighted_S, (0, 1, 4, 2, 3))
        #print 'inverted output shape:', K.shape(inverted_output)
        #inverted_output = K.permute_dimensions(weighted_S, (0, 3, 1, 2))
        # inverted_output: (?, l, k, h, w)
        #shape = K.shape(inverted_output)
        # stack up the sequences
        # output = K.reshape(inverted_output,
        #                    (shape[0], shape[1] * shape[2], shape[3], shape[4]))
        # batch, seq * k, h, w
        #print shape[0]
        output = K.reshape(inverted_output, (-1,
                                             self.embedding_size * self.len_seq,
                                             self.map_height,
                                             self.map_width))
        # output = Reshape((self.embedding_size * self.len_seq,
        #                   self.map_height,
        #                   self.map_width))(inverted_output)
        return output

    def compute_output_shape(self, input_shape):
        # return (input_shape[0],) + (input_shape[1] * self.embedding_size, ) + \
        #     tuple(input_shape[2:4])
        return (input_shape[0],) + (self.embedding_size * self.len_seq,
            self.map_height, self.map_width)
