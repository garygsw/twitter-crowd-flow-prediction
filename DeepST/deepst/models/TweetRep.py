from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer
import numpy as np


class TweetRep(Layer):
    def __init__(self, vocab_size, embedding_size, initial_weights=None, **kwargs):
        super(TweetRep, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initial_weights = initial_weights

    def build(self, input_shape):
        if self.initial_weights is None:
            self.initial_weights = np.random.random((self.vocab_size,
                                                     self.embedding_size))
        self.embeddings = K.variable(self.initial_weights)
        self.trainable_weights = [self.embeddings]

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')
        W = K.gather(self.embeddings, x)    # get vectors at index of embeddings
        V = K.sum(W, axis=-2,)              # sum vectors for all words per grid
        V = K.expand_dims(V, axis=-2)
        V = K.repeat_elements(V, rep=1, axis=-1)
        b = W * V
        b = K.sum(b, axis=-1)
        c = softmax(b, axis=-1)  # to make sum of all weights to sum up to 1
        c = K.expand_dims(c, axis=-1)
        c = K.repeat_elements(c, rep=1, axis=-1)
        weighted_S = W * c
        weighted_S = K.sum(weighted_S, axis=-2)  # weighted sum

        inverted_output = K.permute_dimensions(weighted_S, (0, 1, 4, 2, 3))
        # batch, seq, k, h, w
        shape = K.shape(inverted_output)

        # stack up the sequences
        output = K.reshape(inverted_output,
                           (shape[0], shape[1] * shape[2], shape[3], shape[4]))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1] * self.embedding_size, ) + \
            tuple(input_shape[2:4])
