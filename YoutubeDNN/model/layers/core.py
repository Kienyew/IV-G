import numpy as np
import tensorflow as tf
from deepctr.layers.utils import reduce_max, reduce_mean, reduce_sum, concat_func
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer


class PoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, sampler_config, temperature=1.0, **kwargs):
        self.sampler_config = sampler_config
        self.temperature = temperature
        self.sampler = self.sampler_config['sampler']
        self.item_count = self.sampler_config['item_count']

        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vocabulary_size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.vocabulary_size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        item_embeddings, user_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature
        if self.sampler == "inbatch":
            item_vec = tf.gather(item_embeddings, tf.squeeze(item_idx, axis=1))
            logits = tf.matmul(user_vec, item_vec, transpose_b=True)
            loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)

        else:
            num_sampled = self.sampler_config['num_sampled']
            if self.sampler == "frequency":
                sampled_values = tf.nn.fixed_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                       self.vocabulary_size,
                                                                       distortion=self.sampler_config['distortion'],
                                                                       unigrams=np.maximum(self.item_count, 1).tolist(),
                                                                       seed=None,
                                                                       name=None)
            elif self.sampler == "adaptive":
                sampled_values = tf.nn.learned_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                         self.vocabulary_size, seed=None, name=None)
            elif self.sampler == "uniform":
                try:
                    sampled_values = tf.nn.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                     self.vocabulary_size, seed=None, name=None)
                except AttributeError:
                    sampled_values = tf.random.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
                                                                         self.vocabulary_size, seed=None, name=None)
            else:
                raise ValueError(' `%s` sampler is not supported ' % self.sampler)

            loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,
                                              biases=self.zero_bias,
                                              labels=item_idx,
                                              inputs=user_vec,
                                              num_sampled=num_sampled,
                                              num_classes=self.vocabulary_size,
                                              sampled_values=sampled_values
                                              )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def inbatch_softmax_cross_entropy_with_logits(logits, item_count, item_idx):
    Q = tf.gather(tf.constant(item_count / np.sum(item_count), 'float32'),
                  tf.squeeze(item_idx, axis=1))
    try:
        logQ = tf.reshape(tf.math.log(Q), (1, -1))
        logits -= logQ  # subtract_log_q
        labels = tf.linalg.diag(tf.ones_like(logits[0]))
    except AttributeError:
        logQ = tf.reshape(tf.log(Q), (1, -1))
        logits -= logQ  # subtract_log_q
        labels = tf.diag(tf.ones_like(logits[0]))

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return loss


class EmbeddingIndex(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
