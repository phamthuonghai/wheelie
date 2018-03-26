from glob import glob

import tensorflow as tf
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.ops import variable_scope


def _syntax_directed_score(query, keys, att_scores):
    """
    Args:
        query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    Raises:
        ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    max_time = tf.shape(keys)[1]
    dtype = query.dtype
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys.  "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?"
            % (query, depth, keys, key_units, key_units))

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = array_ops.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_time, 1, max_time].
    # we then squeeze out the center singleton dimension.
    global_score = math_ops.matmul(query, keys, transpose_b=True)
    global_score = array_ops.squeeze(global_score, [1], name='global_score')

    align_w = variable_scope.get_variable("align_w", [depth, depth], dtype=dtype)
    align_v = variable_scope.get_variable("align_v", [depth, 1], dtype=dtype)
    x2 = tf.map_fn(lambda e: tf.matmul(tf.tanh(tf.matmul(e, align_w)), align_v), query)
    x2 = tf.squeeze(x2, [1])
    p_i = tf.cast(max_time, tf.float32) * tf.sigmoid(x2)
    mask = tf.one_hot(tf.cast(p_i, tf.int32), depth=max_time)
    mask = tf.squeeze(mask, [1])
    M_i = tf.boolean_mask(att_scores, mask)
    max_time = tf.cast(max_time, tf.float32) ** 2 / 2
    score = tf.multiply(global_score,
                        tf.exp(-tf.divide(tf.square(M_i), max_time), name='syn_score'))
    # TODO: take into account n-gram SDC
    return score


class SyntaxDirectedAttention(LuongAttention):
    """Implements Syntax-directed attention scoring.
    Kehai Chen, RuiWang, Masao Utiyama, Eiichiro Sumita, and Tiejun Zhao
    "Syntax-Directed Attention for Neural Machine Translation."
    AAAI-2018.  http://arxiv.org/abs/1711.04231
    P/S: inherit from LuongAttention because _BaseAttentionMechanism is private
    """
    def __init__(self,
                 num_units,
                 memory,
                 att_scores,
                 memory_sequence_length=None,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="SyntaxDirectedAttention"):
        super(SyntaxDirectedAttention, self).__init__(
            num_units, memory, memory_sequence_length=memory_sequence_length, scale=False,
            probability_fn=probability_fn, score_mask_value=score_mask_value, dtype=dtype, name=name)
        self._att_scores = att_scores

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.
        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "syndir_attention", [query]):
            score = _syntax_directed_score(query, self._keys, self._att_scores)
        alignments = self._probability_fn(score, previous_alignments)
        return alignments
