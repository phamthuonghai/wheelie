import tensorflow as tf
from tensor2tensor.layers import common_attention, common_layers
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
          previous_alignments: Tensor of dtype matching `self.values` and shape
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


def multihead_attention_tree(query_antecedent,
                             memory_antecedent,
                             relative_tree_distance,
                             bias,
                             total_key_depth,
                             total_value_depth,
                             output_depth,
                             num_heads,
                             dropout_rate,
                             max_relative_position=None,
                             max_relative_tree_distance=None,
                             combine_tree_seq_emb=None,
                             image_shapes=None,
                             attention_type="dot_product",
                             block_length=128,
                             block_width=128,
                             q_filter_width=1,
                             kv_filter_width=1,
                             q_padding="VALID",
                             kv_padding="VALID",
                             cache=None,
                             gap_size=0,
                             num_memory_blocks=2,
                             name=None,
                             save_weights_to=None,
                             make_image_summary=True,
                             dropout_broadcast_dims=None,
                             **kwargs):
    """Multihead scaled-dot-product attention with input/output transformations.

      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
        relative_tree_distance:
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        dropout_rate: a floating point number
        max_relative_position: Maximum distance between inputs to generate
                               unique relation embeddings for. Only relevant
                               when using "dot_product_relative" attention.
        max_relative_tree_distance:
        combine_tree_seq_emb:
        image_shapes: optional tuple of integer scalars.
                      see comments for attention_image_summary()
        attention_type: a string, either "dot_product", "dot_product_relative",
                        "local_mask_right", "local_unmasked", "masked_dilated_1d",
                        "unmasked_dilated_1d" or any attention function with the
                        signature (query, key, value, **kwargs)
        block_length: an integer - relevant for "local_mask_right"
        block_width: an integer - relevant for "local_unmasked"
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
                         to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
                   no padding.
        cache: dict containing Tensors which are the results of previous
               attentions, used for fast decoding. Expects the dict to contrain two
               keys ('k' and 'v'), for the initial call the values for these keys
               should be empty Tensors of the appropriate shape.
                   'k' [batch_size, 0, key_channels]
                   'v' [batch_size, 0, value_channels]
        gap_size: Integer option for dilated attention to indicate spacing between
                  memory blocks.
        num_memory_blocks: Integer option to indicate how many memory blocks to look
                           at.
        name: an optional string.
        save_weights_to: an optional dictionary to capture attention weights
          for vizualization; the weights tensor will be appended there under
          a string key created from the variable scope (including name).
        make_image_summary: Whether to make an attention image summary.
        dropout_broadcast_dims:  an optional list of integers less than 4
          specifying in which dimensions to broadcast the dropout decisions.
          saves memory.
        **kwargs (dict): Parameters for the attention function

      Caching:
        WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
        the caching assumes that the bias contains future masking.

        The caching works by saving all the previous key and value values so that
        you are able to send just the last query location to this attention
        function. I.e. if the cache dict is provided it assumes the query is of the
        shape [batch_size, 1, hiddem_dim] rather than the full memory.

      Returns:
        The result of the attention transformation. The output shape is
            [batch_size, length_q, hidden_dim]
        unless the cache dict is provided in which case only the last memory
        position is calculated and the output shape is [batch_size, 1, hidden_dim]
        Optionaly returns an additional loss parameters (ex: load balance loss for
        the experts) returned by the attention_type function.

      Raises:
        ValueError: if the key depth or value depth are not divisible by the
          number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    with tf.variable_scope(
            name,
            default_name="multihead_attention",
            values=[query_antecedent, memory_antecedent]):
        q, k, v = common_attention.compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                                               total_value_depth, q_filter_width, kv_filter_width,
                                               q_padding, kv_padding)

        if cache is not None:
            if attention_type != "dot_product":
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = common_attention.split_heads(q, num_heads)
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head ** -0.5

        additional_returned_value = None
        if callable(attention_type):  # Generic way to extend multihead_attention
            x = attention_type(q, k, v, **kwargs)
            if isinstance(x, tuple):
                x, additional_returned_value = x  # Unpack
        elif attention_type == "dot_product":
            x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                                       save_weights_to=save_weights_to,
                                                       make_image_summary=make_image_summary,
                                                       dropout_broadcast_dims=dropout_broadcast_dims)
        elif attention_type == "dot_product_relative":
            x = dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                               relative_tree_distance, max_relative_tree_distance,
                                               dropout_rate, image_shapes, combine_tree_seq_emb=combine_tree_seq_emb,
                                               make_image_summary=make_image_summary)
        elif attention_type == "local_within_block_mask_right":
            x = common_attention.masked_within_block_local_attention_1d(q, k, v,
                                                                        block_length=block_length)
        elif attention_type == "local_mask_right":
            x = common_attention.masked_local_attention_1d(q, k, v, block_length=block_length,
                                                           make_image_summary=make_image_summary)
        elif attention_type == "local_unmasked":
            x = common_attention.local_attention_1d(
                q, k, v, block_length=block_length, filter_width=block_width)
        elif attention_type == "masked_dilated_1d":
            x = common_attention.masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                                                  gap_size, num_memory_blocks)
        else:
            assert attention_type == "unmasked_dilated_1d"
            x = common_attention.dilated_self_attention_1d(q, k, v, block_length, block_width,
                                                           gap_size, num_memory_blocks)
        x = common_attention.combine_heads(x)
        x = common_attention.common_layers.dense(
            x, output_depth, use_bias=False, name="output_transform")
        if additional_returned_value is not None:
            return x, additional_returned_value
        return x


def _generate_tree_relative_positions_matrix(relative_tree_distance, max_relative_tree_distance):
    return tf.clip_by_value(tf.cast(relative_tree_distance, dtype=tf.int32), 0, max_relative_tree_distance)


def _generate_tree_relative_positions_embeddings(relative_tree_distance,
                                                 length,
                                                 depth,
                                                 max_relative_position,
                                                 max_relative_tree_distance,
                                                 combine_tree_seq_emb,
                                                 name):
    """Generates tensor of size [batch_size, length, length, depth]."""
    with tf.variable_scope(name):
        tree_relative_positions_matrix = _generate_tree_relative_positions_matrix(
            relative_tree_distance, max_relative_tree_distance)
        tree_vocab_size = max_relative_tree_distance + 1
        tree_embeddings_table = tf.get_variable("tree_embeddings", [tree_vocab_size, depth])
        embeddings = tf.gather(tree_embeddings_table, tree_relative_positions_matrix)

        if combine_tree_seq_emb:
            seq_relative_positions_matrix = common_attention._generate_relative_positions_matrix(
                length, max_relative_position)
            seq_vocab_size = 2 * max_relative_position + 1
            seq_embeddings_table = tf.get_variable("seq_embeddings", [seq_vocab_size, depth])
            embeddings = embeddings + tf.gather(seq_embeddings_table, seq_relative_positions_matrix)

        return embeddings


def _relative_tree_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.

      This batches matrix multiply calculations to avoid unnecessary broadcasting.

      Args:
        x: Tensor with shape [batch_size, heads, length, length or depth].
        y: Tensor with shape [batch_size, heads, length, depth].
        z: Tensor with shape [batch_size, length, length, depth].
        transpose: Whether to transpose inner matrices of y and z. Should be true if
            last dimension of x is depth, not length.

      Returns:
        A Tensor with shape [batch_size, heads, length, length or depth].
    """
    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    # x_t is [batch_size, length, heads, length or depth]
    x_t = tf.transpose(x, [0, 2, 1, 3])
    # x_t_matmul is [batch_size, length, heads, length or depth]
    x_t_matmul = tf.matmul(x_t, z, transpose_b=transpose)
    # x_t_matmul_t is [batch_size, heads, length, length or depth]
    x_t_matmul_t = tf.transpose(x_t_matmul, [0, 2, 1, 3])
    return xy_matmul + x_t_matmul_t


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   relative_tree_distance=None,
                                   max_relative_tree_distance=None,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   combine_tree_seq_emb=False,
                                   name=None,
                                   make_image_summary=True):
    """Calculate relative position-aware dot-product self-attention.

      The attention calculation is augmented with learned representations for the
      relative position between each element in q and each element in k and v.

      Args:
        q: a Tensor with shape [batch, heads, length, depth].
        k: a Tensor with shape [batch, heads, length, depth].
        v: a Tensor with shape [batch, heads, length, depth].
        bias: bias Tensor.
        max_relative_position: an integer specifying the maxmimum distance between
            inputs that unique position embeddings should be learned for.
        relative_tree_distance:
        max_relative_tree_distance:
        dropout_rate: a floating point number.
        image_shapes: optional tuple of integer scalars.
        combine_tree_seq_emb:
        name: an optional string.
        make_image_summary: Whether to make an attention image summary.

      Returns:
        A Tensor.

      Raises:
        ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % max_relative_position)
    with tf.variable_scope(
            name, default_name="dot_product_attention_relative", values=[q, k, v]):

        # This calculation only works for self attention.
        # q, k and v must therefore have the same shape.
        q.get_shape().assert_is_compatible_with(k.get_shape())
        q.get_shape().assert_is_compatible_with(v.get_shape())

        # Use separate embeddings suitable for keys and values.
        depth = q.get_shape().as_list()[3]
        length = common_layers.shape_list(q)[2]
        relations_keys = _generate_tree_relative_positions_embeddings(
            relative_tree_distance, length, depth, max_relative_position,
            max_relative_tree_distance, combine_tree_seq_emb, "relative_positions_keys")
        relations_values = _generate_tree_relative_positions_embeddings(
            relative_tree_distance, length, depth, max_relative_position,
            max_relative_tree_distance, combine_tree_seq_emb, "relative_positions_values")

        # Compute self attention considering the relative position embeddings.
        logits = _relative_tree_attention_inner(q, k, relations_keys, True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        if not tf.get_variable_scope().reuse and make_image_summary:
            common_attention.attention_image_summary(weights, image_shapes)
        return _relative_tree_attention_inner(weights, v, relations_values, False)
