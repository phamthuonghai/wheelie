from collections import defaultdict

import tensorflow as tf
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models.transformer import transformer_ffn_layer
from tensor2tensor.utils import registry, expert_utils
from tensor2tensor.models import transformer

__all__ = ["TransformerReservedHeads"]


@registry.register_model
class TransformerReservedHeads(transformer.Transformer):

    def encode(self, inputs, target_space, hparams, features=None):
        """Encode transformer inputs.

        Args:
          inputs: Transformer inputs [batch_size, input_length, input_height,
            hidden_dim] which will be flattened along the two spatial dimensions.
          target_space: scalar, target space ID.
          hparams: hyperparmeters for model.
          features: optionally pass the entire features dictionary as well.
            This is needed now for "packed" datasets.

        Returns:
          Tuple of:
              encoder_output: Encoder representation.
                  [batch_size, input_length, hidden_dim]
              encoder_decoder_attention_bias: Bias and mask weights for
                  encodre-decoder attention. [batch_size, input_length]
        """
        inputs = common_layers.flatten4d3d(inputs)

        other_heads = defaultdict(list)
        if hasattr(hparams, 'pos_head'):
            other_heads[hparams.pos_head].append(common_layers.flatten4d3d(features['pos']))

        if hasattr(hparams, 'deprel_head'):
            other_heads[hparams.deprel_head].append(common_layers.flatten4d3d(features['deprel']))

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer.transformer_prepare_encoder(
                inputs, target_space, hparams, features=features))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        encoder_output = transformer_encoder(
            encoder_input, self_attention_bias,
            hparams, nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights, other_heads=other_heads)

        return encoder_output, encoder_decoder_attention_bias

    def _fast_decode(self, features, decode_length,
                     beam_size=1, top_beams=1, alpha=1.0):
        hparams = self._hparams

        def _transform_in_decode(inputs, input_label):
            inputs_modality = self._problem_hparams.input_modality[input_label]
            with tf.variable_scope(inputs_modality.name):
                inputs = inputs_modality.bottom(inputs)
            return inputs

        if hasattr(hparams, 'pos_head'):
            features['pos'] = _transform_in_decode(features['pos'], 'pos')

        if hasattr(hparams, 'deprel_head'):
            features['deprel'] = _transform_in_decode(features['deprel'], 'deprel')

        return super(TransformerReservedHeads, self)._fast_decode(features, decode_length, beam_size, top_beams, alpha)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        other_heads=None):
    """A stack of transformer layers.

      Args:
        :param encoder_input: a Tensor
        :param encoder_self_attention_bias: bias Tensor for self-attention
           (see common_attention.attention_bias())
        :param hparams: hyperparameters for model
        :param name: a string
        :param nonpadding: optional Tensor with shape [batch_size, encoder_length]
          indicating what positions are not padding.  This must either be
          passed in, which we do for "packed" datasets, or inferred from
          encoder_self_attention_bias.  The knowledge about padding is used
          for pad_remover(efficiency) and to mask out padding in convoltutional
          layers.
        :param save_weights_to: an optional dictionary to capture attention weights
          for vizualization; the weights tensor will be appended there under
          a string key created from the variable scope (including name).
        :param make_image_summary: Whether to make an attention image summary.
        :param other_heads: {0: [pos_head, dep_head...], // other heads on layer 0
                             1: [...]                    // other heads on layer 1
                             ...
                            }

      Returns:
        y: a Tensors
    """
    x = encoder_input
    attention_dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(hparams, "attention_dropout_broadcast_dims", "")))
    with tf.variable_scope(name):
        if nonpadding is not None:
            padding = 1.0 - nonpadding
        else:
            padding = common_attention.attention_bias_to_padding(
                encoder_self_attention_bias)
            nonpadding = 1.0 - padding
        pad_remover = None
        if hparams.use_pad_remover and not common_layers.is_on_tpu():
            pad_remover = expert_utils.PadRemover(padding)
        for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    if other_heads is not None and len(other_heads[layer]) > 0:
                        y = multihead_attention_with_reserved_heads(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            encoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            other_heads[layer],
                            attention_type=hparams.self_attention_type,
                            save_weights_to=save_weights_to,
                            max_relative_position=hparams.max_relative_position,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims)
                    else:
                        y = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            encoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            save_weights_to=save_weights_to,
                            max_relative_position=hparams.max_relative_position,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims)
                    x = common_layers.layer_postprocess(x, y, hparams)
                with tf.variable_scope("ffn"):
                    y = transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams), hparams, pad_remover,
                        conv_padding="SAME", nonpadding_mask=nonpadding)
                    x = common_layers.layer_postprocess(x, y, hparams)
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        return common_layers.layer_preprocess(x, hparams)


def compute_qkv(query_antecedent,
                memory_antecedent,
                other_qks,
                total_key_depth,
                total_value_depth,
                key_head_width,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
    """Computes query, key and value.

      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: and integer
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
        to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

      Returns:
        q, k, v : [batch, length, depth] tensors
      """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent

    def _compute(inp, depth, filter_width, padding, name):
        if filter_width == 1:
            return common_layers.dense(inp, depth, use_bias=False, name=name)
        else:
            return common_layers.conv1d(inp, depth, filter_width, padding, name=name)

    reserved_size = key_head_width * len(other_qks)

    q = _compute(
        query_antecedent, total_key_depth - reserved_size, q_filter_width, q_padding, "q")
    # changed instead of memory_antecedent in original code
    k = _compute(
        query_antecedent, total_key_depth - reserved_size, kv_filter_width, kv_padding, "k")

    for _id, other_head in enumerate(other_qks):
        q = tf.concat([q, _compute(other_head, key_head_width, q_filter_width, q_padding, "q_%d" % _id)], -1)
        k = tf.concat([k, _compute(other_head, key_head_width, kv_filter_width, kv_filter_width, "k_%d" % _id)], -1)

    v = _compute(
        memory_antecedent, total_value_depth, kv_filter_width, kv_padding, "v")
    return q, k, v


def multihead_attention_with_reserved_heads(query_antecedent,
                                            memory_antecedent,
                                            bias,
                                            total_key_depth,
                                            total_value_depth,
                                            output_depth,
                                            num_heads,
                                            dropout_rate,
                                            other_heads,
                                            max_relative_position=None,
                                            image_shapes=None,
                                            attention_type="dot_product",
                                            block_length=128,
                                            block_width=128,
                                            q_filter_width=1,
                                            kv_filter_width=1,
                                            q_padding="VALID",
                                            kv_padding="VALID",
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
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
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
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
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
        q, k, v = compute_qkv(query_antecedent, memory_antecedent, other_heads,
                              total_key_depth, total_value_depth, total_key_depth/num_heads,
                              q_filter_width, kv_filter_width, q_padding, kv_padding)

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
            x = common_attention.dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                                                dropout_rate, image_shapes,
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
        x = common_layers.dense(
            x, output_depth, use_bias=False, name="output_transform")
        if additional_returned_value is not None:
            return x, additional_returned_value
        return x
