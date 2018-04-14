import tensorflow as tf
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models.transformer import transformer_ffn_layer
from tensor2tensor.utils import registry, expert_utils
from tensor2tensor.models import transformer

from .attentions import multihead_attention_tree

__all__ = ["TransformerTree"]


@registry.register_model
class TransformerTree(transformer.Transformer):
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
                  encoder-decoder attention. [batch_size, input_length]
        """
        inputs = common_layers.flatten4d3d(inputs)
        relative_tree_distance = features["relative_tree_distance"]

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer.transformer_prepare_encoder(
                inputs, target_space, hparams, features=features))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        encoder_output = transformer_tree_encoder(
            encoder_input, self_attention_bias, relative_tree_distance,
            hparams, nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights)

        return encoder_output, encoder_decoder_attention_bias


def transformer_tree_encoder(encoder_input,
                             encoder_self_attention_bias,
                             relative_tree_distance,
                             hparams,
                             name="encoder",
                             nonpadding=None,
                             save_weights_to=None,
                             make_image_summary=True):
    """A stack of transformer layers.

      Args:
        encoder_input: a Tensor
        encoder_self_attention_bias: bias Tensor for self-attention
           (see common_attention.attention_bias())
        relative_tree_distance:
        hparams: hyperparameters for model
        name: a string
        nonpadding: optional Tensor with shape [batch_size, encoder_length]
          indicating what positions are not padding.  This must either be
          passed in, which we do for "packed" datasets, or inferred from
          encoder_self_attention_bias.  The knowledge about padding is used
          for pad_remover(efficiency) and to mask out padding in convoltutional
          layers.
        save_weights_to: an optional dictionary to capture attention weights
          for vizualization; the weights tensor will be appended there under
          a string key created from the variable scope (including name).
        make_image_summary: Whether to make an attention image summary.

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
                    y = multihead_attention_tree(
                        common_layers.layer_preprocess(x, hparams),
                        None,
                        relative_tree_distance,
                        encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        attention_type=hparams.self_attention_type,
                        save_weights_to=save_weights_to,
                        max_relative_position=hparams.max_relative_position,
                        max_relative_tree_distance=hparams.max_relative_tree_distance,
                        combine_tree_seq_emb=hparams.combine_tree_seq_emb,
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
