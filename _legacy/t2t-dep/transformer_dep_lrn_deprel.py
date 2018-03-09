from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import tensorflow as tf

from . import transformer_dep_lrn


@registry.register_model
class TransformerDepLrnDeprel(transformer_dep_lrn.TransformerDepLrn):

    def model_fn_body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
              "targets": Target decoder outputs.
                  [batch_size, decoder_length, hidden_dim]
              "target_space_id"

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """

        hparams = self._hparams

        max_lens = tf.reduce_max(features["dep_pos"], axis=-2, keep_dims=True, name="deprel_norm_max")
        features["dep_pos"] = tf.divide(features["dep_pos"], max_lens, name="deprel_norm_div")

        inputs = features["inputs"] + tf.expand_dims(
            tf.layers.dense(features["dep_pos"], hparams.hidden_size, name="learned_dep_pos"), -2)

        target_space = features["target_space_id"]
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, target_space, hparams)

        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)

        decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(
            targets, hparams)

        return self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams)
