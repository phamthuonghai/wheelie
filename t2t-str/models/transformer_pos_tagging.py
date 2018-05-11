import tensorflow as tf
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models.transformer import transformer_prepare_decoder, features_to_nonpadding
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer


__all__ = ["TransformerPosTagging"]


@registry.register_model
class TransformerPosTagging(transformer.Transformer):
    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs [batch_size, input_length, hidden_dim]

        Returns:
          "targets": Final decoder representation. [batch_size, decoder_length, hidden_dim]
          "target_pos": Source side POS representation. [batch_size, input_length, hidden_dim]
        """
        hparams = self._hparams

        if self.has_input:
            inputs = features["inputs"]
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(
                inputs, target_space, hparams, features=features)
        else:
            encoder_output, encoder_decoder_attention_bias = (None, None)

        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)

        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            targets, hparams, features=features)

        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "targets"))

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        encoder_output = tf.expand_dims(encoder_output, 2)

        return {
            "targets": decoder_output,
            "target_pos": encoder_output,
        }
