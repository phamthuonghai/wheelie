import tensorflow as tf
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models.transformer import transformer_prepare_decoder, features_to_nonpadding
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

from . import transformer_pos_tagging

__all__ = ["TransformerDepParse"]


@registry.register_model
class TransformerDepParse(transformer_pos_tagging.TransformerPosTagging):
    @property
    def second_goal(self):
        return 'dephead'

    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs [batch_size, input_length, hidden_dim]

        Returns:
          "targets": Final decoder representation. [batch_size, decoder_length, hidden_dim]
          "target_dephead": Source side DepHead representation. [batch_size, input_length, hidden_dim]
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

        return {
            "targets": decoder_output,
            "target_" + self.second_goal: self.parsing(),
        }

    def parsing(self):
        att_weight = self.attention_weights['transformer_dep_parse/body/encoder/layer_%d/'
                                            'self_attention/multihead_attention/%s_attention' % (
                                                self.hparams.parse_from_layer, self.hparams.self_attention_type)]
        return tf.squeeze(tf.split(att_weight, [self.hparams.num_heads-1, 1], axis=1)[1], axis=1)

    def _fast_decode(self, features, decode_length, beam_size=1, top_beams=1, alpha=1.0):
        """Fast decoding.

        Implements both greedy and beam search decoding, uses beam search iff
        beam_size > 1, otherwise beam search related arguments are ignored.

        Args:
          features: a map of string to model  features.
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for slonger translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams

        if isinstance(self._problem_hparams.target_modality, dict):
            target_modality = self._problem_hparams.target_modality['targets']
        else:
            target_modality = self._problem_hparams.target_modality

        if self.has_input:
            inputs = features["inputs"]

            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = common_layers.shape_list(inputs)[1] + decode_length

            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.input_modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode, inputs, features["target_space_id"], hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            # In this case, features["inputs"] contains partial targets.
            # We force the outputs to begin with these sequences.
            encoder_output = None
            encoder_decoder_attention_bias = None
            if len(features["inputs"].shape) >= 4:
                partial_targets = tf.squeeze(tf.to_int64(features["inputs"]), [2, 3])
            else:
                partial_targets = tf.squeeze(tf.to_int64(features["inputs"]), [2])
            partial_targets_length = common_layers.shape_list(partial_targets)[1]
            decode_length += partial_targets_length
            batch_size = tf.shape(partial_targets)[0]

        if hparams.pos == "timing":
            timing_signal = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.

            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name + '/targets'):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if hparams.pos == "timing":
                targets += timing_signal[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode, targets, cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias, hparams, cache,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope('targets/' + target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(tf.tile(partial_targets[:, i], [beam_size]),
                                      vocab_size, 0.0, -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = transformer.fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size)
        if partial_targets is not None:
            if beam_size <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]

        modality_2nd = self._problem_hparams.target_modality['target_' + self.second_goal]
        output_2nd = self.parsing()
        with tf.variable_scope('target_' + self.second_goal + '/' + modality_2nd.name):
            logits_2nd = modality_2nd.top_sharded(output_2nd, None, dp)[0]
            ids_2nd = tf.argmax(logits_2nd, axis=-1)
            ret['output_' + self.second_goal] = ids_2nd

        return ret
