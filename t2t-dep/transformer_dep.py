from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import tensorflow as tf


@registry.register_model
class TransformerDep(transformer.Transformer):

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

        inputs = features["inputs"] + features["pos"] + features["gov"] + features["depth"] + features["sib_ord"]

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

    def _model_fn(self, features, skip=False, force_full_predict=False):
        """Computes the entire model and produces sharded logits and losses.

        Args:
            features: A dictionary of feature name to tensor.
            skip: a Boolean, if we're just dummy-calling and actually skip this model
                (but we need to create variables to not confuse distributed training).
            force_full_predict: a Boolean, if set, then last-position-only
                optimizations are not used even when allowed and in PREDICT mode.

        Returns:
            logits: `Tensor`
            losses: a dictionary: {loss-name (string): floating point `Scalar`}.
        """
        start_time = time.time()
        dp = self._data_parallelism

        features['pos'] = tf.expand_dims(tf.expand_dims(features['pos'], -1), -1)
        features['gov'] = tf.expand_dims(tf.expand_dims(features['gov'], -1), -1)
        features['depth'] = tf.expand_dims(tf.expand_dims(features['depth'], -1), -1)
        features['sib_ord'] = tf.expand_dims(tf.expand_dims(features['sib_ord'], -1), -1)

        sharded_features = self._shard_features(features)

        # Construct the model bottom for inputs.
        transformed_features = {}
        all_previous_modalities = []

        for key, input_modality in six.iteritems(
                self._problem_hparams.input_modality):
            previous_modalities = [
                    self.hparams.problems[i].input_modality[key].name
                    for i in xrange(self._problem_idx)
            ]
            all_previous_modalities.extend(previous_modalities)
            do_reuse = input_modality.name in all_previous_modalities
            transformed_features[key + "_raw"] = sharded_features[key]
            with tf.variable_scope(input_modality.name, reuse=do_reuse):
                transformed_features[key] = input_modality.bottom_sharded(
                        sharded_features[key], dp)
            all_previous_modalities.append(input_modality.name)

        # Target space id just gets copied to every shard.
        if "target_space_id" in features:
            transformed_features["target_space_id"] = [features["target_space_id"]
                                                                                                ] * self._num_datashards

        # For features without a modality ending in "_raw", we pass them raw.
        for key, feature in sharded_features.items():
            if key not in transformed_features and key.endswith("_raw"):
                transformed_features[key] = feature

        # Targets are transformed by the autoregressive part of the modality
        previous_tgt_modalities = [
                self.hparams.problems[i].target_modality.name
                for i in xrange(self._problem_idx)
        ]
        all_previous_modalities.extend(previous_tgt_modalities)

        target_modality = self._problem_hparams.target_modality
        target_reuse = target_modality.name in previous_tgt_modalities
        with tf.variable_scope(target_modality.name, reuse=target_reuse):
            transformed_features["targets"] = target_modality.targets_bottom_sharded(
                    sharded_features["targets"], dp)

        # Allows later access to pre-embedding raw targets.
        transformed_features["targets_raw"] = sharded_features["targets"]

        # Construct the model body.
        with tf.variable_scope("body", reuse=self._problem_idx > 0):
            if skip:
                body_outputs = transformed_features["targets"]
                losses = {"extra": 0.0}
            else:
                body_outputs, losses = self.model_fn_body_sharded(transformed_features)
                if not isinstance(losses, dict):  # If it's a single extra loss.
                    losses = {"extra": losses}

        with tf.variable_scope(target_modality.name, reuse=target_reuse):
            last_only = (target_modality.top_is_pointwise and
                                     self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
                                     not force_full_predict)
            if not last_only:
                sharded_logits = target_modality.top_sharded(
                        body_outputs, sharded_features["targets"], dp)
                training_loss = target_modality.loss_sharded(
                        sharded_logits, sharded_features["targets"], dp)

                training_loss *= self._problem_hparams.loss_multiplier
            else:
                # Take body outputs for the last position only, and targets too.
                last_position_body_outputs = [
                        tf.expand_dims(body_shard[:, -1, :, :], axis=[1])
                        for body_shard in body_outputs
                ]
                last_position_targets = [
                        tf.expand_dims(target_shard[:, -1:, :, :], axis=[1])
                        for target_shard in sharded_features["targets"]
                ]
                sharded_logits = target_modality.top_sharded(last_position_body_outputs,
                                                                                                         last_position_targets,
                                                                                                         self._data_parallelism)
                training_loss = None
        losses["training"] = training_loss

        # Scheduled sampling.
        do_scheduled_sampling = (  # Only do it if training and set for it.
                self.hparams.scheduled_sampling_prob > 0.0 and
                self.hparams.mode == tf.estimator.ModeKeys.TRAIN and not skip)
        if do_scheduled_sampling:

            def sample(x):
                """Multinomial sampling from a n-dimensional tensor."""
                vocab_size = target_modality.top_dimensionality
                samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]), 1)
                reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
                return tf.to_int32(reshaped_samples)

            def mix_gold_sampled(gold_targets, sampled_targets):
                return tf.where(
                        tf.less(
                                tf.random_uniform(common_layers.shape_list(sampled_targets)),
                                self.hparams.scheduled_sampling_gold_mixin_prob), gold_targets,
                        sampled_targets)

            def sampled_results():
                """Generate scheduled sampling results."""
                sampled_targets = dp(sample, sharded_logits)
                new_targets = dp(mix_gold_sampled, sharded_features["targets"],
                                                 sampled_targets)
                new_features = transformed_features
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.variable_scope(target_modality.name):
                        new_features["targets"] = target_modality.targets_bottom_sharded(
                                new_targets, dp)
                    with tf.variable_scope("body"):
                        body_outputs, losses = self.model_fn_body_sharded(new_features)
                        if not isinstance(losses, dict):  # If it's a single extra loss.
                            losses = {"extra": losses}
                    with tf.variable_scope(target_modality.name):
                        new_sharded_logits = target_modality.top_sharded(
                                body_outputs, sharded_features["targets"], dp)
                        training_loss = target_modality.loss_sharded(
                                sharded_logits, sharded_features["targets"], dp)
                        training_loss *= self._problem_hparams.loss_multiplier
                    losses["training"] = training_loss
                return new_sharded_logits, losses

            # Run the above conditionally.
            prob = self.hparams.scheduled_sampling_prob
            prob *= common_layers.inverse_exp_decay(
                    self.hparams.scheduled_sampling_warmup_steps, min_value=0.001)
            sharded_logits, losses = tf.cond(
                    tf.less(tf.random_uniform([]), prob), sampled_results,
                    lambda: (sharded_logits, losses))

        if not context.in_eager_mode():
            tf.logging.info("This model_fn took %.3f sec." %
                                            (time.time() - start_time))
        return sharded_logits, losses
