import copy

import tensorflow as tf
from tensor2tensor.layers import common_layers
from tensor2tensor.models.lstm import lstm_bid_encoder
from tensor2tensor.utils import registry, t2t_model
from tensorflow.contrib.seq2seq import AttentionWrapper

from .attentions import SyntaxDirectedAttention

__all__ = ["LSTMSyntaxDirectedAttention"]


def lstm_syndir_attention_decoder(inputs, hparams, train, name, initial_state, encoder_outputs, att_scores):
    """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
    attention_mechanism = SyntaxDirectedAttention(hparams.hidden_size, encoder_outputs, att_scores)

    cell = AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism] * hparams.num_heads,
        attention_layer_size=[hparams.attention_layer_size] * hparams.num_heads,
        output_attention=(hparams.output_attention == 1))

    batch_size = common_layers.shape_list(inputs)[0]

    initial_state = cell.zero_state(batch_size, tf.float32).clone(
        cell_state=initial_state)

    with tf.variable_scope(name):
        output, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)

        # For multi-head attention project output back to hidden size
        if hparams.output_attention == 1 and hparams.num_heads > 1:
            output = tf.layers.dense(output, hparams.hidden_size)

        return output, state


def lstm_seq2seq_internal_syndir_attention_bid_encoder(inputs, targets, att_scores, hparams, train):
    """LSTM seq2seq model with attention, main step used for training."""
    with tf.variable_scope("lstm_seq2seq_internal_syndir_attention_bid_encoder"):
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)

        # LSTM encoder.
        encoder_outputs, final_encoder_state = lstm_bid_encoder(
            tf.reverse(inputs, axis=[1]), hparams, train, "encoder")

        # LSTM decoder with attention
        shifted_targets = common_layers.shift_right(targets)
        hparams_decoder = copy.copy(hparams)
        hparams_decoder.hidden_size = 2 * hparams.hidden_size
        decoder_outputs, _ = lstm_syndir_attention_decoder(
            common_layers.flatten4d3d(shifted_targets), hparams_decoder, train,
            "decoder", final_encoder_state, encoder_outputs, att_scores)
        return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class LSTMSyntaxDirectedAttention(t2t_model.T2TModel):
    def body(self, features):
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        return lstm_seq2seq_internal_syndir_attention_bid_encoder(
            features.get("inputs"), features["targets"], features["relative_tree_distance"], self._hparams, train)
