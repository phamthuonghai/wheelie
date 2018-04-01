from tensor2tensor.models.lstm import lstm_attention_base, lstm_luong_attention, lstm_bahdanau_attention
from tensor2tensor.models.transformer import transformer_big_single_gpu, transformer_base_single_gpu
from tensor2tensor.utils import registry

__all__ = ["lstm_syntax_directed",
           "czeng_lstm_luong_attention",
           "czeng_lstm_bahdanau_attention",
           "czeng_transformer_big_single_gpu",
           "czeng_transformer_base_single_gpu"
           ]


@registry.register_hparams
def lstm_syntax_directed():
    """Hparams for LSTM with Chen's syntax-directed attention."""
    hparams = lstm_attention_base()
    hparams.add_hparam("attention_mechanism", "syntax_directed")
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams


@registry.register_hparams
def czeng_lstm_luong_attention():
    """Hparams for LSTM with luong attention."""
    hparams = lstm_luong_attention()
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams


@registry.register_hparams
def czeng_lstm_bahdanau_attention():
    """Hparams for LSTM with bahdanau attention."""
    hparams = lstm_bahdanau_attention()
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams

@registry.register_hparams
def czeng_transformer_big_single_gpu():
    """Hparams for Transformer."""
    hparams = transformer_big_single_gpu()
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams

@registry.register_hparams
def czeng_transformer_base_single_gpu():
    """Hparams for Transformer."""
    hparams = transformer_base_single_gpu()
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams
