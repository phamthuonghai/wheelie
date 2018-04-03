from tensor2tensor.models.lstm import lstm_attention_base
from tensor2tensor.utils import registry

__all__ = [
    "lstm_syntax_directed",
           ]


@registry.register_hparams
def lstm_syntax_directed():
    """Hparams for LSTM with Chen's syntax-directed attention."""
    hparams = lstm_attention_base()
    hparams.add_hparam("attention_mechanism", "syntax_directed")
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams
