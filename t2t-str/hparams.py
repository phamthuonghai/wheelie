from tensor2tensor.models.lstm import lstm_attention_base
from tensor2tensor.models.transformer import transformer_relative
from tensor2tensor.utils import registry

__all__ = [
    "lstm_syntax_directed",
    "transformer_relative_tree_20_seq",
    "transformer_relative_tree_20",
    "transformer_relative_tree_5",
]


@registry.register_hparams
def lstm_syntax_directed():
    """Hparams for LSTM with Chen's syntax-directed attention."""
    hparams = lstm_attention_base()
    hparams.add_hparam("attention_mechanism", "syntax_directed")
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams


@registry.register_hparams
def transformer_relative_tree_20_seq():
    """Use tree+sequential relative position embeddings instead of absolute sequential position encodings."""
    hparams = transformer_relative()
    hparams.combine_tree_seq_emb = True
    hparams.max_relative_tree_distance = 20
    return hparams


@registry.register_hparams
def transformer_relative_tree_20():
    """Use tree relative position embeddings instead of relative/absolute sequential position encodings."""
    hparams = transformer_relative()
    hparams.combine_tree_seq_emb = False
    hparams.max_relative_tree_distance = 20
    return hparams


@registry.register_hparams
def transformer_relative_tree_5():
    """Use tree relative position embeddings instead of relative/absolute sequential position encodings."""
    hparams = transformer_relative_tree_20()
    hparams.combine_tree_seq_emb = False
    hparams.max_relative_tree_distance = 5
    return hparams
