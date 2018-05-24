from tensor2tensor.models.lstm import lstm_attention_base
from tensor2tensor.models.transformer import transformer_relative, transformer_base
from tensor2tensor.utils import registry


@registry.register_hparams
def lstm_syntax_directed():
    """Hparams for LSTM with Chen's syntax-directed attention."""
    hparams = lstm_attention_base()
    hparams.add_hparam("attention_mechanism", "syntax_directed")
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance:symbol:relative_tree_distance"
    return hparams


@registry.register_hparams
def transformer_relative_tree_traversal():
    """Use tree relative traversal embeddings instead of relative/absolute sequential position encodings."""
    hparams = transformer_relative()
    hparams.input_modalities = "inputs:symbol:default;tree_traversal_str:symbol:relative_tree_distance_str"
    hparams.remove_redundant_modalities = True
    hparams.relative_tree_input = 'tree_traversal_str'
    hparams.combine_tree_seq_emb = False
    hparams.max_relative_tree_distance = 20  # dummy
    return hparams


@registry.register_hparams
def transformer_relative_tree_traversal_seq():
    """Use tree relative traversal+sequential relative position embeddings."""
    hparams = transformer_relative_tree_traversal()
    hparams.combine_tree_seq_emb = True
    return hparams


@registry.register_hparams
def transformer_relative_tree_20():
    """Use tree relative position embeddings instead of relative/absolute sequential position encodings."""
    hparams = transformer_relative()
    hparams.relative_tree_input = 'relative_tree_distance_str'
    hparams.combine_tree_seq_emb = False
    hparams.max_relative_tree_distance = 20
    return hparams


@registry.register_hparams
def transformer_relative_tree_20_seq():
    """Use tree+sequential relative position embeddings instead of absolute sequential position encodings."""
    hparams = transformer_relative_tree_20()
    hparams.input_modalities = "inputs:symbol:default;relative_tree_distance_str:symbol:relative_tree_distance_str"
    hparams.remove_redundant_modalities = True
    hparams.combine_tree_seq_emb = True
    return hparams


@registry.register_hparams
def transformer_relative_tree_5():
    """Use tree relative position embeddings instead of relative/absolute sequential position encodings."""
    hparams = transformer_relative_tree_20()
    hparams.max_relative_tree_distance = 5
    return hparams


@registry.register_hparams
def transformer_relative_tree_5_absseq():
    """Use tree relative position embeddings with absolute sequential position encodings."""
    hparams = transformer_relative_tree_5()
    hparams.pos = 'timing'
    return hparams


@registry.register_hparams
def transformer_relative_tree_5_seq():
    """Use tree+sequential relative position embeddings instead of absolute sequential position encodings."""
    hparams = transformer_relative_tree_20_seq()
    hparams.max_relative_tree_distance = 5
    return hparams


@registry.register_hparams
def transformer_enhanced_pos():
    hparams = transformer_base()
    hparams.add_pos = True
    return hparams


@registry.register_hparams
def transformer_enhanced_deprel():
    hparams = transformer_base()
    hparams.add_deprel = True
    return hparams


@registry.register_hparams
def transformer_reserved_pos():
    hparams = transformer_base()
    hparams.pos_head = True
    return hparams


@registry.register_hparams
def transformer_reserved_deprel():
    hparams = transformer_base()
    hparams.deprel_head = True
    return hparams


@registry.register_hparams
def transformer_relative_reserved_pos():
    hparams = transformer_relative()
    hparams.pos_head = True
    return hparams


@registry.register_hparams
def transformer_pos_tagging_1_head():
    hparams = transformer_base()
    hparams.tagging_num_heads = 1
    return hparams


@registry.register_hparams
def transformer_dep_parse_l5():
    hparams = transformer_base()
    hparams.parse_from_layer = 5
    return hparams


@registry.register_hparams
def transformer_dep_parse_l0():
    hparams = transformer_base()
    hparams.parse_from_layer = 0
    return hparams


@registry.register_hparams
def transformer_dep_parse_l3():
    hparams = transformer_base()
    hparams.parse_from_layer = 3
    return hparams
