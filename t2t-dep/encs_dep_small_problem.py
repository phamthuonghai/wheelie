# coding=utf-8

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

from . import data_utils

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENCS_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 3, 7, "data.export-format/*train")],
]
_ENCS_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 3, 7, "data.export-format/*test")],
]


@registry.register_problem
class TranslateEncsDepSmall(translate.TranslateProblem):
    """Problem spec for WMT English-Czech translation with dependency in the small dataset."""

    @property
    def targeted_vocab_size(self):
        return 2 ** 15

    @property
    def vocab_name(self):
        return "vocab.encs"

    def generator(self, data_dir, tmp_dir, train):
        datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets,
                                           "encs_dep_small_%s" % tag)

        vocab_datasets = [
            [item[0], ["encs_dep_small_%s.lang1" % tag, "encs_dep_small_%s.lang2" % tag]]
            for _id, item in enumerate(datasets)
        ]
        symbolizer_vocab = data_utils.get_or_generate_vocab(data_dir, tmp_dir, self.vocab_file,
                                                            self.targeted_vocab_size, vocab_datasets)
        return data_utils.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.CS_TOK
