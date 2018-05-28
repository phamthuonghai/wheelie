from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import text_encoder, text_problems, generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
import tensorflow as tf

from . import translate_czeng_problem
from . import translate_dep_parse_czeng_problem

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
OOV = '<OOV>'

__all__ = [
    "TranslateDecs",
    "TranslateDepParseDecs",
]


@registry.register_problem
class TranslateDecs(translate_czeng_problem.TranslateCsenCzengPlain):
    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def source_vocab_name(self):
        return "vocab.decs.%d.de" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.decs.%d.cs" % self.approx_vocab_size

    def source_data_files(self, dataset_split):
        # Use scripts/prepare_de2cs.py to compile your own data
        return None

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
        data_path = os.path.join(tmp_dir, "translate_decs-compiled-%s" % tag)

        if self.vocab_type == text_problems.VocabType.SUBWORD:
          generator_utils.get_or_generate_vocab(
              data_dir, tmp_dir, self.vocab_filename, self.approx_vocab_size,
              self.vocab_data_files())

        return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                    data_path + ".lang2")

    def vocab_data_files(self):
        vocab_datasets = [[
            'decs', [
                "translate_decs-compiled-train.lang1",
                "translate_decs-compiled-train.lang2"
            ]
        ]]
        return vocab_datasets


@registry.register_problem
class TranslateDepParseDecs(translate_dep_parse_czeng_problem.TranslateDepParseCsenCzeng):

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def source_vocab_name(self):
        return "vocab.decs.%d.de" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.decs.%d.cs" % self.approx_vocab_size

    def source_data_files(self, dataset_split):
        # Use scripts/prepare_de2cs.py to compile your own data
        return None

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
        data_path = os.path.join(tmp_dir, "translate_decs-compiled-%s" % tag)

        if self.vocab_type == text_problems.VocabType.SUBWORD:
          generator_utils.get_or_generate_vocab(
              data_dir, tmp_dir, self.vocab_filename, self.approx_vocab_size,
              self.vocab_data_files())

        return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                    data_path + ".lang2")

    def vocab_data_files(self):
        vocab_datasets = [[
            'decs', [
                "translate_decs-compiled-train.lang1",
                "translate_decs-compiled-train.lang2"
            ]
        ]]
        return vocab_datasets
