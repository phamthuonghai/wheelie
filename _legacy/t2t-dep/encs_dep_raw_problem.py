# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

from . import data_utils
from . import csen_dep_raw_problem

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class TranslateEncsDepRaw(csen_dep_raw_problem.TranslateCsenDepRaw):

    def generator(self, data_dir, tmp_dir, train):
        datasets = data_utils.ENCS_DEP_TRAIN_DATASETS if train else data_utils.ENCS_DEP_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets,
                                           "encs_dep_%s" % tag)

        vocab_datasets = [
            [item[0], ["encs_dep_%s.lang1" % tag, "encs_dep_%s.lang2" % tag]]
            for _id, item in enumerate(datasets)
        ]
        symbolizer_vocab = csen_dep_raw_problem.get_or_generate_vocab(data_dir, tmp_dir, self.vocab_file,
                                                 self.targeted_vocab_size, vocab_datasets)
        return csen_dep_raw_problem.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)
