from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import os
from collections import defaultdict

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate_encs
from tensor2tensor.data_generators.text_problems import VocabType, text2text_generate_encoded
from tensor2tensor.utils import registry
import tensorflow as tf

from . import data_utils

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
OOV = '<OOV>'


@registry.register_problem
class TranslateCsenCzengPlain(translate_encs.TranslateEncsWmt32k):
    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def vocab_type(self):
        return VocabType.TOKEN

    @property
    def source_vocab_name(self):
        return "vocab.csen.%d.cs" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.csen.%d.en" % self.approx_vocab_size

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = data_utils.CzEngTokenTextEncoder(source_vocab_filename)
        target_token = data_utils.CzEngTokenTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False, side=0):
        vocab_filename = os.path.join(data_dir, self.source_vocab_name if side == 0 else self.target_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            word_counts = defaultdict(int)
            for data_file in self.vocab_data_files():
                with tf.gfile.Open(os.path.join(tmp_dir, data_file[1][side]), mode="r") as f:
                    tf.logging.info("Build vocab from %s" % data_file[1][side])
                    for line in f:
                        for word in line.strip().split():
                            word_counts[word.split('|')[0]] += 1
            word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)[:self.approx_vocab_size]
            vocab = [w[0] for w in word_counts] + [OOV]
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename=None, vocab_list=vocab, replace_oov=OOV)
            encoder.store_to_file(vocab_filename)
        else:
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename, replace_oov=OOV)
        return encoder

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        source_token = self.get_or_create_vocab(data_dir, tmp_dir, side=0)
        target_token = self.get_or_create_vocab(data_dir, tmp_dir, side=1)
        return text2text_generate_encoded(generator, vocab=source_token, targets_vocab=target_token,
                                          has_inputs=self.has_inputs)

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return data_utils.CSEN_DEP_TRAIN_DATASETS if train else data_utils.CSEN_DEP_TEST_DATASETS

    def vocab_data_files(self):
        datasets = self.source_data_files(problem.DatasetSplit.TRAIN)
        vocab_datasets = [[
            datasets[0][0], [
                "%s-compiled-train.lang1" % self.name,
                "%s-compiled-train.lang2" % self.name
            ]
        ]]
        datasets = datasets[1:]
        vocab_datasets += [[item[0], [item[1][0], item[1][1]]] for item in datasets]
        return vocab_datasets
