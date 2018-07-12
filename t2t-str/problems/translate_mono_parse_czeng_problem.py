from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import os
from collections import defaultdict

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
import tensorflow as tf

from . import data_utils
from . import translate_dep_parse_czeng_problem

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
OOV = '<OOV>'
ROOT = '<ROOT>'

__all__ = [
    "TranslateMonoParseCsenCzeng",
]


@registry.register_problem
class TranslateMonoParseCsenCzeng(translate_dep_parse_czeng_problem.TranslateDepParseCsenCzeng):
    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = data_utils.CzEngTokenTextEncoder(source_vocab_filename, replace_oov=OOV)
        target_token = data_utils.CzEngTokenTextEncoder(target_vocab_filename, replace_oov=OOV)
        return {
            "inputs": source_token,
            "targets": target_token,
            "target_dephead": data_utils.CzEngTokenIntEncoder(format_index=3, offset=-2),
        }

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        source_token = self.get_or_create_vocab(data_dir, tmp_dir, side=0)
        target_token = self.get_or_create_vocab(data_dir, tmp_dir, side=1)
        czeng_encoders = {
            "target_dephead": data_utils.CzEngTokenIntEncoder(format_index=3, offset=-2),
        }
        return data_utils.czeng_generate_encoded(generator, vocab=source_token, targets_vocab=target_token,
                                                 has_inputs=self.has_inputs, czeng_encoders=czeng_encoders)

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
            if side == 0:
                vocab.append(ROOT)
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename=None, vocab_list=vocab, replace_oov=OOV)
            encoder.store_to_file(vocab_filename)
        else:
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename, replace_oov=OOV)
        return encoder
