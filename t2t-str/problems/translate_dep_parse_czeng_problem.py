from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import os
from collections import defaultdict

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.text_problems import VocabType
from tensor2tensor.utils import registry
import tensorflow as tf

from . import data_utils
from . import translate_czeng_problem

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID
OOV = '<OOV>'
ROOT = '<ROOT>'

__all__ = [
    "TranslateDepParseCsenCzeng",
]


@registry.register_problem
class TranslateDepParseCsenCzeng(translate_czeng_problem.TranslateCsenCzengPlain):
    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = data_utils.CzEngTokenTextEncoder(source_vocab_filename, replace_oov=OOV, with_root=True)
        target_token = data_utils.CzEngTokenTextEncoder(target_vocab_filename, replace_oov=OOV)
        return {
            "inputs": source_token,
            "targets": target_token,
            "target_dephead": data_utils.CzEngTokenIntEncoder(format_index=4, with_root=True),
        }

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        source_token = self.get_or_create_vocab(data_dir, tmp_dir, side=0)
        target_token = self.get_or_create_vocab(data_dir, tmp_dir, side=1)
        czeng_encoders = {
            "target_dephead": data_utils.CzEngTokenIntEncoder(format_index=4, with_root=True),
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
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename=None, vocab_list=vocab,
                                                       replace_oov=OOV, with_root=(side == 0))
            encoder.store_to_file(vocab_filename)
        else:
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename, replace_oov=OOV, with_root=(side == 0))
        return encoder

    def example_reading_spec(self):
        data_fields = {
            "targets": tf.VarLenFeature(tf.int64),
            "target_dephead": tf.VarLenFeature(tf.int64)
        }
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)

        data_items_to_decoders = None
        return data_fields, data_items_to_decoders

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            p.input_modality = {
                "inputs": (registry.Modalities.SYMBOL, source_vocab_size),
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = {
            "targets": (registry.Modalities.SYMBOL, target_vocab_size),
            "target_dephead": ("symbol:dep_head", 256),  # hparams.max_length
        }
        if self.vocab_type == VocabType.CHARACTER:
            p.loss_multiplier = 2.0
