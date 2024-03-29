from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import os
from collections import defaultdict

from tensor2tensor.data_generators import text_encoder, text_problems, generator_utils
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

__all__ = [
    "TranslateCsenCzengPlain",
    "TranslateCsenCzeng",
]


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
        source_token = data_utils.CzEngTokenTextEncoder(source_vocab_filename, replace_oov=OOV)
        target_token = data_utils.CzEngTokenTextEncoder(target_vocab_filename, replace_oov=OOV)
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


@registry.register_problem
class TranslateCsenCzeng(TranslateCsenCzengPlain):
    @property
    def pos_name(self):
        return "pos.csen.cs"

    @property
    def deprel_name(self):
        return "deprel.csen.cs"

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        pos_filename = os.path.join(data_dir, self.pos_name)
        deprel_filename = os.path.join(data_dir, self.deprel_name)
        source_token = data_utils.CzEngTokenTextEncoder(source_vocab_filename, replace_oov=OOV)
        target_token = data_utils.CzEngTokenTextEncoder(target_vocab_filename, replace_oov=OOV)
        return {
            "inputs": source_token,
            "targets": target_token,
            "relative_tree_distance_str": data_utils.CzEngRelativeTreeDistanceEncoder(),
            "tree_traversal_str": data_utils.CzEngTreeTranversalStrEncoder(),
            "pos": data_utils.CzEngTokenTextEncoder(pos_filename, replace_oov=OOV, format_index=2),
            "deprel": data_utils.CzEngTokenTextEncoder(deprel_filename, replace_oov=OOV, format_index=5),
        }

    def get_or_create_token_features(self, data_dir, tmp_dir, vocab_name, side=0, format_index=0):
        vocab_filename = os.path.join(data_dir, vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            word_counts = defaultdict(int)
            for data_file in self.vocab_data_files():
                with tf.gfile.Open(os.path.join(tmp_dir, data_file[1][side]), mode="r") as f:
                    tf.logging.info("Build feature %d vocab from %s" % (format_index, data_file[1][side]))
                    for line in f:
                        for word in line.strip().split():
                            word_counts[word.split('|')[format_index]] += 1
            word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)[:self.approx_vocab_size]
            vocab = [w[0] for w in word_counts] + [OOV]
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename=None, vocab_list=vocab,
                                                       replace_oov=OOV, format_index=format_index)
            encoder.store_to_file(vocab_filename)
        else:
            encoder = data_utils.CzEngTokenTextEncoder(vocab_filename, replace_oov=OOV, format_index=format_index)
        return encoder

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        source_token = self.get_or_create_vocab(data_dir, tmp_dir, side=0)
        target_token = self.get_or_create_vocab(data_dir, tmp_dir, side=1)
        pos_token = self.get_or_create_token_features(data_dir, tmp_dir, self.pos_name, format_index=2)
        deprel_token = self.get_or_create_token_features(data_dir, tmp_dir, self.deprel_name, format_index=5)
        czeng_encoders = {
            "relative_tree_distance_str": data_utils.CzEngRelativeTreeDistanceEncoder(),
            "tree_traversal_str": data_utils.CzEngTreeTranversalStrEncoder(),
            "pos": pos_token,
            "deprel": deprel_token,
        }
        return data_utils.czeng_generate_encoded(generator, vocab=source_token, targets_vocab=target_token,
                                                 has_inputs=self.has_inputs, czeng_encoders=czeng_encoders)

    def example_reading_spec(self):
        data_fields = {"targets": tf.VarLenFeature(tf.int64)}
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)
            data_fields["relative_tree_distance_str"] = tf.VarLenFeature(tf.string)
            data_fields["tree_traversal_str"] = tf.VarLenFeature(tf.string)
            data_fields["pos"] = tf.VarLenFeature(tf.int64)
            data_fields["deprel"] = tf.VarLenFeature(tf.int64)

        if self.packed_length:
            if self.has_inputs:
                data_fields["inputs_segmentation"] = tf.VarLenFeature(tf.int64)
                data_fields["inputs_position"] = tf.VarLenFeature(tf.int64)
            data_fields["targets_segmentation"] = tf.VarLenFeature(tf.int64)
            data_fields["targets_position"] = tf.VarLenFeature(tf.int64)

        data_items_to_decoders = None
        return data_fields, data_items_to_decoders

    def decode_example(self, serialized_example):
        """Return a dict of Tensors from a serialized tensorflow.Example."""
        data_fields, data_items_to_decoders = self.example_reading_spec()
        if data_items_to_decoders is None:
            data_items_to_decoders = {
                field: tf.contrib.slim.tfexample_decoder.Tensor(
                    field,
                    default_value="" if 'str' in field else 0)
                for field in data_fields
            }

        decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
            data_fields, data_items_to_decoders)

        decode_items = list(data_items_to_decoders)
        decoded = decoder.decode(serialized_example, items=decode_items)
        return dict(zip(decode_items, decoded))

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            p.input_modality = {
                "inputs": (registry.Modalities.SYMBOL, source_vocab_size),
                "relative_tree_distance_str": ("symbol:relative_tree_distance_str", 0),
                "tree_traversal_str": ("symbol:relative_tree_distance_str", 0),
                "pos": (registry.Modalities.SYMBOL, self._encoders["pos"].vocab_size),
                "deprel": (registry.Modalities.SYMBOL, self._encoders["deprel"].vocab_size),
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        if self.vocab_type == VocabType.CHARACTER:
            p.loss_multiplier = 2.0

        if self.packed_length:
            identity = (registry.Modalities.GENERIC, None)
            if self.has_inputs:
                p.input_modality["inputs_segmentation"] = identity
                p.input_modality["inputs_position"] = identity
            p.input_modality["targets_segmentation"] = identity
            p.input_modality["targets_position"] = identity


@registry.register_problem
class TranslateCsenCzengAlt(TranslateCsenCzengPlain):
    @property
    def approx_vocab_size(self):
        return 100000

    def source_data_files(self, dataset_split):
        # Use scripts/prepare_csen_czeng_alt.py to compile your own data
        return None

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
        data_path = os.path.join(tmp_dir, "%s-compiled-%s" % (self.name, tag))

        if self.vocab_type == text_problems.VocabType.SUBWORD:
          generator_utils.get_or_generate_vocab(
              data_dir, tmp_dir, self.vocab_filename, self.approx_vocab_size,
              self.vocab_data_files())

        return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                    data_path + ".lang2")

    def vocab_data_files(self):
        vocab_datasets = [[
            'decs', [
                "%s-compiled-train.lang1" % self.name,
                "%s-compiled-train.lang2" % self.name
            ]
        ]]
        return vocab_datasets
