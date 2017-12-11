# coding=utf-8

"""Data generators for translation data-sets."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

from . import data_utils
from .encs_dep_noid_problem import get_or_generate_vocab

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class TranslateEncsDepRaw(translate.TranslateProblem):
    """Problem spec for WMT English-Czech translation with dependency in the small dataset."""

    @property
    def targeted_vocab_size(self):
        return 2 ** 15

    @property
    def vocab_name(self):
        return "vocab.encs"

    def generator(self, data_dir, tmp_dir, train):
        datasets = data_utils.ENCS_DEP_TRAIN_DATASETS if train else data_utils.ENCS_DEP_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets,
                                           "encs_dep_small_%s" % tag)

        vocab_datasets = [
            [item[0], ["encs_dep_small_%s.lang1" % tag, "encs_dep_small_%s.lang2" % tag]]
            for _id, item in enumerate(datasets)
        ]
        symbolizer_vocab = get_or_generate_vocab(data_dir, tmp_dir, self.vocab_file,
                                                 self.targeted_vocab_size, vocab_datasets)
        return token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.CS_TOK

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = data_utils.DepSubwordTextEncoder(vocab_filename)
        if self.has_inputs:
            return {"inputs": encoder, "targets": encoder,
                    "pos": data_utils.DepPosEncoder(encoder),
                    "gov": data_utils.DepGovEncoder(encoder),
                    "depth": data_utils.DepDepEncoder(encoder),
                    "sib_ord": data_utils.DepSibEncoder(encoder)
                    }
        return {"targets": encoder}

    def example_reading_spec(self):
        data_fields = {
            "targets": tf.VarLenFeature(tf.int64)
        }
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)
            data_fields["pos"] = tf.VarLenFeature(tf.float32)
            data_fields["gov"] = tf.VarLenFeature(tf.float32)
            data_fields["depth"] = tf.VarLenFeature(tf.float32)
            data_fields["sib_ord"] = tf.VarLenFeature(tf.float32)

        data_items_to_decoders = None
        return data_fields, data_items_to_decoders

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            p.input_modality = {
                    "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        if self.has_inputs:
            p.input_space_id = self.input_space_id
        p.target_space_id = self.target_space_id
        if self.is_character_level:
            p.loss_multiplier = 2.0


def token_generator(source_path, target_path, token_vocab, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

        This generator assumes the files at source_path and target_path have
        the same number of lines and yields dictionaries of "inputs" and "targets"
        where inputs are token ids from the " "-split source (and target, resp.) lines
        converted to integers using the token_map.

    Args:
        source_path: path to the file with source sentences.
        target_path: path to the file with target sentences.
        token_vocab: text_encoder.TextEncoder object.
        eos: integer to append at the end of each sequence (default: None).
    Yields:
        A dictionary {"inputs": source-line, "targets": target-line} where
        the lines are integer lists converted from tokens in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ints = token_vocab.encode_from_list_hier(data_utils.tokenizer(source))
                source_pos, source_ints = data_utils.dep_tokenizer(source, source_ints, eos_list)
                target_ints = token_vocab.encode(target) + eos_list
                source_pos.update({"inputs": source_ints + eos_list, "targets": target_ints})
                yield source_pos
                source, target = source_file.readline(), target_file.readline()
