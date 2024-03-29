# coding=utf-8

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from collections import defaultdict
import tarfile

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.data_generators.generator_utils import maybe_download, gunzip_file

from . import data_utils

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class TranslateCsenDepNoid(translate.TranslateProblem):
    """Problem spec for WMT English-Czech translation with small dataset."""

    @property
    def targeted_vocab_size(self):
        return 2 ** 15

    @property
    def vocab_name(self):
        return "vocab.encs"

    def generator(self, data_dir, tmp_dir, train):
        datasets = data_utils.CSEN_DEP_TRAIN_DATASETS if train else data_utils.CSEN_DEP_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets,
                                           "csen_dep_%s" % tag)

        vocab_datasets = [
            [item[0], ["csen_dep_%s.lang1" % tag, "csen_dep_%s.lang2" % tag]]
            for _id, item in enumerate(datasets)
        ]
        symbolizer_vocab = get_or_generate_vocab(data_dir, tmp_dir, self.vocab_file,
                                                 self.targeted_vocab_size, vocab_datasets)
        return token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = data_utils.DepSubwordTextEncoder(vocab_filename)
        if self.has_inputs:
            return {"inputs": encoder, "targets": encoder}
        return {"targets": encoder}

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.CS_TOK


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
                source_ints = token_vocab.encode(source) + eos_list
                target_ints = token_vocab.encode(target) + eos_list
                yield {"inputs": source_ints, "targets": target_ints}
                source, target = source_file.readline(), target_file.readline()


def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator):
    """Inner implementation for vocab generators.

      Args:
        data_dir: The base directory where data and vocab files are stored. If None,
            then do not save the vocab even if it doesn't exist.
        vocab_filename: relative filename where vocab file is stored
        vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
        generator: a generator that produces tokens from the vocabulary

      Returns:
        A DepSubwordTextEncoder vocabulary object.
      """
    if data_dir is None:
        vocab_filepath = None
    else:
        vocab_filepath = os.path.join(data_dir, vocab_filename)

    if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        vocab = data_utils.DepSubwordTextEncoder(vocab_filepath)
        return vocab

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    token_counts = defaultdict(int)
    for item in generator:
        for tok in data_utils.tokenizer(item):
            token_counts[tok] += 1

    vocab = data_utils.DepSubwordTextEncoder.build_to_target_size(
        vocab_size, token_counts, 1, 1e3)

    if vocab_filepath is not None:
        vocab.store_to_file(vocab_filepath)
    return vocab


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources):
    """Generate a vocabulary from the datasets in sources."""

    def generate():
        tf.logging.info("Generating vocab from: %s", str(sources))
        for source in sources:
            url = source[0]
            filename = os.path.basename(url)
            compressed_file = maybe_download(tmp_dir, filename, url)

            for lang_file in source[1]:
                tf.logging.info("Reading file: %s" % lang_file)
                filepath = os.path.join(tmp_dir, lang_file)

                # Extract from tar if needed.
                if not tf.gfile.Exists(filepath):
                    read_type = "r:gz" if filename.endswith("tgz") else "r"
                    with tarfile.open(compressed_file, read_type) as corpus_tar:
                        corpus_tar.extractall(tmp_dir)

                # For some datasets a second extraction is necessary.
                if lang_file.endswith(".gz"):
                    new_filepath = os.path.join(tmp_dir, lang_file[:-3])
                    if tf.gfile.Exists(new_filepath):
                        tf.logging.info(
                            "Subdirectory %s already exists, skipping unpacking" % filepath)
                    else:
                        tf.logging.info("Unpacking subdirectory %s" % filepath)
                        gunzip_file(filepath, new_filepath)
                    filepath = new_filepath

                # Use Tokenizer to count the word occurrences.
                with tf.gfile.GFile(filepath, mode="r") as source_file:
                    file_byte_budget = 16e6
                    counter = 0
                    countermax = int(source_file.size() / file_byte_budget / 2)
                    for line in source_file:
                        if counter < countermax:
                            counter += 1
                        else:
                            if file_byte_budget <= 0:
                                break
                            line = line.strip()
                            file_byte_budget -= len(line)
                            counter = 0
                            yield line

    return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                       generate())
