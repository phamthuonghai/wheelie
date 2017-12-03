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

from .encs_small_problem import tokenizer

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENCS_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*train")],
]
_ENCS_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*test")],
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
        encoder = DepSubwordTextEncoder(vocab_filename)
        if self.has_inputs:
            return {"inputs": encoder, "targets": encoder,
                    "pos": DepPosEncoder(encoder),
                    "gov": DepGovEncoder(encoder),
                    "depth": DepDepEncoder(encoder),
                    "sib_ord": DepSibEncoder(encoder)
                    }
        return {"targets": encoder}

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        if self.has_inputs:
            source_vocab_size = self._encoders["inputs"].vocab_size
            p.input_modality = {
                    "inputs": (registry.Modalities.SYMBOL, source_vocab_size),
                    "pos": (registry.Modalities.SYMBOL, 100),
                    "gov": (registry.Modalities.SYMBOL, 100),
                    "depth": (registry.Modalities.SYMBOL, 100),
                    "sib_ord": (registry.Modalities.SYMBOL, 100)
            }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        if self.has_inputs:
            p.input_space_id = self.input_space_id
        p.target_space_id = self.target_space_id
        if self.is_character_level:
            p.loss_multiplier = 2.0


class DepSubwordTextEncoder(text_encoder.SubwordTextEncoder):
    def encode(self, raw_text):
        """Converts a native string to a list of subtoken ids.

        Args:
          raw_text: a native string.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(tokenizer(raw_text))

    def encode_from_list(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.
        :param tokens: list of Unicode strings
        :return: a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken_ids(tokens)

    def encode_from_list_hier(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.
        :param tokens: list of Unicode strings
        :return: a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_list_subtoken_ids(tokens)

    def _tokens_to_list_subtoken_ids(self, tokens):
        """Converts a list of tokens to a list of subtoken ids.
        :param tokens: a list of strings.
        :return: a list of list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.append(self._token_to_subtoken_ids(token))
        return ret


# TODO: These are not efficient
class DepPosEncoder:
    def __init__(self, dep_subword_text_encoder):
        self._token_vocab = dep_subword_text_encoder

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints)
        return source_pos["pos"]


class DepGovEncoder:
    def __init__(self, dep_subword_text_encoder):
        self._token_vocab = dep_subword_text_encoder

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints)
        return source_pos["gov"]


class DepDepEncoder:
    def __init__(self, dep_subword_text_encoder):
        self._token_vocab = dep_subword_text_encoder

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints)
        return source_pos["depth"]


class DepSibEncoder:
    def __init__(self, dep_subword_text_encoder):
        self._token_vocab = dep_subword_text_encoder

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints)
        return source_pos["sib_ord"]


def dep_tokenizer(sentence, ints):
    """Returns positions in dependency parse
    :param
        sentence: string in czeng export format
            "He|he|PRP|1|2|Sb saved|save|VBD|2|0|Pred my|my|PRP$|3|4|Atr ever-lovin|ever-lovin|NN|4|6|Atr
            '|'|''|5|6|AuxG neck|neck|NN|6|2|Obj .|.|.|7|0|AuxK"
        ints: list of list of ints
    :return:
        dep_pos: dict   {"pos": [],     position in sentence
                        "gov": [],      index of gonvernor
                        "depth": [],    depth of node
                        "sib_ord": []}  order among siblings
        ints: flattened ints
    """
    sentence = [word.split('|') for word in sentence.strip().split(' ')]
    l_s = len(sentence)
    # x[0] is the value of the first word
    pos = [int(word[3]) for word in sentence]
    gov = [int(word[4]) for word in sentence]
    depth = [[] for _ in range(l_s)]
    sib_ord = [[] for _ in range(l_s)]

    # children[1] is the list of children of the first word
    children = [[] for _ in range(l_s+1)]
    for _id, par_id in enumerate(gov):
        children[par_id].append(_id + 1)

    def dfs(cur_id, dep):
        children_ids = sorted(children[cur_id])
        for _id, child in enumerate(children_ids):
            depth[child - 1] = dep
            sib_ord[child - 1] = _id + 1
            dfs(child, dep + 1)

    dfs(0, 1)

    n_subword = [len(x) for x in ints]

    def replicate(arr):
        for _id, v in enumerate(arr):
            arr[_id] = [v] * n_subword[_id]
        return [x for l in arr for x in l]

    ret = {"pos": replicate(pos), "gov": replicate(gov), "depth": replicate(depth), "sib_ord": replicate(sib_ord)}

    return ret, [x for l in ints for x in l]


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
                source_ints = token_vocab.encode_from_list_hier(tokenizer(source))
                source_pos, source_ints = dep_tokenizer(source, source_ints)
                target_ints = token_vocab.encode(target) + eos_list
                yield {"inputs": source_ints + eos_list, "targets": target_ints}.update(source_pos)
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
        A SubwordTextEncoder vocabulary object.
      """
    if data_dir is None:
        vocab_filepath = None
    else:
        vocab_filepath = os.path.join(data_dir, vocab_filename)

    if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        vocab = DepSubwordTextEncoder(vocab_filepath)
        return vocab

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    token_counts = defaultdict(int)
    for item in generator:
        for tok in tokenizer(text_encoder.native_to_unicode(item)):
            token_counts[tok] += 1

    vocab = DepSubwordTextEncoder.build_to_target_size(
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
                    file_byte_budget = 1e6
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
