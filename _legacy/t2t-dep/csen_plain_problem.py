from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import translate_encs
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.utils import registry
import tensorflow as tf

from . import data_utils

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class TranslateCsenPlain(translate_encs.TranslateEncsWmt32k):

    def generator(self, data_dir, tmp_dir, train):

        datasets = data_utils.CSEN_PLAIN_TRAIN_DATASETS if train else data_utils.CSEN_PLAIN_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets,
                                           "csen_plain_%s" % tag)

        vocab_datasets = [
            [item[0], ["csen_plain_%s.lang1" % tag, "csen_plain_%s.lang2" % tag]]
            for _id, item in enumerate(datasets)
        ]

        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, vocab_datasets)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
                                         symbolizer_vocab, EOS)
