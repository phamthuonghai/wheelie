from tensor2tensor.data_generators import text_encoder


CSEN_DEP_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*train")],
]
CSEN_DEP_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*test")],
]

CSEN_PLAIN_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 2, 3, "data.plaintext-format/*train")],
]
CSEN_PLAIN_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 2, 3, "data.plaintext-format/*test")],
]

ENCS_DEP_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 6, 2, "data.export-format/*train")],
]
ENCS_DEP_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 6, 2, "data.export-format/*test")],
]

ENCS_PLAIN_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 3, 2, "data.plaintext-format/*train")],
]
ENCS_PLAIN_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 3, 2, "data.plaintext-format/*test")],
]


class CzEngTokenTextEncoder(text_encoder.TokenTextEncoder):
    def encode(self, sentence):
        """Converts a space-separated string of tokens to a list of ids."""
        tokens = [word.split('|')[0] for word in sentence.strip().split()]
        if self._replace_oov is not None:
            tokens = [t if t in self._token_to_id else self._replace_oov
                      for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret
