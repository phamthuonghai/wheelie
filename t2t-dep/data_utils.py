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
    def __init__(self, dep_subword_text_encoder, is_float=True):
        self._token_vocab = dep_subword_text_encoder
        self._is_float = is_float

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints, is_float=self._is_float)
        return source_pos["pos"]


class DepGovEncoder:
    def __init__(self, dep_subword_text_encoder, is_float=True):
        self._token_vocab = dep_subword_text_encoder
        self._is_float = is_float

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints, is_float=self._is_float)
        return source_pos["gov"]


class DepDepEncoder:
    def __init__(self, dep_subword_text_encoder, is_float=True):
        self._token_vocab = dep_subword_text_encoder
        self._is_float = is_float

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints, is_float=self._is_float)
        return source_pos["depth"]


class DepSibEncoder:
    def __init__(self, dep_subword_text_encoder, is_float=True):
        self._token_vocab = dep_subword_text_encoder
        self._is_float = is_float

    def encode(self, raw_text):
        source_ints = self._token_vocab.encode_from_list_hier(tokenizer(raw_text))
        source_pos, _ = dep_tokenizer(raw_text, source_ints, is_float=self._is_float)
        return source_pos["sib_ord"]


def dep_tokenizer(sentence, ints, eos_list=None, is_float=True):
    """Returns positions in dependency parse
    :param
        sentence: string in czeng export format
            "He|he|PRP|1|2|Sb saved|save|VBD|2|0|Pred my|my|PRP$|3|4|Atr ever-lovin|ever-lovin|NN|4|6|Atr
            '|'|''|5|6|AuxG neck|neck|NN|6|2|Obj .|.|.|7|0|AuxK"
        ints: list of list of ints
        eos_list
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
        return [float(x) if is_float else x for l in arr for x in l] + eos_list

    ret = {"pos": replicate(pos), "gov": replicate(gov), "depth": replicate(depth), "sib_ord": replicate(sib_ord)}

    return ret, [x for l in ints for x in l]


def tokenizer(text):
    """Encode a unicode string as a list of tokens.

      Args:
        text: a unicode string
      Returns:
        a list of tokens as Unicode strings
      """
    if not text:
        return []
    return [word.split('|')[0] for word in text.strip().split(' ')]
