from tensor2tensor.data_generators import text_encoder

CSEN_DEP_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*train")],
]
CSEN_DEP_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 2, 6, "data.export-format/*dev")],
]

CSEN_PLAIN_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 2, 3, "data.plaintext-format/*train")],
]
CSEN_PLAIN_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 2, 3, "data.plaintext-format/*dev")],
]

ENCS_DEP_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 6, 2, "data.export-format/*train")],
]
ENCS_DEP_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-export-format.0.tar",
     ("tsv", 6, 2, "data.export-format/*dev")],
]

ENCS_PLAIN_TRAIN_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 3, 2, "data.plaintext-format/*train")],
]
ENCS_PLAIN_TEST_DATASETS = [
    ["http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.0.tar",
     ("tsv", 3, 2, "data.plaintext-format/*dev")],
]

ROOT_DUMMY = '<ROOT>|<ROOT>|<ROOT>|0|0|<ROOT>'


class CzEngTokenTextEncoder(text_encoder.TokenTextEncoder):
    def __init__(self, vocab_filename, reverse=False, vocab_list=None, replace_oov=None,
                 num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS, format_index=0, with_root=False):
        super(CzEngTokenTextEncoder, self).__init__(vocab_filename, reverse=reverse, vocab_list=vocab_list,
                                                    replace_oov=replace_oov, num_reserved_ids=num_reserved_ids)
        self.format_index = format_index
        self.with_root = with_root

    def encode(self, sentence):
        """Converts an export format sentence to a list of ids."""
        if self.with_root:
            sentence = ROOT_DUMMY + ' ' + sentence
        tokens = [word.split('|')[self.format_index] for word in sentence.strip().split()]
        if self._replace_oov is not None:
            tokens = [t if t in self._token_to_id else self._replace_oov
                      for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret


class CzEngTokenIntEncoder:
    def __init__(self, format_index=4, dtype=int, reverse=False, with_root=False):
        self._format_index = format_index
        self._dtype = dtype
        self._reverse = reverse
        self._with_root = with_root

    def encode(self, sentence):
        """Converts an export format sentence to a list of ids."""
        if self._with_root:
            sentence = ROOT_DUMMY + ' ' + sentence
        ret = [self._dtype(word.split('|')[self._format_index]) for word in sentence.strip().split()]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids):
        return " ".join([str(_id) for _id in ids])


class CzEngRelativeTreeDistanceEncoder:
    def encode(self, sentence):
        """Returns positions in dependency parse
        :param
            sentence: string in czeng export format
                "He|he|PRP|1|2|Sb saved|save|VBD|2|0|Pred my|my|PRP$|3|4|Atr ever-lovin|ever-lovin|NN|4|6|Atr
                '|'|''|5|6|AuxG neck|neck|NN|6|2|Obj .|.|.|7|0|AuxK"
        :return:
            relative_tree_distance: string, adjacent matrix row, node separated by ','
        """
        sentence = [word.split('|') for word in sentence.strip().split()]

        l_s = len(sentence)
        head_ids = [-1] + [int(word[4]) for word in sentence]

        neighbors = [[] for _ in range(l_s + 1)]
        for _id, head_id in enumerate(head_ids):
            if head_id >= 0:
                neighbors[head_id].append(_id)
                # head is also its neighbor
                neighbors[_id].append(head_id)

        visited = [False] * (l_s + 1)
        ret = [['0' for _ in range(l_s)] for _ in range(l_s)]  # Remove ROOT

        # should be BFS, but this is tree, so basically no difference
        def dfs(root, cur_id, dep):
            if cur_id > 0:
                ret[root - 1][cur_id - 1] = str(dep)
            visited[cur_id] = True
            for neighbor in neighbors[cur_id]:
                if not visited[neighbor]:
                    dfs(root, neighbor, dep + 1)

        for _id in range(1, l_s + 1):
            visited = [False] * (l_s + 1)
            visited[_id] = True
            dfs(_id, _id, 1)
            ret[_id - 1] = ','.join(ret[_id - 1])

        return ret


MAX_TRAVERSAL_LENGTH = 10
OUT_OF_REACH_ID = '1'


def _generate_tree_traversal_code():
    # Tree traversal pattern
    # U*D* : go n ups and n downs
    # LD*  : left siblings then downs
    # RD*  : right siblings then downs
    n = MAX_TRAVERSAL_LENGTH
    tree_traversal_code = ['L', 'R']
    for i in range(n + 1):
        for j in range(i + 1):
            tree_traversal_code.append(('U' * j) + ('D' * (i - j)))

    ret = {}
    for _id, code in enumerate(tree_traversal_code):
        ret[code] = str(_id + 2)  # reserve 0 for padding, 1 for out-of-reach
    return ret


TREE_TRAVERSAL_ID = _generate_tree_traversal_code()


class CzEngTreeTranversalStrEncoder:
    def encode(self, sentence):
        """Returns tree traversal id in dependency parse
        :param
            sentence: string in czeng export format
                "He|he|PRP|1|2|Sb saved|save|VBD|2|0|Pred my|my|PRP$|3|4|Atr ever-lovin|ever-lovin|NN|4|6|Atr
                '|'|''|5|6|AuxG neck|neck|NN|6|2|Obj .|.|.|7|0|AuxK"
        :return:
            tree_traversal_str: string, tree traversal id, node separated by ','
        """
        sentence = [word.split('|') for word in sentence.strip().split()]

        l_s = len(sentence)
        head_ids = [-1] + [int(word[4]) for word in sentence]

        children = [[] for _ in range(l_s + 1)]
        for _id, head_id in enumerate(head_ids):
            if head_id >= 0:
                children[head_id].append(_id)

        visited = [False] * (l_s + 1)
        ret = [[OUT_OF_REACH_ID for _ in range(l_s)] for _ in range(l_s)]  # Remove ROOT

        # should be BFS, but this is tree, so basically no difference
        def dfs(root, cur_id, diary):
            if diary == 'UD':
                # Special cases for left and right siblings
                tmp_diary = 'L' if cur_id < root else 'R'
            else:
                tmp_diary = diary

            if cur_id > 0:
                ret[root - 1][cur_id - 1] = TREE_TRAVERSAL_ID.get(tmp_diary, OUT_OF_REACH_ID)

            if len(diary) >= MAX_TRAVERSAL_LENGTH:
                return

            visited[cur_id] = True
            if head_ids[cur_id] >= 0 and not visited[head_ids[cur_id]]:
                dfs(root, head_ids[cur_id], diary + 'U')
            for neighbor in children[cur_id]:
                if not visited[neighbor]:
                    dfs(root, neighbor, diary + 'D')

        for _id in range(1, l_s + 1):
            visited = [False] * (l_s + 1)
            visited[_id] = True
            dfs(_id, _id, '')
            ret[_id - 1] = ','.join(ret[_id - 1])

        return ret


def czeng_generate_encoded(sample_generator, vocab, targets_vocab=None, has_inputs=True, czeng_encoders=None):
    """Encode CzEng samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
        if has_inputs:
            if czeng_encoders is not None:
                for feature, encoder in czeng_encoders.items():
                    sample[feature] = encoder.encode(sample["inputs"])
                    if 'str' not in feature:
                        sample[feature].append(text_encoder.EOS_ID)

            sample["inputs"] = vocab.encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)
        sample["targets"] = targets_vocab.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
        yield sample
