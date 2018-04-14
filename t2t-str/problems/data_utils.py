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
        """Converts an export format sentence to a list of ids."""
        tokens = [word.split('|')[0] for word in sentence.strip().split()]
        if self._replace_oov is not None:
            tokens = [t if t in self._token_to_id else self._replace_oov
                      for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret


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
        head_ids = [int(word[4])-1 for word in sentence]

        neighbors = [[] for _ in range(l_s)]
        for _id, head_id in enumerate(head_ids):
            if head_id >= 0:
                neighbors[head_id].append(_id)
                # head is also its neighbor
                neighbors[_id].append(head_id)

        visited = [False] * l_s
        ret = [['0' for _ in range(l_s)] for _ in range(l_s)]

        # should be BFS, but this is tree, so basically no difference
        def dfs(root, cur_id, dep):
            ret[root][cur_id] = str(dep)
            visited[cur_id] = True
            for neighbor in neighbors[cur_id]:
                if not visited[neighbor]:
                    dfs(root, neighbor, dep+1)

        for _id in range(l_s):
            visited = [False] * l_s
            visited[_id] = True
            dfs(_id, _id, 0)
            ret[_id] = ','.join(ret[_id])

        return ret


def czeng_generate_encoded(sample_generator, vocab, targets_vocab=None, has_inputs=True, czeng_encoders=None):
    """Encode CzEng samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
        if has_inputs:
            if czeng_encoders is not None:
                for feature, encoder in czeng_encoders.items():
                    sample[feature] = encoder.encode(sample["inputs"])

            sample["inputs"] = vocab.encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)
        sample["targets"] = targets_vocab.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
        yield sample
