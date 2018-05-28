# This is an ad-hoc script to preprocess de2cs from export and tokenized format
import os

import tqdm

tmp_folder = './data/tmp'


def read_file(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        ret = list(f)
    return ret


def word_to_export(sentence):
    """
    :param sentence: string of tokenized word
    :return: word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function.
            Zachránil|zachránit_:W|VpYS---XR-AA---|1|0|Pred
    """
    words = sentence.strip().split()
    return ' '.join(['%s|_|_|_|_|_' % w for w in words])


def process_data(src_exp, src_tok, tgt_tok, output, output_2=None):
    src_exp = read_file(src_exp)
    src_tok = read_file(src_tok)
    tgt_tok = read_file(tgt_tok)

    n = len(src_tok)
    if n != len(tgt_tok):
        raise 'WTF! Source and target size mismatch: %s: %d vs %s: %d' % (src_tok, n, tgt_tok, len(tgt_tok))

    print(output)
    exp_id = 0
    src_ret = []
    tgt_ret = []
    for i in tqdm.tqdm(range(n)):
        src_tok[i] = src_tok[i].strip()
        tgt_tok[i] = tgt_tok[i].strip()
        if src_tok[i] == '':
            continue
        elif tgt_tok[i] == '':
            exp_id += 1
        else:
            src_ret.append(src_exp[exp_id].strip())
            tgt_ret.append(tgt_tok[i])
            exp_id += 1

    if output_2 is None:
        # Train and dev set
        with open(output + '.lang1', 'w', encoding='utf-8') as f:
            for i in range(len(src_ret)):
                f.write(src_ret[i] + '\n')
        with open(output + '.lang2', 'w', encoding='utf-8') as f:
            for i in range(len(tgt_ret)):
                f.write(word_to_export(tgt_ret[i]) + '\n')
    else:
        # Test set
        with open(output, 'w', encoding='utf-8') as f:
            for i in range(len(src_ret)):
                f.write(src_ret[i] + '\n')
        with open(output_2, 'w', encoding='utf-8') as f:
            for i in range(len(tgt_ret)):
                f.write(tgt_ret[i] + '\n')


if __name__ == '__main__':
    # source side DE
    train_source_tok = os.path.join(tmp_folder, 'decs/train.de.tok')
    train_source_export = os.path.join(tmp_folder, 'decs/train.de.export')

    dev_source_tok = os.path.join(tmp_folder, 'decs/dev.de.tok')
    dev_source_export = os.path.join(tmp_folder, 'decs/dev.de.export')

    test_source_tok = os.path.join(tmp_folder, 'decs/test.de.tok')
    test_source_export = os.path.join(tmp_folder, 'decs/test.de.export')

    # target side CS
    train_target_tok = os.path.join(tmp_folder, 'decs/train.cs.tok')
    dev_target_tok = os.path.join(tmp_folder, 'decs/dev.cs.tok')
    test_target_tok = os.path.join(tmp_folder, 'decs/test.cs.tok')

    output_train = os.path.join(tmp_folder, 'translate_decs-compiled-train')
    output_dev = os.path.join(tmp_folder, 'translate_decs-compiled-dev')

    output_dev_src = os.path.join(tmp_folder, 'decs/decode-dev.de')
    output_dev_tgt = os.path.join(tmp_folder, 'decs/decode-dev.cs')

    output_test_src = os.path.join(tmp_folder, 'decs/decode-test.de')
    output_test_tgt = os.path.join(tmp_folder, 'decs/decode-test.cs')

    process_data(train_source_export, train_source_tok, train_target_tok, output=output_train)
    process_data(dev_source_export, dev_source_tok, dev_target_tok, output=output_dev)
    process_data(dev_source_export, dev_source_tok, dev_target_tok, output=output_dev_src, output_2=output_dev_tgt)
    process_data(test_source_export, test_source_tok, test_target_tok, output=output_test_src, output_2=output_test_tgt)
