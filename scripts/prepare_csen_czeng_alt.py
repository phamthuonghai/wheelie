# This is an ad-hoc script to preprocess de2cs from export and tokenized format
import os

tmp_folder = './data/tmp'


TRANSLATE_WORD = '__translate__|_|_|_|_|_'
DEPPARSE_WORD = '__depparse__|_|_|_|_|_'


def read_file(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        ret = list([line.strip() for line in f])
    return ret


def word_to_export(sentence):
    """
    :param sentence: string of tokenized word
    :return: word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function.
            Zachránil|zachránit_:W|VpYS---XR-AA---|1|0|Pred
    """
    words = sentence.strip().split()
    return ' '.join(['%s|_|_|_|_|_' % w for w in words])


def get_head(sentence):
    """
    :param: word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function.
            Zachránil|zachránit_:W|VpYS---XR-AA---|1|0|Pred
    :return: sentence: string of tokenized head word-form
    """
    try:
        words = [w.strip().split('|') for w in sentence.strip().split()]
        ret = [words[int(w[-2])-1][0] if int(w[-2]) > 0 else '__ROOT__' for w in words]
    except Exception as e:
        print(sentence)
        raise e
    return ' '.join(ret)


def get_word(sentence):
    """
    :param: word-form|lemma|morphological-tag|index-in-sentence|index-of-governor|syntactic-function.
            Zachránil|zachránit_:W|VpYS---XR-AA---|1|0|Pred
    :return: sentence: string of tokenized word-form
    """
    ret = [w.strip().split('|')[0] for w in sentence.strip().split()]
    return ' '.join(ret)


def process_data(src_tok, tgt_tok, output, output_2=None, is_dev=False):
    src_ret = read_file(src_tok)
    tgt_ret = read_file(tgt_tok)

    n = len(src_tok)
    if n != len(tgt_tok):
        raise 'WTF! Source and target size mismatch: %s: %d vs %s: %d' % (src_tok, n, tgt_tok, len(tgt_tok))

    print(output)

    if output_2 is None:
        # Train and dev set
        with open(output + '.lang1', 'w', encoding='utf-8') as f:
            for i in range(len(src_ret)):
                f.write(src_ret[i] + ' ' + TRANSLATE_WORD + '\n')
                f.write(src_ret[i] + ' ' + DEPPARSE_WORD + '\n')
        with open(output + '.lang2', 'w', encoding='utf-8') as f:
            for i in range(len(tgt_ret)):
                f.write(tgt_ret[i] + '\n')
                f.write(word_to_export(get_head(src_ret[i])) + '\n')
    else:
        # Test set
        with open(output, 'w', encoding='utf-8') as f:
            for i in range(len(src_ret)):
                f.write(src_ret[i] + ' ' + TRANSLATE_WORD + '\n')
                f.write(src_ret[i] + ' ' + DEPPARSE_WORD + '\n')
        with open(output_2, 'w', encoding='utf-8') as f:
            for i in range(len(tgt_ret)):
                f.write((get_head(tgt_ret[i]) if is_dev else tgt_ret[i]) + '\n')
                f.write(get_head(src_ret[i]) + '\n')


if __name__ == '__main__':
    # source side CS
    train_source = os.path.join(tmp_folder, 'translate_csen_czeng-compiled-train.lang1')
    dev_source = os.path.join(tmp_folder, 'translate_csen_czeng-compiled-dev.lang1')
    test_source = os.path.join(tmp_folder, 'data.export-format/09decode-test-10k.cs')

    # target side EN
    train_target = os.path.join(tmp_folder, 'translate_csen_czeng-compiled-train.lang2')
    dev_target = os.path.join(tmp_folder, 'translate_csen_czeng-compiled-dev.lang2')
    test_target = os.path.join(tmp_folder, 'data.export-format/09decode-test-10k.en')

    output_train = os.path.join(tmp_folder, 'translate_csen_czeng_alt-compiled-train')
    output_dev = os.path.join(tmp_folder, 'translate_csen_czeng_alt-compiled-dev')

    output_dev_src = os.path.join(tmp_folder, 'data.export-format/decode-alt-dev.cs')
    output_dev_tgt = os.path.join(tmp_folder, 'data.export-format/decode-alt-dev.en')

    output_test_src = os.path.join(tmp_folder, 'data.export-format/decode-alt-test-10k.cs')
    output_test_tgt = os.path.join(tmp_folder, 'data.export-format/decode-alt-test-10k.en')

    process_data(train_source, train_target, output=output_train)
    process_data(dev_source, dev_target, output=output_dev)
    process_data(dev_source, dev_target, output=output_dev_src, output_2=output_dev_tgt, is_dev=True)
    process_data(test_source, test_target, output=output_test_src, output_2=output_test_tgt)
