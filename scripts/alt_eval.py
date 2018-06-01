import argparse

from sacrebleu import corpus_bleu


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('decoded_file')
    arg_parser.add_argument('source_file')
    arg_parser.add_argument('target_file')

    # From sacrebleu
    arg_parser.add_argument('-lc', action='store_true', default=False,
                            help='use case-insensitive BLEU (default: actual case)')
    arg_parser.add_argument('--smooth', '-s', choices=['exp', 'floor', 'none'], default='exp',
                            help='smoothing method: exponential decay (default), floor (0 count -> 0.01), or none')
    arg_parser.add_argument('--tokenize', '-tok', choices=['13a', 'intl', 'zh', 'none'], default='none',
                            help='tokenization method to use')
    arg_parser.add_argument('--force', default=False, action='store_true',
                            help='insist that your tokenized input is actually detokenized')

    args = arg_parser.parse_args()

    f_decoded = open(args.decoded_file, 'r', encoding='utf-8')
    f_source = open(args.source_file, 'r', encoding='utf-8')
    f_target = open(args.target_file, 'r', encoding='utf-8')

    decoded_sentences = [s.strip() for s in list(f_decoded)]
    source_sentences = [s.strip() for s in list(f_source)]
    target_sentences = [s.strip() for s in list(f_target)]

    if len(decoded_sentences) != len(target_sentences):
        raise Exception("Decoded file length mismatched!")

    decoded_translations = []
    target_translations = []
    acc_total = 0
    for i in range(len(decoded_sentences)):
        t_s = source_sentences[i].split()

        if t_s[-1] == '__translate__|_|_|_|_|_':
            decoded_translations.append(decoded_sentences[i])
            target_translations.append(target_sentences[i])
        elif t_s[-1] == '__depparse__|_|_|_|_|_':
            predicted_tags = decoded_sentences[i].split()
            gold_tags = target_sentences[i].split()

            l_gold_tags = len(gold_tags)
            l_predicted_tags = len(predicted_tags)
            if l_predicted_tags != l_gold_tags:
                print('Warning: Line %d: Predicted and gold length mismatched: %d %s %d' % (
                    i + 1, l_predicted_tags, '<' if l_predicted_tags != l_gold_tags else '>', l_gold_tags))
            acc = sum([1. if gold_tags[t] == predicted_tags[t] else 0.
                       for t in range(min(l_gold_tags, l_predicted_tags))]) / l_gold_tags
            acc_total += acc
        else:
            raise Exception('Unrecognized task in line %d' % (i+1))
            # Remove <ROOT> and <EOS>

    print('DEP accuracy: %.4f' % (acc_total * 100 / (len(decoded_sentences) - len(decoded_translations))))

    f_source.close()
    f_decoded.close()
    f_target.close()

    # BLEU
    bleu = corpus_bleu(decoded_translations, [target_translations], smooth=args.smooth, force=args.force,
                       lowercase=args.lc, tokenize=args.tokenize)
    print(
        'BLEU+ = {:.2f} {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP = {:.3f} ratio = {:.3f} hyp_len = {:d} '
        'ref_len = {:d})'.format(
            bleu.score, bleu.precisions[0], bleu.precisions[1], bleu.precisions[2],
            bleu.precisions[3], bleu.bp, bleu.sys_len / bleu.ref_len, bleu.sys_len, bleu.ref_len))
