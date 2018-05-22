import argparse
from sacrebleu import corpus_bleu


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('decoded_file')
    arg_parser.add_argument('source_file')
    arg_parser.add_argument('target_file')
    arg_parser.add_argument('--task', choices=['dep_head', 'pos_tag'], default='dep_head',
                            help='the task you are evaluating')

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
    if len(decoded_sentences) != len(source_sentences):
        raise Exception("Decoded file length mismatched!")

    decoded_sentences_split = []
    acc_total = 0
    info_id = 2 if args.task == 'pos_tag' else 4
    for i in range(len(decoded_sentences)):
        t = decoded_sentences[i].split('\t')
        decoded_sentences_split.append(t[0].strip())
        if len(t) < 2:
            print('Warning: Line %d: No predicted labels' % (i+1))
            predicted_tags = []
        else:
            predicted_tags = t[1].strip().split()
            # Remove <ROOT> and <EOS>
            if args.task == 'dep_head':
                predicted_tags = predicted_tags[0:-2]
        gold_tags = [w.split('|')[info_id] for w in source_sentences[i].strip().split()]
        l_gold_tags = len(gold_tags)
        if len(predicted_tags) != l_gold_tags:
            print('Warning: Line %d: Predicted and gold length mismatched: %d vs %d' % (
                i+1, len(predicted_tags), l_gold_tags))
        acc = sum([1. if gold_tags[t] == predicted_tags[t] else 0.
                   for t in range(min(l_gold_tags, len(predicted_tags)))]) / l_gold_tags
        acc_total += acc

    print('%s accuracy: %.4f' % (args.task.upper(), acc_total * 100 / len(decoded_sentences)))

    # BLEU
    bleu = corpus_bleu(decoded_sentences_split, [target_sentences], smooth=args.smooth, force=args.force,
                       lowercase=args.lc, tokenize=args.tokenize)
    print(
        'BLEU+ = {:.2f} {:.1f}/{:.1f}/{:.1f}/{:.1f} (BP = {:.3f} ratio = {:.3f} hyp_len = {:d} ref_len = {:d})'.format(
            bleu.score, bleu.precisions[0], bleu.precisions[1], bleu.precisions[2],
            bleu.precisions[3], bleu.bp, bleu.sys_len / bleu.ref_len, bleu.sys_len, bleu.ref_len))

    f_decoded.close()
    f_source.close()
    f_target.close()
