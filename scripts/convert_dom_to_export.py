# Convert Dominik's format to export
import argparse


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('tok_file')
    arg_parser.add_argument('dheads_file')
    arg_parser.add_argument('dtags_file')
    arg_parser.add_argument('output_file')

    args = arg_parser.parse_args()

    with open(args.tok_file, 'r', encoding='utf-8') as f:
        toks = [line.strip() for line in list(f)]

    with open(args.dheads_file, 'r', encoding='utf-8') as f:
        dheads = [line.strip() for line in list(f)]

    with open(args.dtags_file, 'r', encoding='utf-8') as f:
        dtags = [line.strip() for line in list(f)]

    l_toks = len(toks)
    l_dheads = len(dheads)
    l_dtags = len(dtags)

    if l_toks != l_dheads or l_toks != l_dtags or l_dheads != l_dtags:
        raise Exception('Input files have different lengths: tok: %d, dheads: %d, dtags:%d' %
                        (l_toks, l_dheads, l_dtags))

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for _id in range(l_toks):
            ret = ['%s|_|_|%d|%s|%s' % (tok, s_id+1, head, tag)
                   for s_id, (tok, head, tag) in
                   enumerate(zip(toks[_id].split(), dheads[_id].split(), dtags[_id].split()))]
            f.write('%s\n' % ' '.join(ret))
