import argparse
from nltk.tokenize.moses import MosesDetokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('source_file')
    parser.add_argument('target_file')
    parser.add_argument('-e', '--export-format', action='store_true')
    args = parser.parse_args()

    fi = open(args.data_file, 'r', encoding='utf-8')
    fo_src = open(args.source_file, 'w', encoding='utf-8')
    fo_ref = open(args.target_file, 'w', encoding='utf-8')

    detok = MosesDetokenizer()

    line = fi.readline()
    while line:
        data = line.strip().split('\t')
        if args.export_format:
            source_line = data[2]
            target_line = data[3]
            w_in_target = [w.strip().split('|')[0] for w in target_line.strip().split()]
            target_line = detok.detokenize(w_in_target, return_str=True)
        else:
            source_line = data[2]
            target_line = data[3]

        fo_src.write(source_line + '\n')
        fo_ref.write(target_line + '\n')

        line = fi.readline()

    fi.close()
    fo_src.close()
    fo_ref.close()
