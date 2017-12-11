import sys

if __name__ == '__main__':
    fi = open(sys.argv[1], 'r', encoding='utf-8')
    fo_src = open(sys.argv[2], 'w', encoding='utf-8')
    fo_ref = open(sys.argv[3], 'w', encoding='utf-8')

    line = fi.readline()
    while line:
        data = line.strip().split('\t')
        source_line = data[2]
        target_line = data[6]
        w_in_target = [w.strip().split('|')[0] for w in target_line.strip().split()]

        fo_src.write(source_line + '\n')
        fo_ref.write(' '.join(w_in_target) + '\n')

        line = fi.readline()

    fi.close()
    fo_src.close()
    fo_ref.close()
