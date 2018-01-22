import argparse


def linearize(sent, top_down):
    sent = [word.split('|') for word in sent.strip().split(' ')]

    l_s = len(sent)
    # x[0] is the value of the first word
    gov = [int(word[4]) for word in sent]

    # children[1] is the list of children of the first word
    children = [[] for _ in range(l_s + 1)]
    for _id, par_id in enumerate(gov):
        children[par_id].append(_id + 1)

    def dfs(cur_id, dep):
        children_ids = sorted(children[cur_id])
        s = ''
        for _, child in enumerate(children_ids):
            if top_down:
                s = s + ' %s <%s>' % (sent[child - 1][0], sent[child - 1][3]) + dfs(child, dep + 1)
            else:
                s = s + dfs(child, dep + 1) + ' %s <%s>' % (sent[child - 1][0], sent[child - 1][3])

        return s

    return dfs(0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('output_file')
    parser.add_argument('--cs', action='store_true')
    parser.add_argument('--en', action='store_true')
    parser.add_argument('--top_down', action='store_true')
    parser.add_argument('-e', '--export-format', action='store_true')
    args = parser.parse_args()

    fi = open(args.data_file, 'r', encoding='utf-8')
    fo = open(args.output_file, 'w', encoding='utf-8')

    line = fi.readline()
    while line:
        data = line.strip().split('\t')

        fo.write('%s\t%s\t%s\t%s\n' % (data[0], data[1],
                                       linearize(data[2], top_down=args.top_down) if args.cs else data[2],
                                       linearize(data[6], top_down=args.top_down) if args.en else data[6]))

        line = fi.readline()

    fi.close()
    fo.close()
