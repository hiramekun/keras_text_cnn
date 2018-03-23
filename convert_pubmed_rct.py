def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./pubmed-rct')
    parser.add_argument('--data_size', choices=('20k', '200k'), default='200k')
    parser.add_argument('--out', default='./data')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--replace_number', default=False, type=bool)
    args = parser.parse_args()
    return args


def load_row_data(args):
    def extract_file_path(args):
        fdir = f'{args.data_path}/PubMed_{args.data_size}_RCT'

        if args.replace_number:
            fdir = f'{fdir}_numbers_replaced_with_at_sign'

        fpath = f'{fdir}/{args.mode}.txt'
        return fpath

    with open(extract_file_path(args)) as f:
        lines = [line for line in f.readlines() if not line.startswith('##')]

    return lines


def convert_data(args, row_data):
    import os
    import pandas as pd
    import shutil
    from tqdm import tqdm

    outdir = f'{args.out}/{args.mode}'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
        print('deleted old directory')

    os.makedirs(outdir)
    fname = f'{outdir}/{args.mode}.tsv'

    with open(fname, mode='w') as f:
        f.writelines(row_data)

    print('writing lines...')
    for row in tqdm(pd.read_table(fname, header=None).iterrows()):
        with open(f'{outdir}/{row[1][0].lower()}.tsv', mode='a') as f:
            f.write(row[1][1] + '\n')


def main(args):
    data = load_row_data(args)
    convert_data(args, data)


if __name__ == '__main__':
    main(args_parser())
