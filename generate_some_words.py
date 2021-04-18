import argparse,os,sys,pickle
from wordgen import WordgenLearned, save_wg, load_wg


parser = argparse.ArgumentParser(description='Generate some words from a pickled WordgenLearned')
parser.add_argument('filename',help='path to pickle file containing the WordgenLearned', type=str, metavar="pickle_file_name",nargs=1)
parser.add_argument('--num_words','-n',
                    nargs='?',
                    help='How many words to generate.',
                    type=int,
                    default=100,
                    metavar="number of words")
parser.add_argument('--min_len','-l',
                    nargs='?',
                    help='Minimum length a word needs to be in order to be displayed.',
                    type=int,
                    default=4,
                    metavar="minimum word length")
parser.add_argument('--orth','-o',
                    action='store_true',
                    help='Whether to use orthography_table.csv to convert IPA tokens to some spelling system.',)
args=parser.parse_args()

wg = load_wg(args.filename[0])

num_printed = 0
while num_printed < args.num_words:
    word_orth = None
    if args.orth:
        word, word_orth = wg.generate_word(orthographize=True)
    else:
        word = wg.generate_word()
    if len(word)>=args.min_len:
        line_to_print = word
        if word_orth is not None: line_to_print += '\t' + word_orth
        print(line_to_print)
        num_printed += 1

