import argparse,os,sys,pickle
from wordgen import WordgenLearned, save_wg, load_wg


parser = argparse.ArgumentParser(description='Generate some words from a pickled WordgenLearned')
parser.add_argument('filename',help='path to pickle file containing the WordgenLearned', type=str, metavar="pickle file")
parser.add_argument('--num_words','-n',
                    nargs='?',
                    help='How many consecutive phones (IPA glyphs) to consider at a time in the learned distribution.',
                    type=int,
                    default=100,
                    metavar="number of words")
parser.add_argument('--min_len','-l',
                    nargs='?',
                    help='Minimum length a word needs to be in order to be displayed.',
                    type=int,
                    default=4,
                    metavar="minimum word length")
try:
  args=parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

wg = load_wg(args.filename)

for _ in range(args.num_words):
  word = wg.generate_word()
  if len(word)>=args.min_len: print(word)

