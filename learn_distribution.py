# Idea:
# ./learn_distribution.py peuptext.txt -w 3 -l eng-Latn
# this learns and saves distribution in peuptext.pkl
# the window size can be chosen
# the saved distribution is the array (sparse) along with the glyph dict, original language code, and window size
# show a progress bar, and also show a sampling of those words that are getting dropped because they contained things not in the set of allowed glyphs
# at first the wordgen2 classes will be here, but then you can move them to wordgen2.py for use in the later language generator

import argparse,os,sys,pickle
from wordgen import WordgenLearned, save_wg, load_wg


parser = argparse.ArgumentParser(description='Learn distribution of sounds that tend to go together given some text in a particular language.')
parser.add_argument('filename',help='path to text file containing some text in a certain language', type=str, metavar="filename")
parser.add_argument('lang_code',
                    help='Language code, for example \"eng-Latn\". See epitran/README.md for more on the language code.',
                    type=str,
                    metavar=("language_code",))
parser.add_argument('--window_size','-w',
                    nargs='?',
                    help='How many consecutive phones (IPA glyphs) to consider at a time in the learned distribution. Default is 3.',
                    type=int,
                    default=3,
                    metavar="window size")
try:
  args=parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)


wg = WordgenLearned(window_size = args.window_size, lang_code = args.lang_code)
wg.learn_distribution(args.filename)

print("\nSome genrated words:\n")
for _ in range(100):
  word = wg.generate_word()
  if len(word)>4: print(word)

filename_pkl = os.path.splitext(args.filename)[0]+".pkl"
save_wg(wg,filename_pkl)

