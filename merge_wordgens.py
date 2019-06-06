import argparse,os,sys,pickle
from wordgen import WordgenLearned, WordgenMerged, save_wg, load_wg


parser = argparse.ArgumentParser(description='Merge some WordgenLearned objects to create a new language.')
parser.add_argument('files',
                    nargs='*',
                    help='List of pickled WordgenLearned objects for various languages to be merged.\
                          These files could be produced using the learn_distribution script.',
                    metavar="pickle files list")
parser.add_argument('--name','-n',
                    nargs='?',
                    help='A name for the new language',
                    default='a fictional language made by merge_wordgens.py',
                    metavar="new language name")
parser.add_argument('--output','-o',
                    nargs='?',
                    help='Output filename for the pickled WordgenMerged object.',
                    default='merge_wordgens_output.pkl',
                    metavar="output pickle file name")
args=parser.parse_args()


if not args.files :
    print("Please provide at least 1 file to work with.")
    print()
    parser.print_help()
    sys.exit(1)

wgs_learned = [load_wg(filename) for filename in args.files]
window_size = wgs_learned[0].window_size
if not all(wg.window_size==window_size for wg in wgs_learned):
    print("All provided WordgenLearned objects should have the same window_size!")
    sys.exit(1)
wgm = WordgenMerged(window_size,args.name)
wgm.merge_langauges(wgs_learned)

print("Here are the sounds of the new language, and the way in which they've been identified.")
print("For each sound, the list next to it is the list of sounds that are treated as being the same sound for the purpose of the merging.\n")
for seg,equivclass in wgm.token_to_equivclass.items():
    if seg in ['WORD_START','WORD_END'] : continue
    print(seg,'\t',equivclass)

print('\n\nHere are a few words from the new language:\n')
for _ in range(100):
    word = wgm.generate_word()
    sys.stdout.write(word+', ')
print('\n')

print("Saving Wordgen object to",args.output)
save_wg(wgm,args.output)
