# Wordgen 2

Wordgen is a random word generator that can learn from natural language text.
It uses the sound combinations it finds in sample texts to generate new pronounceable words.

Use `learn_distribution.py` at the command line to learn a wordgen object, and then use `generate_words.py` on that wordgen.
Use `merge_wordgens.py` to merge different wordgens and create a totally unique sounding wordgen.

This could be a useful tool for the generation of names or entire vocabularies for fictional settings.

## Example output

Here are some words that were made by merging two wordgens, one trained on an English novel and another trained on a Hungarian novel:

```
anean, wojtsan, kaeat, meret, lahazam, azet, alambar, howrga, andenkart,
porogwa, alrenaɟan, angarn, annon, fakt, nawmea, cwar, cudnakethna, annapeant,
enantajangalesh, cart, anam, wttsthannea, alkanne, anea, lawarjedetta, nahataken, annee
```

Here are some words that were made by merging the English wordgen with one trained on some Polish text:
```
rentnem, uyona, envervrarumi, tam, 'jethov, dutshi, 'amrer, thundayond, wotner, letsh, 
nyebu, tamten,  rojothtuti, tjimurone, thame, dodjer, 'odobje, idzi, rejundaytshe, gyowrdz,
wudnyaw, do'reterano, retsht,  envervye, 'ovdutel, nyemeeni, 'ovdu, nyengu, murun, anto
```

How are these pronounced? There is actually a "correct" pronunciation, if you want it, because the output is produced in IPA.
It's [up to you](#Orthography) how to convert the IPA into a natural looking spelling system. The words above are the result of one such conversion.

## Setup

You need a Python 3 environment with some packages:
```
pip3 install marisa_trie regex unicodecsv numpy
```
For work with English text, you also need [flite 2.0.5](http://tts.speech.cs.cmu.edu/awb/flite-2.0.5-current.tar.bz2) or later. Download that, build it, then navigate to the `testsuite` subdirectory and build the target `lex_lookup`. Then make sure you are working in an environment that can find the compiled `lex_lookup` executable. Example steps for all this:
```
$ tar xjf flite-2.0.5-current.tar.bz2
$ cd flite-2.0.5-current
$ ./configure && make
$ sudo make install
$ cd testsuite
$ make lex_lookup
$ sudo cp lex_lookup /usr/local/bin
```
## Learning a word generator from text

For detailed help:
```
python learn_distribution.py -h
```
Example: suppose you've got some Spanish text in a file `spanish_text.txt`. Then
```
python learn_distribution.py spanish_text.txt spa-Latn --window_size 4
```
would go through each word in the text and look at which 4-letter groups of sounds occur. In order to focus on sounds, wordgen works with the IPA for each word. It determines the IPA using [epitran](https://github.com/dmort27/epitran), and this is why the language code `spa-Latn` needs to be provided. Once the entire file is processed, a family of probability distributions is saved in the file `spanish_text.pkl`. These distributions can be used to generate fake words that sound like they could have come from Spanish.

For a list of languages and epitran language codes, see [the table in the epitran readme](https://github.com/ebrahimebrahim/epitran#transliteration-languagescript-pairs).

Watch out for the file size of the saved distributions. The `window_size` option has an exponential effect.

## Generating words from saved wordgen data
For detailed help:
```
python generate_words.py -h
```
Continuing the example from above, suppose we ended up with the saved object `spanish_text.pkl`. To generate some words:
```
python generate_words.py spanish_text.pkl
```

## Orthography

By default `generate_words.py` will print words in IPA.
To give the words some flavor (e.g. as though they come from some fictional culture) they can be spelled out using some mapping out of IPA into some chosen target symbols. This mapping is specified in `orthography_table.csv`. 

To print this sort of spelling in addition to the IPA, use the flag `-o` when running `generate_words.py`.

Each row in `orthography_table.csv` contains a phoneme (an IPA token) and a space-separated list of possible ways to spell out that phoneme.
When a wordgen object is created (e.g. when `learn_distribution.py` is used),
it chooses for itself an orthography by choosing _one_ of the possible spellings for each phoneme in `orthography_table.csv`.

Make changes in `orthography_table.csv` to set up your own possible "flavors" of spelling. 
For each row in the table, each possible spelling gets chosen with equal probability. You can repeat some possible spellings to give them different weights.

When an IPA token does not have an entry in `orthography_table.csv`, the IPA is defaulted to as the spelling. Feel free to add entries in this case.

Clarification: The choices of possible spellings _do not vary_ with each run of `generate_words.py`.
One set of choices is made when you run `learn_distribution.py` and that set of choices is frozen into the resuling pkl file.
The idea is that a particular and consistent way of spelling things can give some sense of culture.

## Merging word generators

You can merge learned word generators to create some more unique sound distributions.
For detailed help:
```
python merge_wordgens.py -h
```
Suppose that we used `learn_distribution.py` on some spanish text and some hindi text to generate the saved Wordgen objects `spanish_text.pkl` and `hindi_text.pkl`.
Then we can do the following to generate a new distribution that randomly combines elements of spanish and hindi sound combinations:
```
python merge_wordgens.py spanish_text.pkl hindi_text.pkl
```

### How merging works

Wordgens that were learned based on different texts or different languages could involve completely different sets of IPA tokens.
Omitting some tokens isn't so good, because too many sound combinations become impossible and the resulting wordgen starts
to become repetitive with the few sound combinations that were compatible with all the source wordgens.
Unioning all token sets together wouldn't be good either, because it creates a a wordgen with an unreasonable amount of possible sounds.

The solution we use here is to union the token sets together and then cut down the size of the token set by identifying some tokens as equivalent.
In a natural language, different sounds that are treated as equivalent by the language
are called _allophones_.
We need a reasonable way to identify some IPA tokens across different languages as being allophones in a merged language.

Here I have used the [PHOIBLE](https://phoible.github.io/) database of phonological features and allophone pairs.
It allows me to convert each IPA token into a phonological feature vector.
I used PyTorch to train a linear embedding from phonological feature space into a euclidean space such that points that are close together tend to have
come from allophones. Allophone pairs from 1270 different natural languages were used to do this.
By identifying sounds that are close together per the embedding, the large unioned set of IPA tokens can be cut down to a reasonable size.
The code used to produce the embedding can be found in `exploration/exploration4.ipynb`.

Finally, each grouping of n sounds (where n is the `window_size` arg from `learn_distribution.py`) derives its frequency from one of the merged
wordgens chosen at random.

Clarification: The choices of allophone classes, the choices of representative IPA tokens for each allophone class, and the choices of wordgens used for each
n-sound-grouping _do not vary_ with each run of `generate_words.py`. One set of choices is made when you run `merge_wordgens.py` and that set of choices is frozen into the resuling pkl file. Repeated runs of `merge_wordgens.py` can yield different and interesting results each time.

## History

This is a successor to [wordgen](https://github.com/ebrahimebrahim/wordgen).
Where the approach here is statistical, the approach in the first wordgen was
focused on manually coding [phonotactic constraints](https://en.wikipedia.org/wiki/Phonotactics)
for specific languages.
