# Wordgen 2

This is a successor to wordgen with a more statistical approach rather than being focused on phonotactic constraints.

It is still in an exploratory phase at the moment.

Ultimately, being able to learn and even generate distributions could make this a useful tool for the generation of languages for fictional settings.

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

## Merging learned word generators to create fictional languages
For detailed help:
```
python merge_wordgens.py -h
```
Suppose that we used `learn_distribution.py` on some spanish text and some hindi text to generate the saved Wordgen objects `spanish_text.pkl` and `hindi_text.pkl`.
Then we can do the following to generate a new distribution that randomly combines elements of spanish and hindi sound combinations:
```
python merge_wordgens.py spanish_text.pkl hindi_text.pkl
```


