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

Watch out for the file size of the saved distributions. The `window_size` option has an exponential effect.

## Generating words from saved wordgen data
For detailed help:
```
python generate_some_words.py -h
```
Continuing the example from above, suppose we ended up with the saved object `spanish_text.pkl`. To generate some words:
```
python generate_some_words.py spanish_text.pkl
```

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

## Next

Generate an orthography (that looks nice/reasonable to english speakers) that can be used to display generated words as an alternative to the IPA display.
