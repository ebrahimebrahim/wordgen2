import numpy as np

import os,sys
def add_path_to_local_module(module_name):
  module_path = os.path.abspath(os.path.join(module_name))
  if module_path not in sys.path:
    sys.path.append(module_path)
add_path_to_local_module("epitran")
add_path_to_local_module("panphon")
import panphon
import epitran
import pickle



class Wordgen(object):
  """ Base class for learned and generated random word generators """
  def __init__(self,window_size,lang_name):
    """
      window_size (int) : How many consecutive sounds are considered in a group. 
                          E.g. if this is 3, then the generator picks each sound based on the previous 2.
      lang_name (str)   : Name of language, e.g. "English" or "Fictional language #1"
    """
    if window_size < 2: raise Exception("A window size less than 2 does not make sense.")
    self.window_size = window_size
    self.lang_name = lang_name

  def get_ipa_tokens(self):
    raise NotImplementedError

  def token_to_int(self,token):
    raise NotImplementedError

  def int_to_token(self,i):
    raise NotImplementedError

  def get_distribution(self):
    raise NotImplementedError

  def generate_word(self):
    raise NotImplementedError




class SuppressedMessenger(object):
  """ A class to output messages with output being supressed at a certain point."""
  def __init__(self,name,max_messages):
    self.name = name
    self.num_printed = 0
    self.max_messages = max_messages
    self.stopped_printing = False if max_messages > 0 else True

  def print(self,msg):
    if self.num_printed < self.max_messages:
      print(msg)
      self.num_printed += 1
    elif not self.stopped_printing:
      print("[Further output regarding "+self.name+" will be suppressed]")
      self.stopped_printing = True



class WordgenLearned(Wordgen):
  """ Learn a random word generator based on sounds that appear in a text file """
  def __init__(self,window_size,lang_code):
    """
      window_size (int) : How many consecutive sounds are considered in a group. 
                          E.g. if this is 3, then the generator picks each sound based on the previous 2.
      lang_code (str)   : Language code, e.g. "eng-Latn". See epitran/README.md for details on language code.
    """
    super().__init__(window_size=window_size, lang_name=lang_code)

    sys.stdout.write("Loading Epitran with language code "+lang_code+"... ")
    sys.stdout.flush()
    self._epi = epitran.Epitran(lang_code)
    # TODO if the choice of language was one of the bad ones that epitran can try to do but sucks at.... then warn the user here
    print("success!")

    self._ipa_tokens = self.load_ipa_chars(lang_code)
    self._ipa_tokens.add('WORD_START')
    self._ipa_tokens.add('WORD_END')
    self._int_to_token = dict(enumerate(self._ipa_tokens))
    self._token_to_int = {v:k  for k,v in self._int_to_token.items()}
    self._distribution = None

    
    
  def load_ipa_chars(self,lang_code):
    """Return set of characters that epitran will use for phonemes for the given language code"""
    if lang_code in epitran.Epitran.special:
      if lang_code == "eng-Latn":
        flite  = epitran.flite.Flite()
        ipa_chars = set(flite._read_arpabet("epitran/epitran/data/arpabet.csv").values())
      else:
        raise NotImplementedError("load_ipa_chars still does not know how to handle this language!")
    else:
      simple = epitran.simple.SimpleEpitran(lang_code)
      ipa_chars=set(p for l in simple._load_g2p_map(lang_code,False).values() for p in l)
    if '' in ipa_chars : ipa_chars.remove('')
    ipa_chars_segmented = set()
    ft = panphon.featuretable.FeatureTable()
    for p in ipa_chars: ipa_chars_segmented.update(ft.segs_safe(p))
    return ipa_chars_segmented

  def get_ipa_tokens(self):
    """ Return set of ipa tokens, including word start and end tokens """
    return self._ipa_tokens

  def token_to_int(self,token):
    return self._token_to_int[token]

  def int_to_token(self,i):
    return self._int_to_token[i]

  def learn_distribution(self,filename):
    """ Go through txt file and count chunks of sounds appearing in words, learning a distribution to generate from. """

    print("About to learn from",filename)
    print("For each word, each chunk of",self.window_size,"sounds will be considered.")
    print("Some of the words that could not be processed will be printed below; just check that nothing too bad is happening.")

    s_msg = SuppressedMessenger(name="unprocessed words",max_messages = 30)

    def extract_words(line):
      for word in line.split():
        word_ipa = self._epi.trans_list(word.strip(' .()!:;,\n'))
        if word_ipa and all(c in self.get_ipa_tokens() for c in word_ipa):
          yield map(self.token_to_int,word_ipa)
        else: s_msg.print("\""+word+"\" was not processed.")

    num_tokens  = len(self.get_ipa_tokens())
    start_token = self.token_to_int('WORD_START')
    end_token   = self.token_to_int('WORD_END')

    counts = np.zeros((num_tokens,)*self.window_size,dtype=np.dtype('u8')) # TODO use sparse array instead and be smarter about dtype.

    with open(filename) as f:
      for line_num,line in enumerate(f.readlines()):
        for word in extract_words(line):
          previous = [start_token]*(self.window_size-1)
          for t in word:
            counts[tuple(previous)+(t,)] += 1
            previous = previous[1:]+[t]
          counts[tuple(previous)+(end_token,)] += 1
        sys.stdout.write("> "+str(line_num+1)+" lines processed                   \r")
        sys.stdout.flush()
      print()
                  
    totals = counts.sum(axis=self.window_size-1)
    self._distribution = counts / (np.vectorize(lambda x : x if x!=0 else 1)(totals.reshape(totals.shape+(1,))))

  def get_distribution(self):
    """ Return distribution array D.
        Let w be the window_size.
        For integers i_1,...,i_w (which enumerate some IPA tokens),
        D[i_1,...,i_w] is the probability that, given the consecutive sounds represented by i_1 ... i_{w-1} appearing in a word, that the next sound is i_w.
        
        The distribution should be learned from some text data before this method gets called."""
    if self._distribution is None:
      raise Exception("Attempted to access distribution before it was instantiated. In this case, you probably wanted to learn_distribution first.")
    else:
      return self._distribution

  def generate_word(self):
    num_tokens  = len(self.get_ipa_tokens())
    previous = [self.token_to_int('WORD_START')]*(self.window_size-1)
    word = []
    while previous[-1]!=self.token_to_int('WORD_END'):
      if self.get_distribution()[tuple(previous)].sum()==0:
        next_char = np.random.choice(range(num_tokens))
        print("Uh oh! This shouldn't happen, right?",previous) # TODO: Decide whether this is allowed or should yield a proper error
      else:
        next_char = np.random.choice(range(num_tokens),p=self.get_distribution()[tuple(previous)])
      previous = previous[1:]+[next_char]
      word.append(next_char)
    return ''.join(self.int_to_token(c) for c in word[:-1])

  def get_distribution_sparsity(self):
    """ Return sparsity of distribution tensor; i.e. the proportion of entries that are zero. """
    d=self.get_distribution()
    return 1-np.count_nonzero(d)/float(d.size)



def save_wg(wg,filename):
  with open(filename,'wb') as pickle_file:
    pickle.dump(wg, pickle_file)

def load_wg(filename):
  with open(filename,'rb') as pickle_file:
    wg = pickle.load(pickle_file)
  return wg
