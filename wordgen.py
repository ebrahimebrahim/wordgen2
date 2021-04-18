import numpy as np
from itertools import product
import os,sys
def add_path_to_local_module(module_name):
    module_path = os.path.abspath(os.path.join(module_name))
    if module_path not in sys.path:
        sys.path.append(module_path)
add_path_to_local_module("epitran")
add_path_to_local_module("panphon")
from suppressed_messenger import SuppressedMessenger
import panphon
import epitran
import pickle
import phonological_embedding, orthographizer



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
        self.orthographizer = orthographizer.Orthographizer()
    
    def get_ipa_tokens(self):
        """ Return set of ipa tokens, including word start and end tokens """
        return self._ipa_tokens
  
    def token_to_int(self,token):
        return self._token_to_int[token]
  
    def int_to_token(self,i):
        return self._int_to_token[i]
  
    def get_distribution(self):
        """ Return distribution array D.
            Let w be the window_size.
            For integers i_1,...,i_w (which enumerate some IPA tokens),
            D[i_1,...,i_w] is the probability that, given the consecutive sounds represented by i_1 ... i_{w-1} appearing in a word, that the next sound is i_w.
            
            The distribution should be learned from some text data before this method gets called."""
        if self._distribution is None:
            raise Exception("Attempted to access distribution before it was instantiated.")
        else:
            return self._distribution
  
    def generate_word(self,orthographize = False):
        """ Use distribution to randomly generate a word. 
            
            orthographize (bool) : If True then will use this wordgen's orthography to also print a version of the word

            Returns either the IPA of the generated word,
                    or both the IPA of the generated word as well as a spelled out version using orthography"""
        num_tokens  = len(self.get_ipa_tokens())
        previous = [self.token_to_int('WORD_START')]*(self.window_size-1)
        word = []
        word_len = 0
        while previous[-1]!=self.token_to_int('WORD_END'):
            if self.get_distribution()[tuple(previous)].sum()==0:
                next_char = np.random.choice(range(num_tokens))
                print("Uh oh! This shouldn't happen, right?",previous) # TODO: Decide whether this is allowed or should yield a proper error
            else:
                next_char = np.random.choice(range(num_tokens),p=self.get_distribution()[tuple(previous)])
            previous = previous[1:]+[next_char]
            word.append(next_char)
            word_len += 1
            if word_len > 1000 :
                print("We appear to be stuck generating this word. Stopping!")
                break
        word_vec = [self.int_to_token(c) for c in word[:-1]]

        if orthographize:
            return ''.join(word_vec),''.join(self.orthographize(word_vec))
        else:
            return ''.join(word_vec)

    def orthographize(self,word_vec): #TODO Make this a class with orthography dict as static member var. Current approach is garbage.
        return self.orthographizer.orthographize(word_vec)
  
    def get_distribution_sparsity(self):
        """ Return sparsity of distribution tensor; i.e. the proportion of entries that are zero. """
        d=self.get_distribution()
        return 1-np.count_nonzero(d)/float(d.size)
  


class WordgenMerged(Wordgen):
    """ Merge several learned word generators to create a word generator for a fictional language.
        Use merge_languages method to initialize. """

    def merge_langauges(self,learned_gens):
        """
            Merge some learned word generator distributions to create a new one.
            The IPA tokens of the learned distributions will be unioned together, then
            they will be grouped into equivalence classes (kind of like allophones) with chosen representatives
            to constitute the sounds of the new language.
            Then the distribution of segment clusters will be created by randomly pooling together information
            from the distributions of the learned languages.

            learned_gens (list of WordgenLearned) : a nonempty list of learned word generators.
        """
        
        assert(all(gen.window_size == self.window_size for gen in learned_gens))

        # Load in the fixed phonological embedding. This was created in exploration4.ipynb
        ph_embed = phonological_embedding.PhonologicalEmbedding()

        # First gather all ipa_tokens
        all_ipa_tokens = set()
        for gen in learned_gens:
            all_ipa_tokens.update(gen.get_ipa_tokens())

        # Remove any bad tokens that phoible didn't like for the next part. These are not good tokens for a wordgen anyway.
        # We will also remove the start and end tokens; they will be brought back in later
        bad_tokens = ['ː','̃','ʲ','WORD_START','WORD_END']
        for t in bad_tokens: 
            if t in all_ipa_tokens: 
                all_ipa_tokens.remove(t)

        # all_ipa_tokens will from now on be an ordered list
        all_ipa_tokens = list(all_ipa_tokens)
        num_all_tokens = len(all_ipa_tokens)

        # Compute max and min distances tokens have from each other via the feature embedding
        dists = []
        for c1 in all_ipa_tokens:
            for c2 in all_ipa_tokens:
                if c1 != c2 : dists.append(ph_embed.dist(c1,c2))
        max_dist,min_dist = max(dists),min(dists)

        # Create equivalence classes of phonemes to cut down number of sounds
        projection = [n for n in range(num_all_tokens)] # we start with identity mapping and will gradually identify things
        # think of projection as mapping from indices representing ipa_chars to equivalence classes
        # the number of equivalence classes is len(set(projection))
        M = int(np.random.normal(35,10)) # choose number of sounds
        step_size = (max_dist-min_dist)/float(num_all_tokens*500) # this is just some small step size for the loop below. we just want to be certain the loop terminates
        spread = (max_dist-min_dist)/20.
        for r0 in np.arange(min_dist,max_dist,step_size):
            r = max(step_size,np.random.normal(r0,spread))
            ipa_char_index = np.random.randint(num_all_tokens)
            s0 = all_ipa_tokens[ipa_char_index]
            for n in range(num_all_tokens):
                s = all_ipa_tokens[n]
                if ph_embed.dist(s,s0)<r:
                    projection[n]=projection[ipa_char_index]
            if len(set(projection))<=M: break

        # Choose a representative for each equivalence class to serve as the standard/official pronunciation for the new language
        direction = np.array([np.random.normal() for _ in range(ph_embed.embed_dim)])
        section = {} # Think of section as a map back from the codomain of projection to the domain which picks a rep of each equivalence class
        for p in set(projection):
            equiv_class = [n for n in range(num_all_tokens) if projection[n]==p]
            rep = max(equiv_class,key=lambda n : np.dot(ph_embed.embed(all_ipa_tokens[n]),direction).item())
            for n in equiv_class: section[n] = rep
            # print(all_ipa_tokens[rep],[all_ipa_tokens[n] for n in equiv_class])
        
        # Set up dict mapping ipa tokens from representative tokens to list of tokens in all_ipa_tokens that are equivalent to it
        token_to_equivclass = {}
        for rep_index in set(section.values()):
            equiv_class_indices = [n for n in range(num_all_tokens) if projection[n]==projection[rep_index]]
            equiv_class_tokens = [all_ipa_tokens[n] for n in equiv_class_indices]
            token_to_equivclass[all_ipa_tokens[rep_index]] = equiv_class_tokens
        token_to_equivclass['WORD_START']=['WORD_START']
        token_to_equivclass['WORD_END']=['WORD_END']
        self.token_to_equivclass = token_to_equivclass 

        ipa_tokens = list(token_to_equivclass.keys())
        num_tokens = len(ipa_tokens)
        self._ipa_tokens = set(ipa_tokens)
        self._int_to_token = {n:ipa_tokens[n] for n in range(num_tokens)}
        self._token_to_int = {val:key for key,val in self._int_to_token.items()}



        # Alternative approach:

        def token_to_nearest_int(token,gen): # Find index (w.r.t. gen) corresponding to the token of gen that is nearest to the given token
            if token in ['WORD_START','WORD_END']: return gen.token_to_int(token)
            segs = gen.get_ipa_tokens()
            segs.difference_update(['WORD_START','WORD_END'])
            segs.difference_update(bad_tokens) # TODO there should be a more elegant way to handle bad_tokens
            return gen.token_to_int(ph_embed.nearest(token,segs))

        # This latter function would have to be called many times on the same inputs, so we cache the needed outputs here:
        token_to_nearest_int_dicts = {}
        for gen in learned_gens:
            token_to_nearest_int_dicts[gen] = {}
            for token in self._ipa_tokens:
                token_to_nearest_int_dicts[gen][token] = token_to_nearest_int(token,gen)
        

        self._distribution = np.zeros((num_tokens,)*self.window_size,dtype=np.dtype('float32'))
        for token_indices_init in product(range(num_tokens),repeat=self.window_size-1):
            for token_index_last in range(num_tokens):

                token_indices = token_indices_init + (token_index_last,)

                tokens = [self.int_to_token(i) for i in token_indices]
                equiv_classes = [token_to_equivclass[t] for t in tokens]

                gen = np.random.choice(learned_gens)
                indices_to_use = tuple(token_to_nearest_int_dicts[gen][t] for t in tokens)

                self._distribution[token_indices] = gen.get_distribution()[indices_to_use]


        # Now the generator we created could create some impossible combinations, so here is a helper function to clear them out.

        def clear_out_impossible_combos(D, t_to_i):
            """Given a distribution D, make one pass through the entries and clear out
               impossible combinations by zeroing probabilities.
               More precisely (assuming window size of 3 in the notation here):

               For each i,j, we ensure that if D[i,j,k] is zero for all k,
               then D[l,i,j] is also zero for all l.

               After one iteration through all i,j, more impossible combinations may be created,
               so this function would have to be called repeatedly until that is done.

               Arguments:
                   D is the wordgen distribution
                   t_to_i is the dict mapping IPA tokens to indices

               Return number of nonzero D[l,i,j] found (and repaired) for which D[i,j,k] was zero for all k"""

            num_impossibles_found = 0

            w = len(D.shape) # win size
            n = D.shape[0] # num_tokens
            for token_indices in product(range(n),repeat=w-1): # For each i,j
                if t_to_i['WORD_END'] in token_indices:
                    continue
                if D[token_indices].sum()<=0: # if D[i,j][k] is zero for all k
                    for l in range(n):
                        if D[(l,)+token_indices]!=0:
                            num_impossibles_found += 1
                            D[(l,)+token_indices]=0

            return num_impossibles_found


        def clear_out_useless_combos(D, t_to_i):
            """Given a distribution D, make one pass through the entries and clear out
               useless combinations by zeroing probabilities.
               More precisely (assuming window size of 3 in the notation here):

               For each i,j, we ensure that if D[l,i,j] is zero for all l,
               then D[i,j,k] is also zero for all k.

               After one iteration through all i,j, more useless combinations may be created,
               so this function would have to be called repeatedly until that is done.

               Arguments:
                   D is the wordgen distribution
                   t_to_i is the dict mapping IPA tokens to indices

               Return number of nonzero D[l,i,j] found (and repaired) for which D[i,j,k] was zero for all k"""

            num_found = 0

            w = len(D.shape) # win size
            n = D.shape[0] # num_tokens
            for token_indices in product(range(n),repeat=w-1): # For each i,j
                if all(i==t_to_i['WORD_START'] for i in token_indices):
                    continue
                if all(D[(l,)+token_indices] <= 0 for l in range(n)) : # if D[l][i,j] is zero for all l
                    for k in range(n):
                        if D[token_indices+(k,)]!=0:
                            num_found += 1
                            D[token_indices+(k,)]=0
                            # print("Removed useless combo",[self.int_to_token(t) for t in token_indices])

            return num_found

        num_impossibles_found = 1
        num_useless_found = 1
        while num_impossibles_found + num_useless_found != 0:
            num_impossibles_found = clear_out_impossible_combos(self._distribution,self._token_to_int)
            num_useless_found = clear_out_useless_combos(self._distribution,self._token_to_int)
            print(num_impossibles_found,"impossible entries zeroed.")
            print(num_useless_found,"useless entries zeroed.")


        # The distribution still needs to be normalized:

        totals = self._distribution.sum(axis=self.window_size-1)
        self._distribution = self._distribution / (np.vectorize(lambda x : x if x!=0 else np.float32(1))(totals.reshape(totals.shape+(1,))))


    



class WordgenLearned(Wordgen):
    """ Learn a random word generator based on sounds that appear in a text file """
    def __init__(self,window_size,lang_code):
        """
          window_size (int) : How many consecutive sounds are considered in a group. 
                              E.g. if this is 3, then the generator picks each sound based on the previous 2.
          lang_code (str)   : Language code, e.g. "eng-Latn". See epitran/README.md for details on language code.
        """
        super().__init__(window_size=window_size, lang_name=lang_code)

        sys.stdout.write("Verifying that Epitran can be loaded with language code "+lang_code+"... ")
        sys.stdout.flush()
        epi = epitran.Epitran(lang_code)
        print("... it works!")
        if lang_code in ['ara-Arab','cat-Latn','ckb-Arab','fas-Arab','fra-Latn','fra-Latn-np','mya-Mymr','por-Latn']:
            print("BUT ... epitran has limited support for this language due to its ambiguous orthography. This might not work too well.")
        epi = None
    
        self.lang_code = lang_code

        # These will be initialized later by load_ipa_tokens
        self._ipa_tokens = None
        self._int_to_token = None
        self._token_to_int = None
        self._distribution = None

      
    def load_ipa_tokens(self,filename,words_to_try = 10000):
        """ Inititalize set of IPA tokens that can come up based on transliterating a bunch of words and looking at what phonemes come up.
            Only "pure" words are counted for this; i.e. those words in which nothing was skipped during transliteration because
            it didn't appear in epitran's mapping.

            This will also initialize an int-token mapping to be used to interpret the distribution tensor.

            filename (str) : File containing words in the target language. 
                             Can be the same as the file from which distribution will be learned
            words_to_try (int) : Number of pure words to transliterate in order to build set of ipa tokens.
                                 If this is too small, the set of tokens might be incomplete and some sound combinations will be skipped
                                 while learning the distribution. If this is too large, it might take unnecessarily long time to build
                                 the set of IPA tokens.
         """




        def ensure_all_segmented(possibly_unsegmented_strings):
            """ Use panphon on all elements of given set of ipa strings to split them into what panphon
                considers to be "segments", in case they were not already split up like that. """
            ft = panphon.featuretable.FeatureTable()
            definitely_segmented_strings = set()
            for p in possibly_unsegmented_strings: definitely_segmented_strings.update(ft.segs_safe(p))
            return definitely_segmented_strings


        if self.lang_code in epitran.Epitran.special:
            if self.lang_code == "eng-Latn":
                flite  = epitran.flite.Flite()
                ipa_tokens = set(flite._read_arpabet("epitran/epitran/data/arpabet.csv").values())
                if '' in ipa_tokens : ipa_tokens.remove('')
                ipa_tokens = ensure_all_segmented(ipa_tokens)
            else:
                raise NotImplementedError("load_mapping_phonemes not know how to handle this language!")

        else:
            # First let's see what are phonemes that epitran has in its grapheme-to-phomeme mapping
            # It will be good to make sure at the end that at least these end up in the set of ipa tokens
            # These alone are not necessarily enough due to the pre- and post- processing that can introduce
            # other more IPA segments than appear in that mapping.
            simple = epitran.simple.SimpleEpitran(self.lang_code)
            mapping_phonemes_raw = set(p for l in simple._load_g2p_map(self.lang_code,False).values() for p in l)
            if '' in mapping_phonemes_raw : mapping_phonemes_raw.remove('')
            mapping_phonemes = ensure_all_segmented(mapping_phonemes_raw)
                
            # Next, we look at words in the given file and transliterate them and pick out phonemes
            print("Gathering phonemes from",filename)
            epi = epitran.Epitran(self.lang_code)
            num_words = 0
            ipa_tokens = set()
            with open(filename) as f:
                for line in f.readlines():
                    for word in line.split():
                        word = word.strip(' .()!:;,\n')
                        if simple.check_purity(word):
                            ipa_tokens.update(epi.trans_list(word))
                            num_words += 1
                            sys.stdout.write("> "+str(num_words)+" words processed                   \r")
                            sys.stdout.flush()
                    if num_words >= words_to_try : break
                print()
    
            for m in mapping_phonemes:
                if m not in ipa_tokens:
                    print("Warning: The symbol",m,"was spotted in epitran's grapheme-phoneme mapping,",
                           "but was never encountered in the text and is therefore being omitted from the set of ipa tokens.")

    
        # Inititalize attributes
        self._ipa_tokens = ipa_tokens
        self._ipa_tokens.add('WORD_START')
        self._ipa_tokens.add('WORD_END')
        self._int_to_token = dict(enumerate(self._ipa_tokens))
        self._token_to_int = {v:k  for k,v in self._int_to_token.items()}


 
  
  
    def learn_distribution(self,filename):
        """ Go through txt file and count chunks of sounds appearing in words, learning a distribution to generate from. """


        if self._ipa_tokens is None:
            self.load_ipa_tokens(filename)

    
        print("About to learn from",filename)
        print("For each word, each chunk of",self.window_size,"sounds will be considered.")
        print("Some of the words that could not be processed will be printed below; just check that nothing too bad is happening.")
    
        s_msg = SuppressedMessenger(name="unprocessed words",max_messages = 1000)

        epi = epitran.Epitran(self.lang_code)
    
        def extract_words(line):
            for word in line.split():
                word_ipa = epi.trans_list(word.strip(' .()!:;,\n'))
                if word_ipa and all(c in self.get_ipa_tokens() for c in word_ipa):
                    yield map(self.token_to_int,word_ipa)
                else:
                    bad_chars = [c for c in word_ipa if c not in self.get_ipa_tokens()]
                    if bad_chars and any(c not in '1234567890-()[]{}=@' for c in bad_chars): # not useful to see this warning in most cases
                        s_msg.print("\""+word+"\" was not processed due to: "+str(bad_chars))
    
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
  



def save_wg(wg,filename):
    with open(filename,'wb') as pickle_file:
        pickle.dump(wg, pickle_file)

def load_wg(filename):
    with open(filename,'rb') as pickle_file:
        wg = pickle.load(pickle_file)
    return wg
