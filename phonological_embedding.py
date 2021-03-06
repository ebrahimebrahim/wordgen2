import numpy as np
import pickle

class PhonologicalEmbedding(object):
    """ Manages conversion of phoible ipa segments to points in euclidean space.
        Points tend to be close when those ipa segments tend to be allophones.
        The actual embedding was learned in exploration4.ipynb. """

    __embedding_matrix = None  # In any runtime we should load up at most one of these, so it's a static member variable
    __to_phoible_feats_dict = None
    __epitran_phoible_replacements = {
        'd͡ʒ':'d̠ʒ',
        't͡ʃ':'t̠ʃ',
        't͡ɕ':'tɕ',
        't͡s':'ts',
        'd͡ʑ':'dʑ',
        'ɕːʲ':'ʃʲ',
        'ʂʲː':'ʃʲ',
        'tɕʲ':'t̠ʃʲ',
        't͡ɬ':'tɬ',
        'p͡f':'pf',
        'd͡z':'dz',
    }

    def __init__(self):
        if PhonologicalEmbedding.__embedding_matrix is None or PhonologicalEmbedding.__to_phoible_feats_dict is None:
            with open("phonological_embedding_data.pkl",'rb') as pickle_file:
                PhonologicalEmbedding.__to_phoible_feats_dict , PhonologicalEmbedding.__embedding_matrix = pickle.load(pickle_file)
        self.embed_dim = len(list(PhonologicalEmbedding.__to_phoible_feats_dict.values())[0])


    def epitran_to_phoible(self,epitran_ipa):
        phoible_ipa = epitran_ipa
        for a,b in PhonologicalEmbedding.__epitran_phoible_replacements.items():
            phoible_ipa = phoible_ipa.replace(a,b)
        return phoible_ipa

    def to_phoible_fts(self,ipa_seg):
        """ Convert a single ipa segment to a numpy array of phoible features, ready for embedding matrix to be applied to it.
            ipa_seg (string) : a single unicode ipa segment as would appear in phoible or as could be output by epitran """
        ipa_seg = self.epitran_to_phoible(ipa_seg)
        if ipa_seg not in PhonologicalEmbedding.__to_phoible_feats_dict.keys():
            raise KeyError("The ipa segment "+str(ipa_seg)+" was not found in the phoible ipa-to-features dict. "+\
                           "We use phoible data to work with features, while we use epitran to generate transliterations. "+\
                           "Even though both stick to a strict standard, IPA in unicode, they sometimes have different representations "+\
                           "which can cause this error.\n Consider writing an exception into PhonologicalEmbedding.__epitran_phoible_replacements.")
        return np.array(PhonologicalEmbedding.__to_phoible_feats_dict[ipa_seg],dtype='float32')

    def embed(self,ipa_seg):
        """ Convert a single ipa segment to a numpy array representing point in euclidean space in which closeness corresponds to likelihood to be allophones.
            ipa_seg (string) : a single unicode ipa segment as would appear in phoible or as could be output by epitran """
        pre_embedded = self.to_phoible_fts(ipa_seg)
        return np.matmul(PhonologicalEmbedding.__embedding_matrix,pre_embedded)

    def dist(self,ipa_seg1,ipa_seg2):
        """ Embed two ipa segments and measure their euclidean distance; this should correspond to their tendency to be allophones.
            ipa_seg1 (string) : a single unicode ipa segment as would appear in phoible or as could be output by epitran
            ipa_seg2 (string) : as above """
        v1 = self.embed(ipa_seg1)
        v2 = self.embed(ipa_seg2)
        return np.sqrt(((v1-v2)**2).sum())

    def nearest(self,ipa_seg,segs):
        """ Determine the ipa segment from a particular set of segs which is nearest to a given ipa segment

            ipa_seg (string) : a single unicode ipa segment as would appear in phoible or as could be output by epitran
            segs (iterable of strings)  : set of segments in which to search for nearest neighbor 
            
            Returns string, the element of segs which is nearest to ipa_seg according to euclidean distance after the embedding. """

        if ipa_seg in segs: return ipa_seg
        return min(segs,key=lambda s : self.dist(ipa_seg,s))
