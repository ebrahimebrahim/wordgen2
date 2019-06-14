import csv
import numpy as np

class Orthographizer(object):
    """ Converts IPA for a word to orthographic form.
        There are many ways to do this-- an Orthographizer generates one way. """

    def __init__(self):
        self. __orthography_table = {}
        with open('orthography_table.csv','r') as orthography_table_file:
            r = csv.reader(orthography_table_file)
            header = next(r)
            for line in r:
                self.__orthography_table[line[0]] = np.random.choice(line[1].split())

    def orthographize(self,word_vec):
        apply_orthography_dict_if_can = lambda phoneme : self.__orthography_table[phoneme] if phoneme in self.__orthography_table.keys() else phoneme
        return map(apply_orthography_dict_if_can,word_vec)
