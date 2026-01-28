import copy
import pdb
import sys, os
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import logging
# BRAID utlities
import braidpy.utilities as utl
import braidpy.lexicon as lex


class _Lexical:
    """
    The _Lexical class is an inner class of the modality class and represents the lexical submodel, either in the orthographic or phonological modality.

    :param eps : float. epsilon value for the lexical representations.
    :param learning : boolean. If True, learning happens in this modality at the end of the simulation.
    :param remove_stim : boolean. if True, remove stim from the lexicon before simulation (in the corresponding modality)
    :param forbid_list : list. list of words to exclude from lexicon
    :param force_app, force_update, force_word : booleans. force the new learning, the updating, or the correct identity of the stimulus.
    :param fMin, fMax : float, allowed frequency in the lexicon
    :param maxItem : int, max number of items in the lexicon
    :param maxItemLen : int, max number of items per length in the lexicon
    :param lenMin : int, minimum length in the lexicon
    :param lenMax : int, maximum length in the lexicon

    :param store : boolean. if True, information of the modality is used during lexicon extraction.
    :param log_freq : boolean. if True, the lexical prior is the log frequency
    :param cat : string. if not None, the grammatical category to be selected in the lexicon
    :param remove_neighbors : boolean. If True, neighbors of the stimulus are excluded from the lexicon
    :param remove_lemma : boolean. If True, words with the same lemma as the stimulus are excluded from the lexicon. (only in Franch or with a lemmatized lexicon)
    :param mixture_knowledge : boolean. if True, representations are mixed between good, bad, uniform
    :param shift : boolean. if True, comparison with +1/-1 length in the lexicon
    """

    def __init__(self, modality, eps=0.001, learning=True, remove_stim=False, forbid_list=None,
                 force_app=False, force_update=False, force_word=False,
                 fMin=0, fMax=sys.maxsize, maxItem=None, maxItemLen=None, lenMin=1, lenMax=11,
                 phlenMin=1, phlenMax=11,
                 store=True, log_freq=True, cat=None, remove_neighbors=False, remove_lemma=True,
                 mixture_knowledge=False, shift=False):

        self.modality = modality
        self.eps = eps
        self.learning = learning
        self.remove_stim = remove_stim
        self.removed = False
        self.removed_words = None
        self.restore = False
        self.old_repr = {}
        self.old_store = {}
        self.forbid_list = forbid_list if forbid_list else []
        self.force_app, self.force_update, self.force_word = force_app, force_update, force_word
        self.leak = 12
        self.dist = {"word": None, "ld": None}
        self.log_freq = log_freq
        # orthographic and phonological representations for all lengths (orthographic length)
        # attention: all_repr in both modalities is indexed by self.N
        self.all_repr = None
        # orthographic and phonological representations for current length
        self.repr = None
        self.lexicon_size = 0
        self.df = None
        self.unknown = "uniform"
        ## Lexical knowledge
        self.fMin, self.fMax, self.maxItem, self.maxItemLen = fMin, fMax, maxItem, maxItemLen
        self.lenMin, self.lenMax, self.phlenMin, self.phlenMax = lenMin, lenMax, phlenMin, phlenMax
        self.remove_neighbors, self.remove_lemma, self.cat, self.store = remove_neighbors, remove_lemma, cat, store
        self.mixture_knowledge = mixture_knowledge
        self.shift, self.shift_begin, self.all_shift_begin = shift, sys.maxsize, 0

        # frequency for words of current length
        self.freq = self.freq_shift = self.N_max = None
        self.extract_lexicon()
        self.handle_languages()
        self.verify_chars()
        self.build_all_repr()


    ###################################
    ##### INIT LEXICON DATAFRAME ######
    ###################################

    def open_lexicon(self):
        """
        Reads a CSV file containing the lexicon to assign it to build the lexicon dataframe. Removes homophones from the same length class by selecting the most frequent one.
        """
        self.df = pd.read_csv(
            os.path.realpath(os.path.dirname(__file__)) + '/../resources/lexicon/' + self.modality.model.lexicon_name,
            keep_default_na=False)
        self.df = self.df.loc[self.df.groupby(['len', 'phlen', 'pron'])['freq'].idxmax()].reset_index(drop=True)
        # self.df = self.df[self.df.len == self.df.phlen]

    def extract_lexicon(self):
        """
        Extracts the lexicon dataframe based on various criteria such as word length, frequency, category, and a list of forbidden words.
        """

        self.df["word"] = self.df.word.str.replace("'", "").replace("-", "").replace(" ", "")
        # self.df.len = self.df.word.str.len()
        if self.cat is not None and 'cat' in self.df.columns:
            self.df = self.df[self.df.cat == self.cat]
        self.df = self.df.assign(store=self.store)
        if self.forbid_list is not None:
            self.df.loc[self.df.word.isin(self.forbid_list), 'store'] = False
        if self.fMin is not None:
            self.df.loc[self.df['freq'] < self.fMin, 'store'] = False
        if self.fMax is not None:
            self.df.loc[self.df['freq'] > self.fMax, 'store'] = False
        # attention si change lenMax, ça change les mots chosit, on peut pas augmenter au fur et à mesure les len
        # par contre on peut augmenter maxItemLen
        self.df['len'] = self.df['len'].astype(int)
        self.df = self.df.set_index('word')
        if self.lenMin is not None and self.lenMax is not None:
            self.df = self.df[(self.df['len'] >= self.lenMin) & (self.df['len'] <= self.lenMax)]
        if self.phlenMin is not None and self.phlenMax is not None:
            self.df = self.df[(self.df['phlen'] >= self.phlenMin) & (self.df['phlen'] <= self.phlenMax)]
        if self.maxItem is not None:
            self.df = self.df.nlargest(self.maxItem, 'freq')
        if self.maxItemLen is not None and (self.maxItem is None or self.maxItemLen < self.maxItem):
            self.df = self.df.groupby('len').apply(
                lambda x: x.sort_values(by='freq', ascending=False).head(self.maxItemLen)).reset_index(0, drop=True)
        self.df = self.df.head(self.maxItem)
        if self.modality.mod == "phono":
            self.modality.N_max = max(self.df.phlen)
        elif self.modality.mod == "ortho":
            self.modality.N_max = max(self.df.len)
        # self.modality.N_max = max(self.df.len)
        # print(f"max length = {self.modality.N_max}")
        self.df = self.df[self.df.columns.intersection(['word', 'freq', 'len', 'idx',
                                                        'store', 'cat', 'phlen'])]
        if self.modality.enabled and len(self.df[self.df.store]) == 0:
            raise ValueError('Incomplete lexicon in the ' + self.modality.mod + 'modality')
        if self.mixture_knowledge:
            np.random.seed(2021)
            self.df['repr_type'] = np.random.choice([0, 0, 0, 0, 0, 1, 1, 1, 2], size=len(self.df))

    @utl.abstractmethod
    def simplify_alphabet(self):
        pass

    @utl.abstractmethod
    def verify_chars(self):
        """
        Checks if all the characters in the lexicon are in the list of characters. If not, it removes the words containing the unknown character from the lexicon
        """
        return

    def handle_languages(self):
        """
        The function handles language-specific operations during the lexicon extraction
        """
        if self.modality.model.langue == "fr":
            self.simplify_alphabet()
        if self.modality.model.langue == "ge":
            self.df.reset_index(inplace=True)
            self.df['word'] = self.df['word'].apply(lambda s: unicodedata.normalize('NFC', s))
            self.df['len'] = self.df.word.str.len()

    def extract_proba(self):
        """
        Extracts the prior probability for words.
        """
        f_type = "freq_log" if self.log_freq else "freq"
        if f_type == "freq_log":
            self.df["freq_log"] = np.log(self.df.freq + 1)
        self.freq = np.array(self.df.sort_values(by='idx')[f_type])

    def change_freq(self, newF=1, string=""):
        """
        Artificially changes the frequency of a word (for the freq effect simulation)
        """
        string = string if len(string) > 0 else self.stim
        if len(string) > 0 and string in self.df.index and newF is not None:
            f = self.df.loc[string].freq
            self.old_freq = f
            self.df.loc[string, 'freq'] = float(newF)
            self.extract_proba()
            wd = self.df.loc[string]
            logging.braid(f"mot : {string}, brut freq= {wd.freq}, old freq= {f}")
        else:
            logging.warning("mot inconnu ou mauvaise fréquence, impossible de changer la fréquence")

    #####################################
    ##### INIT LEXICAL REPRESENTATIONS ##
    #####################################

    def normalize_repr(self, n=2):
        """
        Normalizes word representations according to the measure considered

        :param n: the LX measure to considere, defaults to 2, corresponding to the L2 measure (optional)
        """
        self.all_repr = {k: utl.norm3D(v, n=n) if len(v) > 0 else v for k, v in self.all_repr.items()}

    def build_all_repr(self):
        """
        Creates the list containing 3d matrix of phonological/orthographical representations for each word
        of the lexicon (multilength variation)
        """
        if self.modality.enabled:
            lex = self.df.reset_index()
            if len(lex) == 0:  # empty lexicon
                self.all_repr = []
            else:
                forbid_idx = list(
                    lex[lex['store'] == False].idx) + self.get_forbid_entries()  # words that shouldn't be included
                wds_idx = self.get_all_repr_indices(lex, forbid_idx)
                if self.modality.model.mixture_knowledge:
                    self.all_repr = utl.create_repr_mixt(wds_idx, self.modality.n, self.eps, np.array(lex.repr_type))
                else:
                    try:
                        self.all_repr = utl.create_repr(wds_idx, self.modality.n, self.eps)
                    except:
                        pdb.set_trace()
                if self.modality.model.L2_L_division:
                    self.normalize_repr()
            # print(f"Taille du lexique : {len(self.all_repr)}")
            self.sample_idx = [i for i in range(len(self.all_repr))]


    def get_all_repr_indices(self, lex, forbid_idx=None):
        """
        Gets the string for words of some length that will be used to create the lexical representations.
        """

        wds = [wd.split('_')[0] if i not in forbid_idx else '' for i, wd in enumerate(list(lex["word"]))]
        wds = [wd + '~' * (self.modality.N_max - len(wd)) if i not in forbid_idx else '' for i, wd in enumerate(wds)]
        if len(wds) > 0:
            return self.get_repr_indices(wds)
        else:
            return []

    def get_repr_indices(self, wds):
        wds_idx = np.array(
            [[self.modality.chars.index(letter) if letter in self.modality.chars else -1 for letter in wd]
             if len(wd) > 0 else [-1 if self.unknown == "uniform" else -2] * self.modality.N_max for wd in wds])
        return wds_idx

    def get_forbid_entries(self, string=None):
        """
        Returns the list of the forbidden entries (if there are some).

        :param string: The input string for which we want to find forbidden entries. If no string is provided, it uses the stimulus attribute.
        :return: a list of forbidden entries. If `self.model.remove_neighbors` is True, the function returns a list of words from the lexicon that differ from the
        input string by only one character. If `self.model.remove_lemma` is True, the function returns a list of words with same lemma as the input.
        """

        def isNeighb(w1, w2):
            return (len(w1) == len(w2)) & (sum([i != j for i, j in zip(w1, w2)]) == 1)

        if string is None:
            string = self.modality.model.ortho.stim if self.modality.model.ortho is not None else ''
        try:
            if self.modality.model.remove_neighbors:
                lx = self.modality.model.df.reset_index()
                df = pd.DataFrame({"dist": lx.word.apply(partial(isNeighb, string))})
                return list(lx[df.dist == True].word.values)
            if self.modality.model.remove_lemma:
                lemma = str(self.modality.model.df_lemma.loc[self.modality.model.ortho.stim].lemme)
                liste = list(self.modality.model.df_lemma[self.modality.model.df_lemma.lemme == lemma].index.values)
                liste = [i for i in liste if
                         len(i) == len(self.modality.model.ortho.stim) and i in self.modality.model.df.index]
                return liste if self.remove_stim else [i for i in liste if i != self.modality.model.ortho.stim]
        except:
            return []
        return []

    def build_sample(self, size=None, minsize=50):
        """
        Samples the lexical representations for a given size.

        :param size: the size of the sample
        :return: a 3d matrix of size `size` x `self.N` x `self.n`
        """
        if size==None:
            size = len(self.repr)
        self.modality.word.build_similarity()
        self.modality.word.build_word()
        self.sample_idx = utl.sample_words(self.all_repr, self.modality.word.dist["word"], size, minsize)
        other_mod = self.modality.model.ortho if self.modality.mod == "phono" else self.modality.model.phono
        other_mod.lexical.sample_idx = self.sample_idx

    def set_repr(self, sample=None):
        """
        Sets the current orthographical/phonological knowledge to the length of the current stim
        """
        # print(f"set repr {self.modality.mod}")

        self.repr = self.all_repr
        self.modality.exists = len(self.repr) > 0 and not np.all((self.repr == 0))
        if self.modality.enabled:
            self.repr_norm = utl.calculate_norm3D(utl.norm3D(self.repr, n=1))
            # to avoid problems when performing normalization with 0 distributions
            self.repr_norm = np.array([i if i > 0 else 1 for i in self.repr_norm])
        self.lexicon_size = len(self.repr)
        if self.shift:
            self.shift_begin = self.all_shift_begin[self.modality.N]
        self.extract_proba()

    def remove_stim_repr(self):
        """
        Removes the stimulus from the model's lexicon and from the model's representations
        """
        if self.modality.enabled and self.remove_stim and not self.removed and self.modality.stim in self.df.index:
            wds = list(set(self.forbid_list + [self.modality.stim] if self.remove_stim else []))
            self.removed_words = self.get_word_list_entry(wds)
            for key, raw in self.removed_words.iterrows():
                self.all_repr[raw.idx, :, :] = self.get_empty_percept(
                    1 / self.modality.n if self.unknown == "uniform" else 0)
                self.df.loc[key, 'store'] = False
            self.removed = True
            try:
                self.df.loc[self.modality.stim, 'freq'] = 0
            except:
                pdb.set_trace()

    ############ INIT TOP DOWN #################

    #######################
    ##### LEARNING ########
    #######################

    def learn(self):
        """
        This function updates the lexicon and and the lexical representations (orthographic and phonological) after a simulation.
        """
        if self.learning:
            if self.modality.model.PM:  # creation of new lexical representations
                name = self.modality.stim;
                if self.modality.stim in self.df.index:
                    # handles the case with multiple lexical representations for the same word (add _i add the end)
                    name = self.modality.stim
                    name += '_' + str(
                        len(self.df[self.df.index.str.contains(name + '_') & (~self.df.index.str.contains('~'))]))
                self.add_word(name)
            else:  # update of lexical representations
                # according to selected word during the identification, updates the need to create or update the ortho/phono trace
                # if not wd.ortho and self.ortho.lexical.learning and self.ortho.enabled :
                #    self.ortho.PM = True
                #    self.df.loc[wd.name, 'ortho'] = True
                name = self.modality.word.chosen_word
                self.df.loc[name, 'store'] = True
                self.df.loc[name, 'freq'] += 1
                # if not wd.phono and self.phono.learning and self.phono.enabled :
                #    self.phono.PM = True
                #    self.lexical.df.loc[wd.name, 'store'] = True
            # update of the lexical representations
            self.create_trace() if self.modality.word.PM else self.update_trace(name)
            # self.ortho.create_trace(wd) if self.ortho.PM else self.ortho.update_trace(wd)

    def add_word(self, string):
        """
        This function adds a new word to the lexicon dataframe.

        :param word: The word to be added to the lexicon
        :param pron: The word's pronunciation.
        :param ortho: boolean, if True the orthography of the word is learned.
        :param phono: boolean, if True the phonology of the word is learned.
        :param freq: The frequency of the word.
        """
        # calculation of the index of the word
        idx = int(self.df.count()['len'])
        dico = {"len": len(utl.str_transfo(string)), "freq": 1, "freq_log": np.log(1 + 1),
                "store": True, "idx": int(idx)}
        # append row to the dataframe
        self.df = self.df.append(pd.Series(data=dico, name=string), ignore_index=False)

    @utl.abstractmethod
    def handle_no_learning(self):
        pass

    def create_trace(self, alpha=0.5):
        """
        Creates a new trace in the current modality

        :param wd: the word that is being learned
        :param alpha: the learning rate
        """
        ### Calculation
        if self.modality.enabled and self.learning:
            dist = self.modality.percept.dist["percept"]
            u = self.get_empty_percept()
            knowledge = utl.norm2D(alpha * dist + (1 - alpha) * u, 2 if self.modality.word.L2_L_division else 1)[
                np.newaxis]
        else:
            knowledge = self.handle_no_learning()
        if not self.modality.model.PM:
            wd = self.get_word_entry(self.modality.word.chosen_word)
            self.repr[int(wd.idx)] = knowledge
        elif self.modality.model.shift:
            self.repr = np.concatenate(
                (self.repr[:self.modality.model.shift_begin], knowledge, self.repr[self.modality.model.shift_begin:]),
                0) \
                if len(self.repr) > 0 and knowledge is not None else knowledge
        else:
            self.repr = np.concatenate((self.repr, knowledge), 0) \
                if len(self.repr) > 0 else knowledge
        ### Print
        newTrace = [max(utl.norm1D(knowledge[0][j])) for j in range(np.shape(knowledge)[1])]
        logging.simu(f"New {self.mod} trace")
        logging.simu(f"Trace= {[round(i, 3) for i in newTrace]}\n")
        self.all_repr[self.modality.N - 1] = self.repr

    def update_trace(self, name, alpha=0.9):
        """
        Updates the trace of a word in the lexicon

        :param wd: the word to be updated
        :param alpha: the learning rate
        """
        if self.modality.enabled and self.modality.lexical.learning:
            wd = self.get_word_entry(name)
            p = self.modality.percept.dist["percept"]
            ### Calculation
            learningRate = 1.0 / (5 * wd['freq'] + 1)
            u = self.get_empty_percept()
            newP = learningRate * alpha * p + (1 - learningRate * alpha) * u
            newTrace = utl.norm2D(self.repr[int(wd.idx)] * newP, 1)
            if self.modality.word.L2_L_division:  # on norme les L en L2
                newTrace = utl.norm2D(newTrace, 2)

            ### Update
            self.repr[int(wd.idx)] = newTrace

            ### Print
            logging.simu(f"Update {self.mod} trace : {(wd.name).replace('#', '')}")
            TraceValue = [max(utl.norm1D(self.repr[int(wd.idx), j, :])) for j in range(np.shape(self.repr)[1])]
            logging.simu(f"Trace= {[round(i, 3) for i in TraceValue]}")
            logging.simu(f"New freq = {wd['freq'] + 1}\n")

    #######################
    ##### INFO ############
    #######################

    def pseudodirac(self, word):
        """
        Given a word, returns its Pseudo Dirac distribution, which is a matrix of size $n \times n$ where $n$ is the length of the word, where the $i$th row is the probability distribution of the
        $i$th letter of the word

        :param word: the word we want to get the distribution
        :return: a 2d matrix corresponding to the lexical representation.
        """
        return np.array([[1 - (self.eps * (self.modality.n - 1)) if self.modality.chars[i] == letter else self.eps for i
                          in range(self.modality.n)] for letter in word])

    def get_word_list_entry(self, liste=None, check_store=False):
        """
        If the list is not empty, return the lexicon entries that contain any of the words in the list and have the same length as the current word

        :param liste: a list of words to search for
        :param check_store: boolean. If True, checks, that the word is stored in this modality, i.e. that the colum store is set at True for this word
        :return: A dataframe with the columns of the lexicon and the words that are in the list and have the same length as the word.
        """
        if liste is not None and len(liste) > 0:
            res = self.df[self.df.index.str.startswith(tuple([i + '_' for i in liste])) | self.df.index.isin(liste)]
            if check_store:
                res = res[res.store == True]
        return res

    def get_word_entries(self, string=None, check_store=False):
        """
        If the string is in the lexicon, returns the corresponding row. If not, return None

        :param string: the string to look up in the lexicon
        :param check_store: boolean. If True, the raws returned should have the column 'store' to True.
        """
        string = string if string is not None else self.modality.stim
        if string not in self.df.index:
            return None
        res = self.df[(~self.df.index.str.contains('~')) & (
                (self.df.index == string) | (self.df.index.str.contains(string + '_')))]
        if check_store:
            res = res[res.store == True]
        return res

    def get_word_entry(self, string=None):
        """
        Gets the entry of the lexicon dataframe that corresponds to the stimulus.

        :param string: If not None, the correspondance is made with the string.
        :return: A dataframe that contains relevant entries in the dataframe
        """
        string = string if string else self.modality.stim
        df = self.df[
            (self.df.index == string) | (self.df.index.str.contains(string + '_'))] if string in self.df.index else None
        if isinstance(df, pd.DataFrame):
            df = df.iloc[0]
        return df

    def get_name(self, index):
        """
        Returns the name of a word in a lexicon dataframe based on its index.

        :param index: int. The index of the word in a lexicon dataframe
        :return: string. The name of the word corresponding to the given index in the lexicon dataframe.
        """
        res = self.get_names([index])
        return self.get_names([index])[0] if len(res) > 0 else None

    def get_names(self, indexes):
        """
        Takes a list of indexes and returns the corresponding words

        :param indexes: the indexes of the words to be retrieved
        :return: The words that are being returned are the words that are in the lexicon and have the same length as the ortho.N.
        """
        try:
            # on trie selon l'ordre indiqué par indexes
            res = self.df[(self.df.idx.isin(indexes))]
            res_words = list(res.index.values)
            res_idx = list(res.idx.values)
            return [res_words[res_idx.index(i)] for i in indexes if i in res_idx]
        except IndexError:
            logging.exception("Word index not found")
            pdb.set_trace()

    def get_empty_percept(self, value=None):
        """
        Creates an empty percept with the same dimensions as the percept distribution

        :param value: The value to fill the empty percept with. If no value is provided, it will be filled with equal probabilities.
        :return: returns a numpy array with dimensions `(self.N,self.n)`
        """
        u = np.empty((self.modality.N_max, self.modality.n))
        u.fill(value if value is not None else 1.0 / self.modality.n)
        return u


class _LexicalOrtho(_Lexical):

    def __init__(self, modality, **modality_args):
        self.mod = "ortho"
        super().__init__(modality=modality, **modality_args)

    ###################################
    ##### INIT LEXICON DATAFRAME ######
    ###################################

    def extract_lexicon(self):
        self.open_lexicon()
        self.df["word"] = self.df.word.str.lower()
        super().extract_lexicon()
        self.df["idx"] = self.df.reset_index().index

    def verify_chars(self):
        if self.modality.enabled:
            lexicon_chars = list(set("".join(self.df.reset_index().word)))
            for i in lexicon_chars:
                if i not in list(self.modality.chars) + ['~', '_']:
                    logging.simu(f"Unknown letter in lexicon : {i}")
                    self.df = self.df[~self.df.index.str.contains(i)]

    def simplify_alphabet(self):
        """
        Simplifies the letters in the lexicon dataframe by merging some characters (like à and a).
        """
        self.df = lex.simplify_letters(self.df).set_index('word', drop=True)

    #######################
    ##### INFO ############
    #######################

    def get_ortho_name(self, string=None):
        """
        Gets the orthographic name corresponding to the phonological name.
        :param string: orthographic name
        :return: the phonological name
        """
        data = self.modality.model.phono.lexical.get_word_entry(string)
        if data is not None:
            return self.get_name(data.idx)
        return ""

        return self.get_name(idx)

    ######### Learning ###########

    def handle_no_learning(self):
        ph = self.modality.model.phono
        if ph.learning and ph.enabled and ph.PM:
            # no learning in this modality but learning in the other modality -> adds a 'zero' distribution here
            return self.get_empty_percept(0)[np.newaxis]
        return None


class _LexicalPhono(_Lexical):
    def __init__(self, modality, placement_auto=True, **modality_args):
        self.mod = "phono"
        self.placement_auto = placement_auto
        super().__init__(modality=modality, **modality_args)

    ###################################
    ##### INIT LEXICON DATAFRAME ######
    ###################################

    def handle_languages(self):
        """
        The function handles language-specific operations during the lexicon extraction
        """
        if self.modality.model.langue == 'en':
            # '~' shift already used for word fragments and shifted representations
            self.df.index = self.df.index.str.replace('~', '(')
        super().handle_languages()

    def extract_lexicon(self):
        self.open_lexicon()
        self.df['word'] = self.df.pron
        super().extract_lexicon()
        self.df["idx"] = self.df.reset_index().index

    def verify_chars(self):
        if self.modality.enabled:
            lexicon_chars = list(set("".join(self.df.index)))
            for i in lexicon_chars:
                if i != '~' and i != '_' and i not in self.modality.chars:
                    print(f"Unknown phoneme in lexicon : {i}")
                    self.df = self.df[~self.df.index.str.contains(i)]

    def simplify_alphabet(self):
        """
        Simplifies the phonemes in the lexicon dataframe by merging some characters (like 2 and 9).
        """
        self.df = lex.simplify_phonemes(self.df)  # .set_index('word',drop=True)


    #####################################
    ##### INIT LEXICAL REPRESENTATIONS ##
    #####################################

    def set_repr(self):
        super().set_repr()

    #######################
    ##### INFO ############
    #######################

    def get_phono_name(self, string=None):
        """
        Gets the phonological name corresponding to the orthographic name.
        :param string: orthographic name
        :return: the phonological name
        """
        data = self.modality.model.ortho.lexical.get_word_entry(string)
        # pdb.set_trace()
        if data is not None:
            return self.get_name(data.idx)
        return ""

    #######################
    ####### Learning ######
    #######################

    def handle_no_learning(self):
        ortho = self.modality.model.ortho
        if ortho.learning and ortho.enabled and ortho.PM:
            # no learning in this modality but learning in the other modality -> adds a 'zero' distribution here
            return self.get_empty_percept(0)[np.newaxis]
        return None
