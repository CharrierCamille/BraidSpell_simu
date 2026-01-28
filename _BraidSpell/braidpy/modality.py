# General purpose libraries
import copy
import math

from functools import partial

import logging
import os
## debugging
import pdb
from time import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
# Scientific/Numerical computing
import pandas as pd
from scipy.stats import entropy
import braidpy.utilities as utl
import braidpy.sensor as sens
import braidpy.lexical as lxc
import braidpy.percept as per
import braidpy.attention as att
import braidpy.word as wrd


class _Modality:
    """
    The _Modality class is an inner class of the model class and represents either an orthographic or phonological modality.
    """

    def __init__(self, model, mod, other_mod, dist, stim=None, chars_filename="", enabled=True, **modality_args):
        """
        :param model : braid model instance
        :param mod : string. the class modality (orthographic or phonological)
        :param other_mod : string. the other modality than the class (orthographic or phonological)
        :param dist : dict of probability distributions. Keys are the distribution names, among self.model.dist_names
        :param stim : string corresponding to the stimulus
        :param chars_filename : string. name of the character file
        :param enabled : boolean. if True, the modality is 'activated'
    """
        self.param = {}
        self.param["sensor"] = ['gaze', 'conf_mat_name', 'scaleI', 'scalePhi', 'slopeG', 'crowding']
        self.param["percept"] = ['leak', 'top_down','gamma_ratio_p','decoding_ratio_op', 'decoding_ratio_po']
        self.param["word"] = ['leakW', 'L2_L_divison', 'ld_thr', 'word_reading', 'ld_weight', 'gamma_ratio_w']
        self.param["lexical"] = [ 'eps', 'learning', 'remove_stim', 'force_app', 'force_update',
                                 'force_word', 'fMin', 'fMax',
                                 'store', 'log_freq', 'cat', 'maxItem', 'maxItemLen', 'lenMin', 'lenMax',
                                  'phlenMin', 'phlenMax',
                                 'remove_neighbors', 'remove_lemma', 'mixture_knowledge', 'shift']
        self.param["attention"] = ['Q', 'sd', 'sdM', 'mean']
        # extraction of the parameters for each inner class
        self.inner_class_param = {
            inner: {key: value for key, value in modality_args.items() if key in self.param[inner]} for inner in
            ["sensor", "percept", "word", "lexical", "attention"]}
        self.deactivated_inferences = {"build": [], "update": []}

        self.enabled = True
        self.enabled = enabled
        self.model = model
        self.mod = mod
        self.other_mod = other_mod
        self._stim = stim
        self.n = None
        self.N = self.M = None

        # use a reduced alphabet to simplify
        self.chars = ""
        self.chars_filename = chars_filename
        self.set_char_dict()

        self.removed_words = None
        self.limited_TD = True  # only the first 50 representation to limit calculation cost

    @utl.abstractmethod
    def __contains__(self, item):
        return False

    @property
    def stim(self):
        return self._stim

    @stim.setter
    @utl.abstractmethod
    def stim(self, value):
        pass

    @utl.abstractmethod
    def get_pos(self):
        pass

    def __setattr__(self, name, value):
        """
        This function allows for setting attributes in the inner classes of class modality by checking if the attribute is in the corresponding class

        :param name: The name of the attribute being set
        :param value: The value that is being set for the attribute named 'name'.
        on the attribute being set
        """

        if 'param' in self.__dict__:
            for inner_class, params in self.param.items():
                if name in params and inner_class in self.__dict__:
                    getattr(self, inner_class).__setattr__(name, value)
                    return

        super().__setattr__(name, value)

    ############################
    ##### INIT #################
    ############################

    def set_char_dict(self):
        """
        Reads a csv file to create the orthographic or phonological alphabet
        """
        col = ['idx', 'char']  # if mod == "ortho" else ['idx', 'char', 'ipa']
        dict_path = os.path.realpath(os.path.dirname(__file__)) + '/../resources/chardicts/' + self.chars_filename
        df = pd.read_csv(dict_path, usecols=col)
        self.chars = "".join(df.char)  # it is simply a string
        # pdb.set_trace()
        self.n = len(self.chars)

    def reset_modality(self, reset):
        """
        Resets all model's modal attributes

        :param reset: dictionary that contains two keys: "lexicon" and "dist". The value associated with each key determines whether to reset the lexicon or the distributions
        """

        self.reset_inferences()

        if reset["lexicon"]:
            self.lexical.extract_lexicon()
            self.lexical.build_all_repr()
        self.lexical.set_repr()  # needed for reset_dist et build_context

        if reset["dist"]:
            self.percept.dist["percept"] = self.lexical.get_empty_percept()
            self.word.dist["ld"] = np.array([0.5, 0.5])
            self.word.dist["gamma"] = 0
            self.percept.dist["gamma"] = 0
            self.percept.dist["gamma_sem"] = 0
            self.percept.dist["gamma_sim"] = 0
            self.word.dist["word"] = utl.norm1D(self.lexical.freq[:self.lexical.shift_begin])
            self.word.dist["word_sim"] = utl.wsim(self.lexical.repr[:self.lexical.shift_begin],
                                                  self.percept.dist["percept"])
            self.word.dist["word_sim_att"] = np.column_stack((utl.norm1D(self.lexical.freq),np.zeros(utl.norm1D(self.lexical.freq).shape)))
            self.percept.used_idx, self.percept.used_idx_tmp, self.percept.used_mask = {key: [] for key in
                                                                                        range(30)}, [], []
        self.build_bottom_up()  # needs position to be set before

    @utl.abstractmethod
    def reset_inferences(self):
        pass

    ######################################
    ######## INFERENCES ##################
    ######################################

    ########## Bottom-up Inferences #####

    def build_bottom_up(self):
        """
        Builds the bottom-up matrix according to the interference matrix and the attention distribution
        """
        self.attention.build_attention_distribution()
        if 'sensor' not in self.deactivated_inferences ['build']:
            self.sensor.build_interference_matrix()
            if self.attention.mean >= 0 and self.sensor.dist[
                "interference"] is not None and self.attention.dist is not None and self.model.input_type=="visual" :
                logging.debug(f"application de l'attention {self.mod}")
                self.percept.bottom_up_matrix = [i * a + (1 - a) / self.n for (i, a) in
                                                 zip(self.sensor.dist["interference"], self.attention.dist)]
            else:
                self.percept.bottom_up_matrix = self.sensor.dist["interference"]
            if len(self.percept.bottom_up_matrix)<self.N_max:
                self.percept.bottom_up_matrix = np.concatenate((self.percept.bottom_up_matrix, np.ones((self.N_max-len(self.percept.bottom_up_matrix),self.n))*1/self.n))

    def build_modality(self):
        """
        Calculates bottom-up inferences
        """
        if self.enabled:
            if 'sensor' not in self.deactivated_inferences['build']: self.sensor.build_sensor()
            if 'percept' not in self.deactivated_inferences['build']: self.percept.build_percept()
            if 'similarity' not in self.deactivated_inferences['build']: self.word.build_similarity()
            if 'decoding' not in self.deactivated_inferences['build']: self.percept.build_decoding()
            if 'word' not in self.deactivated_inferences['build']: self.word.build_word()
            if 'ld' not in self.deactivated_inferences['build']: self.word.build_ld()
            # if 'word_sem' not in self.deactivated_inferences['build']: self.word.update_word_sem()

    ########## Top-Down Inferences #####

    def update_modality(self):
        """
        Calculates top-down inferences
        """
        if self.enabled:
            self.word.gamma()
            self.percept.gamma()
            # self.percept.gamma_sem()
            if 'word' not in self.deactivated_inferences['update']: self.word.update_word()
            if 'percept' not in self.deactivated_inferences['update']: self.percept.update_percept()
            # if 'percept_sem' not in self.deactivated_inferences['update']: self.percept.update_percept_sem()
            pass

    ######################################
    ######### RESULT #####################
    ######################################

    def print_all_dists(self):
        """
        Prints out information about all distributions (ld,percept,word).
        """
        if self.enabled:
            logging.simu(f"\n {self.mod.upper()} : mot {'non' if self.word.PM else ''} reconnu")
            logging.simu(self.percept.print_dist())
            logging.simu(self.word.print_dist('ld'))
            logging.simu(self.word.print_dist("word"))
            logging.simu(self.word.print_dist("word_sim"))

    @utl.abstractmethod
    def detect_context_error(self):
        """
        Detects a context error (word identified in the context, but not the stimulus).

        :return: a boolean value.
        """
        dec = self.word.decision("word")
        return (not self.word.PM or self.model.mismatch) and dec != self.stim and dec in self.semantic.context_sem_words and self.stim not in self.semantic.context_sem_words

    def get_dirac(self, string=None):
        """
        Returns the maximum value of the distribution on each letter/phoneme for a given stimulus.

        :param string: The input word. If it is None, the function calculates the distribution for the current model stimulus.
        :return: a list of maximum values of the distribution `P(Li|W=string)` if the word is known, `P(Pi)` if it's novel, for each letter position `i`.
        """
        if not self.enabled or self.word.PM or (string not in self):
            p = self.percept.dist["percept"]
            return [round(max(i), 3) for i in p] if p is not None else None
        else:
            idx = self.lexical.get_word_entry().idx if string is not None else self.decision("word_index")
            wd = self.lexical.repr[idx]
            return [round(max(i) / sum(i), 3) for i in wd]


class _Ortho(_Modality):
    def __init__(self, stim="partir", **modality_args):
        super().__init__(stim=stim, **modality_args)

        # submodels in the orthographic branch
        self.sensor = sens.sensorOrtho(self, **self.inner_class_param['sensor'])
        self.percept = per.perceptOrtho(self, **self.inner_class_param['percept'])
        self.attention = att.attentionOrtho(self, **self.inner_class_param['attention'])
        self.lexical = lxc._LexicalOrtho(self, **self.inner_class_param['lexical'])
        self.word = wrd._WordOrtho(self, **self.inner_class_param['word'])
        self.reset_inferences()

    def __contains__(self, item):
        """
        Return True if the item is in the lexicon and its orthography is known
        """
        if item not in self.lexical.df.index:
            return False
        ortho = self.lexical.df.loc[item].ortho
        return any(ortho) if isinstance(ortho, pd.Series) else ortho

    @_Modality.stim.setter
    def stim(self, value):
        """
        Setter for the variable 'stim' with all associated actions :
        - sets the stimulus name in both modalities
        - calculates forbid entries for the current stimulus
        - build bottom-up information
        - affects the new value for the phonological length (N)
        - extracts the graphemic segmentation when it is known
        """

        logging.debug("ortho stim setter")

        if value is not None:
            self._stim = value if isinstance(value, str) else ""
            # On affecte N et M dans les 2 modalités pour plus de simplicité
            if self.model.input_type=="visual":
                self.N = len(self._stim)
                if self.lexical.shift:
                    self.lexical.shift_begin = self.lexical.all_shift_begin[self.N]
                if self.model.remove_neighbors or self.model.remove_lemma:
                    self.forbid = self.model.ortho.get_forbid_entries()
                try:
                    self.build_bottom_up()
                except:
                    pass
                if 'phono' in self.model.__dict__ and self.model.phono.enabled and self.model.phono.lexical.df is not None:
                    self.model.phono.N = self.N
                    self.model.phono.M = self.model.phono.N_max
                    self.M = self.model.phono.M
                    try:
                        self.model.phono.stim = self.model.phono.lexical.get_phono_name()
                    except:
                        print("error in stim setting")
                        pdb.set_trace()
                    if self.model.phono.stim is not None:
                        try:
                            raw = self.model.df_graphemic_segmentation.loc[value]
                            # theoritical phonological position for each ortho position
                            self.model.gs = "".join([len(val) * str(i) for i, val in enumerate(raw['segm'].split('.'))])
                        except:
                            pass
                    self.model.phono.attention.calculate_attention_parameters()
                    self.model.phono.attention.build_attention_distribution()


    @property
    def pos(self):
        return self.attention.mean

    @pos.setter
    def pos(self, value):
        """
        Sets attention and eye position, starts at 0. Position should be set at -1 at the end of a simulation

        :param: value: int, the position to be set.
        """
        logging.debug(f"ortho position is trying to be set : {value} ({self.N})")
        if value <= -1 or (self.N is not None and value >= self.N):
            # pdb.set_trace()
            logging.warning(f"bad ortho position is trying to be set : {value} {self.N}")
        self.attention.mean = value
        self.sensor.gaze = value
        self.build_bottom_up()

    ############################
    ##### INIT #################
    ############################

    def set_char_dict(self):
        """
        Chooses the alphabet to choose for the simulations
        """
        if len(self.chars_filename) == 0:
            self.chars_filename = "alphabet_ge.csv" if self.model.langue == "ge" else "alphabet_en.csv" if self.model.langue == "en" else "alphabet_lat.csv"
        if self.model.langue == "fr":
            self.chars_filename = "alphabet_fr_simplified.csv"
        super().set_char_dict()

    def reset_inferences(self):
        """
        Sets inferences that are not used during calculation.
        """

        if self.model.input_type == "visual":
            self.deactivated_inferences = {"build": ["word_sem"], "update": ["percept_sem"]}
            # self.deactivated_inferences = {"build": ["word_sem","decoding"], "update": ["percept_sem","word","percept"]}
        if self.model.input_type == "auditory":
            self.deactivated_inferences = {"build": ["sensor", "percept", "word_sem"], "update": ["percept_sem"]}
            # self.deactivated_inferences = {"build": ["sensor", "percept", "word_sem"], "update": ["percept_sem","word","percept"]}

    def detect_context_error(self):
        """
        Detects a context error (word identified in the context, but not the stimulus).

        :return: a boolean value.
        """
        dec = self.word.decision("word")
        return (not self.word.PM or self.model.mismatch) and dec != self.stim and dec in self.model.semantic.context_sem_words_phono and self.stim not in self.model.semantic.context_sem_words

class _Phono(_Modality):
    def __init__(self, stim="paRtiR", placement_auto=True, **modality_args):

        super().__init__(stim=stim,**modality_args)
        self.model.set_lexicon_name(self.enabled)  # needs to know if phono is enabled to choose the lexicon name
        self.sensor = sens.sensorPhono(self, **self.inner_class_param['sensor'])
        self.percept = per.perceptPhono(self, **self.inner_class_param['percept'])
        self.attention = att.attentionPhono(self, **self.inner_class_param['attention'])
        self.lexical = lxc._LexicalPhono(self, **self.inner_class_param['lexical'])
        self.word = wrd._WordPhono(self, **self.inner_class_param['word'])
        self.reset_inferences()

    def __contains__(self, item):
        """
        If the item is in the lexicon, and the item has a phonological representation, then return True

        :param item: the word to be checked
        :return: The word entry for the word in the lexicon.
        """
        lx = self.lexical.df
        return item in lx.index.values and self.lexical.get_word_entry(item).store == True

    @_Modality.stim.setter
    def stim(self, value):
        """
        Setter for the variable 'stim' with all associated actions :
        - sets the stimulus name in both modalities
        - calculates forbid entries for the current stimulus
        - build bottom-up information
        - affects the new value for the phonological length (N)
        - extracts the graphemic segmentation when it is known
        """

        logging.debug("phono stim setter")

        if value is not None:
            self._stim = value if isinstance(value, str) else ""
            # On affecte N et M dans les 2 modalités pour plus de simplicité
            if self.model.input_type=="auditory":
                self.M = len(self._stim)
                if 'ortho' in self.model.__dict__ and self.model.ortho.enabled and self.model.ortho.lexical.df is not None:
                    try:
                        self.model.ortho.M = self.M
                        self.model.ortho.N = self.model.ortho.N_max
                        self.N = self.model.ortho.N
                        self.model.ortho.stim = self.model.ortho.lexical.get_ortho_name()
                        # if self.model.ortho.stim is not None:
                        #     try:
                        #         raw = self.model.df_graphemic_segmentation.loc[value]
                        #         # theoritical phonological position for each ortho position
                        #         self.model.gs = "".join([len(val) * str(i) for i, val in enumerate(raw['segm'].split('.'))])
                        #     except:
                        #         pass
                        self.model.ortho.attention.calculate_attention_parameters()
                        self.model.ortho.attention.build_attention_distribution()
                    except:
                        print("stim setting fail")
                        pdb.set_trace()
                        self.model.ortho.stim = ""
                        self.model.ortho.N = 0
                        self.model.ortho.M = self.M


    @property
    def pos(self):
        return self.attention.mean

    @pos.setter
    def pos(self, value):

        """
        Sets attention position, starts at 0. Position should be set at -1 at the end of a simulation

        :param value: int, the position to be set.
        """
        logging.debug(f"phono position is trying to be set : {value} ({self.M})")
        # pdb.set_trace()
        if value < -1 or (self.M is not None and value > self.M):
            logging.warning(f"bad phono position is trying to be set : {value} {self.M}")
            if (self.M is not None and value > self.M):
                self.pos = self.M
        self.attention.mean = value
        self.sensor.gaze = value
        self.build_bottom_up()

    ############################
    ##### INIT #################
    ############################

    def set_char_dict(self):
        """
        Automatically chooses the dictionary according to the language
        """
        if len(self.chars_filename) == 0:
            self.chars_filename = "xsampa_celex.csv" if self.model.langue in "en" else \
                "xsampa_fr_simplified.csv" if self.model.langue in "fr" else "xsampa_sp.csv" if self.model.langue == "sp" else "xsampa_celex_german.csv"
        super().set_char_dict()

    def reset_inferences(self):
        """
        Sets inferences that are not used during calculation.
        """

        if self.model.input_type == "visual":
            self.deactivated_inferences = {"build": ["sensor", "percept", "word_sem"], "update": ["percept_sem"]}
            # self.deactivated_inferences = {"build": ["sensor", "percept", "word_sem"], "update": ["percept_sem","word","percept"]}
        if self.model.input_type == "auditory":
            self.deactivated_inferences = {"build": ["word_sem"], "update": ["percept_sem"]}
            # self.deactivated_inferences = {"build": ["word_sem","decoding"], "update": ["percept_sem","word","percept"]}

    ######################################
    ######## INFERENCES ##################
    ######################################

    ######################################
    ######### RESULT #####################
    ######################################

    def print_all_dists(self):
        """
        Prints all probability distributions.
        """
        super().print_all_dists()
        if self.enabled:
            try:
                logging.simu(f"Psi score : {self.percept.psi_score()}")
            except:
                pdb.set_trace()

    def detect_context_error(self):
        """
        Detects a context error (word identified in the context, but not the stimulus).

        :return: a boolean value.
        """
        dec = self.word.decision("word")
        return (
                           not self.word.PM or self.model.mismatch) and dec != self.stim and dec in self.model.semantic.context_sem_words_phono and self.stim not in self.model.semantic.context_sem_words_phono
