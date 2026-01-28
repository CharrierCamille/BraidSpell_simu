# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:41:03 2020

@author: Alexandra, Ali
"""

# General purpose libraries
import copy
import unicodedata

import logging
import numpy as np
import os
# Scientific/Numerical computing
import pandas as pd
## debugging
import pdb
import sys

import braidpy.modality as mod
import braidpy.semantic as sem
# BRAID utlities
import braidpy.utilities as utl
import braidpy.lexicon as lex


class braid:
    """ instantaneous BRAID model : Inner class of the Simu class"""

    def __init__(self, ortho_param, phono_param, semantic_param, path='../', langue="fr", lexicon_name="",
                 recalage=True, input_type="visual"):
        """
        :param ortho_param: dict, parameters for the inner class
        :param phono_param: dict, parameters for the inner class
        :param langue: string, among "en" ,"fr" or "ge"
        :param lexicon_name: string, lexicon filename (abcdef.csv)
        :param recalage: boolean. if True, "recalage" performed at the end of the simulation
        :param input_type: string, among "visual" or "auditory". "both" will be implemented later.
        """
        if ortho_param is None:
            ortho_param = {}
        if phono_param is None:
            phono_param = {}
        if semantic_param is None:
            semantic_param = {}
        self.ortho_param_names = ['Q', 'stim', 'eps', 'top_down', 'leak', 'crowding', 'scaleI', 'slopeG', 'segm',
                                  'gamma_ratio_p', 'gamma_ratio_w' ,'att_factor', 'markov_sim', 'mean', 'gaze',
                                  'force_app', 'force_update', 'force_word', 'ld_thr', 'ld_weight', 'decoding_ratio_po',
                                  'lenMin', 'lenMax', 'phlenMin', 'phlenMax']
        self.phono_param_names = ['Q', 'leak', 'use_word', 'placement_auto', 'ld_thr', 'ld_weight', 'decoding_ratio_op',
                                  'gamma_ratio_p', 'gamma_ratio_w', 'lenMin', 'lenMax', 'phlenMin', 'phlenMax']
        self.path, self.langue, self.lexicon_name = path, langue, lexicon_name
        self.init_model_args()
        self.input_type = input_type

        # current length, corresponding phoneme length (max), theoritical max length of stimulus
        self.df_lemma = pd.read_csv(self.path + 'resources/lexicon/Lexique_lemma.csv')[['word', 'lemme']].set_index(
            'word') if self.remove_lemma else None
        self.df_graphemic_segmentation = pd.read_csv(
            self.path + 'resources/lexicon/graphemic_segmentation.csv').groupby('word').first()
        self.stim_graphemic_segmentation = {}
        self.dist_names = ["percept", "word", "ld", "gamma", "gamma_sem", "gamma_sim", "TD_dist", "TD_dist_sem",
                           "word_sim_att"]
        self.phono = mod._Phono(mod="phono", other_mod="ortho", model=self, dist={key: None for key in self.dist_names},
                                **phono_param)
        self.ortho = mod._Ortho(mod="ortho", other_mod="phono", model=self, dist={key: None for key in self.dist_names},
                                **ortho_param)
        self.semantic = sem.semantic(model=self, **semantic_param)
        if self.phono.enabled and self.ortho.enabled:
            self.phono.attention.set_regression()
            self.ortho.attention.set_regression()

        if self.input_type == "visual":
            self.ortho.stim = ortho_param['stim'] if 'stim' in ortho_param else self.ortho.stim
        elif self.input_type == "auditory":
            self.phono.stim = phono_param['stim'] if 'stim' in phono_param else self.phono.stim
        else :
            raise Exception("Invalid input_type value. Should be either 'visual' or 'auditory'.")

         # on le définit que maintenant pour pouvoir avoir accès au stim phono correspondant
        self.recalage = recalage

        self.mismatch = False
        self.PM = None
        self.chosen_modality = None

    def init_model_args(self):
        """
        This function initializes various attributes to their default values for the model.
        """
        self.old_freq = None
        self.PM = False
        self.chosen_modality = None
        self.mismatch = False
        self.mismatch = False

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value):
        new = 'shift' in self.__dict__ and self._shift != value
        self._shift = value
        if new and 'ortho' in self.__dict__:
            self.start()

    def enable_phono(self, value):
        """
        This function enables or disables phonological processing and updates the lexical knowledge accordingly.

        :param value: boolean value that determines whether the "phono" feature is enabled or disabled.
        """
        self.phono.enabled = value
        if value and self.df is not None:
            self.phono.lexical.build_all_repr()
            if self.ortho.N is not None:
                self.phono.set_repr()
        else:
            self.phono.all_repr = None
            self.phono.repr = None

    def __getattr__(self, name):
        """
        Getter that also checks if the attribute is in the ortho/phono class and returns it if it is.

        :param name: variable that contains the name of the attribute that is being accessed.
        :return: If the attribute name is found in the model class, it's returned. If it's in `ortho_param_names` or `phono_param_names`, the corresponding attribute value from the `ortho` or
        `phono` object is returned using the `__getattribute__` method. Otherwise, an `AttributeError` will be raised.
        """
        if 'ortho_param_names' in self.__dict__ and name in self.ortho_param_names:
            return self.ortho.__getattribute__(name)
        elif 'phono_param_names' in self.__dict__ and name in self.phono_param_names:
            return self.phono.__getattribute__(name)

    def __getstate__(self):
        # necessary to use pkl because getattr is overwritten
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __setattr__(self, name, value):
        """
        This function allows for setting attributes in the model/ortho/phono classes by checking if the attribute is in the ortho or phono class.

        :param name: The name of the attribute being set
        :param value: The value that is being set for the attribute named 'name'.
        on the attribute being set
        """


        if 'ortho_param_names' in self.__dict__ and name in self.ortho_param_names:
            self.ortho.__setattr__(name, value)
        elif 'phono_param_names' in self.__dict__ and name in self.phono_param_names:
            self.phono.__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def op(self):
        """
        Returns the list of enabled modalities.

        :return: the list that contains string modalities among 'ortho', 'phono'.
        """
        return (["ortho"] if self.phono.enabled else []) + (["phono"] if self.phono.enabled else [])

    ################## INIT METHODS #################################

    def set_lexicon_name(self, phono_enabled):
        """
        Automatically sets the lexicon name

        :param phono_enabled: boolean.
        """
        # c'est amodal donc ça reste dans braid
        if len(self.langue) == 0:
            self.langue = "en"
        if len(self.lexicon_name) == 0:
            if self.langue == "en":
                self.lexicon_name = "celex.csv" if phono_enabled else "ELP.csv"
            elif self.langue == "ge":
                self.lexicon_name = "celex_german.csv"
            elif self.langue == "sp":
                self.lexicon_name = "lexique_espagnol.csv"
            else:
                # self.lexicon_name = "lexique_fr.csv" if phono_enabled else "FLP.csv"
                # self.lexicon_name = "lexique3_fr.csv" if phono_enabled else "FLP.csv"
                # self.lexicon_name = "lexiconLiliPaco_v3.csv" if phono_enabled else "FLP.csv"
                self.lexicon_name = "Adapted_lexique_infra_nouns.csv" if phono_enabled else "FLP.csv"
                # self.lexicon_name = "LiliPaco.csv" if phono_enabled else "FLP.csv"
                # self.lexicon_name = "lexicon_consistency.csv" if phono_enabled else "FLP.csv"

    ############ INIT TOP DOWN #################


    ################# INIT MODEL CONFIGURATION FOR SIMU #########################

    def reset_model(self, reset):
        """
        init top down and bottom up informations in each modality.
        """
        logging.debug("reset model")
        self.semantic.build_context()  # fait avant pour que l'init de sem se passe bien
        self.ortho.reset_modality(reset)
        self.phono.reset_modality(reset)

    def change_freq(self, newF=1, string=""):
        """
        Artificially changes the frequency of a word (for the freq effect simulation)
        """
        string = string if len(string) > 0 else self.stim
        self.ortho.lexical.change_freq(newF, string)
        phono_string = self.phono.lexical.get_phono_name(string)
        self.phono.lexical.change_freq(newF, phono_string)

    ########### MAIN #########################

    def one_iteration(self):
        """ One iteration of the simulation """
        self.ortho.build_modality()
        self.phono.build_modality()
        self.ortho.update_modality()
        self.phono.update_modality()

    ###################### UPDATE AFTER SIMU #######################

    def recalage_stim(self):
        """
        This function attempts to increase the similarity between the phoneme percept and the phonological lexicon by deleting or adding a new phoneme.
        To do this it compares the similarity between the percept and the lexicon to the same similarity with a modified percept with insertion or deletion.
        The insertion/deletion is kept only if its similarity with the lexicon is greater than the original similarity.
        """
        if not self.phono.decision("ld") and len(
                self.phono.repr) > 0 and self.shift_begin > 0:  # insertions/deletions needed ?
            for _ in range(2):  # 2 successive attempts to allow for 2 insertions/deletions
                # we delete/insert a character and see if it improves the comparison
                psi = self.phono.dist["percept"] # probability distribution
                pron_str = self.phono.percept.decision() # string
                n=next(i for i in reversed(range(len(pron_str))) if pron_str[i] != '#')+1 # nb of phonemes in the string
                n_maxi, res, maxi = -1, None, 1000
                n_ph = self.phono.n;
                unif = np.ones(n_ph) / n_ph  # uniform distribution
                # comparison between 2 similarities : percept/lexicon and modified percept/lexicon
                # because modified percept is built by making a deletion (for example) and put an uniform at the end, we have to compensate
                # the loss of 'informativeness' of this new percept without insertion/deletion by changing it and putting an uniform too
                # so the reference for the comparison will not be the percept itself, but a modified percept too.
                # first 4 arrays (indices from 0 to 3) are the insertion/deletion representations, the 4 last are the references
                for exch in range(n):  # calculation of the insertions/deletions and their references at each position
                    cmp = np.zeros((8, self.phono.M, self.phono.n))
                    # deletion
                    cmp[0] = np.concatenate((np.delete(psi, exch, axis=0), unif[np.newaxis]))
                    cmp[4] = copy.copy(psi);
                    cmp[4, exch] = unif
                    # insertion
                    cmp[2] = np.concatenate((psi[:exch], unif[np.newaxis, :], psi[exch:-1]))
                    cmp[6] = np.concatenate((psi[:-1], unif[np.newaxis]))
                    if exch < n - 1:  # double insertions or double deletion
                        cmp[1] = np.concatenate(
                            (np.delete(np.delete(psi, exch, axis=0), exch, axis=0), unif[np.newaxis], unif[np.newaxis]))
                        cmp[5] = copy.copy(psi);
                        cmp[5, exch] = unif;
                        cmp[5, exch + 1] = unif
                        cmp[3] = np.concatenate((psi[:exch], unif[np.newaxis, :], unif[np.newaxis, :], psi[exch:-2]))
                        cmp[7] = np.concatenate((psi[:-2], unif[np.newaxis], unif[np.newaxis]))
                    # similarity calculation
                    sim_word = np.prod(np.einsum('lij,kij->lki', cmp, self.phono.repr[:self.shift_begin]), axis=2)
                    sim = np.max(sim_word, axis=1)
                    for i in range(4):
                        if sim[i + 4] > 0 and sim[i] / sim[i + 4] > maxi:
                            i_res, maxi, res, typ = np.argmax(sim_word[i]), sim[i] / sim[i + 4], cmp[
                                i], "add" if i > 1 else "del"
                if res is not None:  # intertion/deletion selected
                    self.phono.dist["percept"] = res
                    logging.simu("recalage : "+pron_str +" -> "+self.phono.percept.decision()+ " coeff = "+str(maxi))
                    logging.simu("for word "+self.ortho.get_name(i_res))


    def detect_mismatch(self):
        """
        Detects a mismatch between the orthographic percept and the orthographic word corresponding to the phonological word identified
        """
        idx = self.phono.word.decision("word_index")
        dec_repr = self.ortho.lexical.repr[idx]
        p = self.ortho.percept.dist["percept"]
        if sum(sum(dec_repr)) > 0:
            sim = np.prod(np.einsum('ij,ij->i', dec_repr, p))
            sim_repr = np.prod(np.einsum('ij,ij->i', dec_repr, dec_repr))
            sim_p = np.prod(np.einsum('ij,ij->i', p, p))
            return sim / (sim_repr * sim_p) < 0.5

    def PM_decision_global(self):
        """
        Decisions at the end of the simulation :
        lexical membership evaluation according to the evaluation in the 2 modalities + modality choice + most probable word (in the chosen modality)
        """
        if self.phono.enabled:
            self.PM = True if self.ortho.word.PM and self.phono.word.PM else False
            if self.ortho.word.PM and not self.phono.word.PM:
                # if the word is phonologically known, verification that the letter percept is not incoherent with an eventual existing ortho representation
                mismatch = self.detect_mismatch()
                if mismatch:
                    logging.simu(f'/!\ Lexicalisation probable')
                    self.mismatch = self.ortho.word.PM = self.phono.word.PM = self.PM = True
                    self.chosen_modality = None
                else:
                    self.mismatch = False
            if not self.ortho.word.PM and not self.phono.word.PM:
                # chosen modality according to the maximum of the word distribution
                self.chosen_modality = "phono" if max(self.phono.word.dist["word"]) > max(
                    self.ortho.word.dist["word"]) else "ortho"
            elif not self.ortho.word.PM or not self.phono.word.PM:
                # if identification in one modality, checks if there is an existing lexical trace in the other modality (even if word not recognized)
                self.chosen_modality = "phono" if self.ortho.word.PM else "ortho"
                data = getattr(self, self.chosen_modality)
                other_data = getattr(self, data.other_mod)
                idx = data.word.decision("word_index")
                if sum(sum(other_data.lexical.repr[idx])) > 0:
                    other_data.word.PM = False
            else:
                self.chosen_modality = "phono"
        else:
            self.PM = self.ortho.word.PM
            self.chosen_modality = "ortho"
        idx = getattr(self, self.chosen_modality).word.decision(
            "word_index") if self.chosen_modality is not None else -1
        self.ortho.word.chosen_word = self.ortho.lexical.get_name(
            idx) if idx >= 0 else ""  # identification in the chosen modality
        self.phono.word.chosen_word = self.phono.lexical.get_name(
            idx) if idx >= 0 else ""  # identification in the chosen modality
