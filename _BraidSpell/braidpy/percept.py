# BRAID utlities
import copy
import pdb
from time import time

import numpy as np
import braidpy.utilities as utl
import braidpy.lexicon as lex
from scipy.stats import entropy
import logging


class percept:
    """
    The percept class is an inner class of the modality class and represents the perceptual submodel, either in the orthographic or phonological modality.

    :param leak: float. Parameter for the decline of the percept distribution
    """

    def __init__(self, modality, leak, gamma_ratio_p, top_down, decoding_ratio_op, decoding_ratio_po):
        self.modality = modality
        self.leak = leak
        self.gamma_ratio_p = gamma_ratio_p
        self.top_down = top_down
        self.decoding_ratio_op = decoding_ratio_op
        self.decoding_ratio_po = decoding_ratio_po
        self.gamma_sem_init = 0
        self.limited_TD = True  # only the first 50 representation to limit calculation cost
        self.dist = {}
        self.max_HDer = 1000
        self.decision_thr = 0.25

        #################################

    #### INFERENCES #################
    #################################

    #### Calcul Gamma ##############

    def sigm(self, val):
        """
        Calculates the top-down lexical retroaction strength

        :param val: input value to the sigmoid function (generally the value of the lexical decision distribution)
        :return: the top-down lexical retroaction strength
        """
        # print("percept gamma ratio " + str(self.gamma_ratio))
        return 2e-6 + 1 * (self.gamma_ratio_p / np.power((1. + np.exp(-(97 * val) + 95)), .7))

    def gamma(self):
        """
        Updates the gamma coefficient for top-down lexical retroaction
        """
        self.dist["gamma"] = self.sigm(self.modality.word.dist["ld"][0])

    def init_gamma_sem(self):
        sem = self.modality.model.semantic
        if sem.top_down:
            pdb.set_trace()
            self.dist["gamma_sem"] = np.dot(self.modality.word.dist["word"], sem.dist["sem"])
            self.dist["sim_sem"] = 1
            self.gamma_sem_init = self.dist["gamma_sem"]

    def gamma_sem(self):
        """
        Calculates values for gamma_sem and gamma_sim based on the similarity between the semantic distribution and the word distribution.
        """
        if self.modality.model.semantic.top_down:
            sim = np.dot(self.modality.word.dist["word"],
                         self.modality.model.semantic.dist['sem']) / self.gamma_sem_init
            gamma = 2 * self.modality.word.gamma_ratio_p / np.power((1. + np.exp(-3 * sim + 11)), .7)
            self.dist["gamma_sem"] = gamma
            self.dist["sim_sem"] = sim

    ##### Calcul probability distributions ########

    def build_percept(self):
        """
        Builds the percept distribution with bottom-up information
        """
        mem = (self.dist["percept"] + self.leak) / (1 + self.modality.n * self.leak)
        self.dist["percept"] = utl.norm_percept(mem * self.bottom_up_matrix)

    def update_percept(self):
        """
        Updates the percept distribution with the top down retroaction
        """
        if self.modality.enabled and self.modality.percept.top_down:
            if self.limited_TD:
                idx = self.modality.word.dist['word'].argsort()[::-1][:50]
                dist = utl.TD_dist(self.modality.word.dist["word"][idx],
                                   self.modality.lexical.repr[:self.modality.lexical.shift_begin][idx])
            else:
                dist = utl.TD_dist(self.modality.word.dist["word"],
                                   self.modality.lexical.repr[:self.modality.lexical.shift_begin])
            self.dist["percept"] *= (
                    self.dist["gamma"] * dist + (1 / self.modality.n) * (1 - self.dist["gamma"]) * np.ones(
                self.dist["percept"].shape))
            self.dist["percept"] = utl.norm_percept(self.dist["percept"])

    def update_percept_sem(self):
        """
        Updates the percept distribution with the top down retroaction according to the semantic context
        """
        if self.modality.enabled and self.modality.model.semantic.top_down:
            dist = utl.TD_dist(self.modality.word.dist["word"],
                               self.modality.lexical.repr[:self.modality.lexical.shift_begin])
            self.dist["percept"] *= (
                    self.dist["gamma_sem"] * dist + (1 / self.modality.n) * (1 - self.dist["gamma_sem"]) * np.ones(
                self.dist["percept"].shape))
            self.dist["percept"] = utl.norm_percept(self.dist["percept"])

    def build_decoding(self):
        """
        Builds the percept distribution of the other modality.
        """
        self.decoding()
        return

    def most_sim_word_other_mod(self):
        """
        Returns the representations of the most similar lexical words of
        the other modality to its percept, and their alignment and index.
        """
        other_mod = self.modality.model.ortho if self.modality.mod == "phono" else self.modality.model.phono

        mask = utl.norm1D(other_mod.word.dist["word_sim_att"][:, 0])
        nb_words = min(25, len(mask))
        idx_sort = np.sort(np.argpartition(mask, len(mask) - nb_words)[-nb_words:])
        self.used_idx_tmp = [i for i in idx_sort if
                             mask[i] > 0]  # for the case where there are less than 10 words in the lexicon
        try:
            self.used_idx[other_mod.pos] = list(
                dict.fromkeys(self.used_idx[other_mod.pos] + self.used_idx_tmp))
        except:
            pdb.set_trace()
        self.used_mask = [mask[i] for i in self.used_idx_tmp]

        other_mod_words = other_mod.lexical.repr[self.used_idx_tmp]
        other_mod_align = other_mod.word.dist["word_sim_att"][self.used_idx_tmp, 1]

        return other_mod_words, other_mod_align

    def decoding(self):
        """
        Performs the decoding step of the other modality.
        """
        other_mod = self.modality.model.ortho if self.modality.mod == "phono" else self.modality.model.phono

        other_mod_words, other_mod_align = self.most_sim_word_other_mod()

        end_per_pos = len(other_mod.stim)

        # print(f"used words {other_mod.lexical.get_names(self.used_idx_tmp)}")
        # print(f"corresponding words {self.modality.lexical.get_names(self.used_idx_tmp)}")
        # print(f"align {other_mod_align}")

        try:
            tmp_repr = utl.merged_repr_decoding(self.used_idx_tmp, self.modality.lexical.repr, other_mod_align, other_mod_words, end_per_pos)
        except:
            pdb.set_trace()


        # logging.simu(f"used words : {other_mod.lexical.get_names(self.used_idx_tmp)}")
        # logging.simu(f"word_sim : {utl.norm1D(other_mod.word.dist['word_sim'][self.used_idx_tmp,0])}")
        # logging.simu(f"word_sim_att : {utl.norm1D(other_mod.word.dist['word_sim_att'][self.used_idx_tmp, 0])}")


        # 4. Merge them and modulate this new representation with the attention profile
        tmp_push = np.einsum('i,ijk->jk', utl.norm1D(self.used_mask), tmp_repr)
        # tmp_push = np.einsum('i,ijk->jk', [1 / len(self.used_mask)] * len(self.used_mask), tmp_repr)

        decoding_ratio = self.decoding_ratio_op if self.modality.mod == "phono" else self.decoding_ratio_po

        filt = np.array([i * a + (1 - a) / len(self.dist["percept"]) for (i, a) in
                      zip(tmp_push, self.modality.attention.dist * decoding_ratio)])

        mem = (self.dist["percept"] + self.leak) / (1 + self.modality.n * self.leak)

        # 5. Combine previous percept with new information ready to push

        try:
            self.dist["percept"] = utl.norm_percept(mem * filt)
        except:
            print("fail attribution of percept dist value")
            pdb.set_trace()

        if np.isnan(self.dist["percept"]).any():
            print("any value of percept is nan")
            pdb.set_trace()

        # print(f"decision : {self.decision(dist=tmp_push)}")
        # print(f"new percept : {self.decision(dist=self.dist['percept'])}")

    #################################
    #### INFO #######################
    #################################

    def get_entropy(self, dist=None):
        """
        Calculates the entropy of the percept distribution

        :param dist: a distribution in the same shape as self.dist["percept"]. If None, returns the percept entropy.
        :return: The entropy of the distribution, one value per position.
        """
        # print(f"Entropy of stim : {self.modality.stim}")
        if (self.modality.mod == "phono" and self.modality.model.input_type == "visual") or (
                self.modality.mod == "ortho" and self.modality.model.input_type == "auditory"):
            until_end_detection = True
        else:
            until_end_detection = False
        t_dist = dist if dist is not None else self.dist["percept"]
        if not until_end_detection:
            return [entropy(i) for i in t_dist[0:len(self.modality.stim)]]
        else:
            pos_end = np.where(t_dist[:, 35] > 0.5)
            if len(pos_end[0]) > 1:
                return [entropy(i) for i in t_dist[0:pos_end[0][1]]]
            else:
                return [entropy(i) for i in t_dist[0:len(self.dist["percept"])]]

    #################################
    #### DECISIONS ##################
    #################################

    # @utl.abstractmethod
    def decision(self, dist=None):
        """
        Returns a string representing the pronunciation based on the maximum values of the phonological percept distribution.

        :param dist: 2D numpy array. The distribution of probabilities for each phoneme in a given word.
        :return: a string that represents the pronunciation of the word.
        """
        if isinstance(dist, str):
            raise TypeError("distribution should be an array of float values")
        try:
            dist = dist if dist is not None else self.dist["percept"]
            maxi = [max(d) for d in dist][::-1]
            last_idx = len(dist) - 1 - (next(i for i, val in enumerate(maxi) if val > self.decision_thr) if max(maxi) > self.decision_thr else 0)
            pron_idx = [np.argmax(dist[i, :]) if max(dist[i, :]) > self.decision_thr and list(dist[i, :]).count(dist[i, 0]) != len(
                dist[i, :]) and (
                                                         i <= last_idx or not (
                                                         self.modality.enabled and self.modality.lexical.repr_unif)) else -1
                        for i in range(np.shape(dist)[0])]
        except:
            pass

        return "".join([self.modality.chars[i] if i > -1 else '~' for i in pron_idx])

    def evaluate_decision(self, dist=None):
        """
        Evaluates the percept decision taken by the model (function decision of this class).

        :param dist: a distribution in the same shape as self. If None, the model state is used instead.
        :return: a boolean value indicating whether the decision is correct.
        """
        dist = dist if dist else self.dist["percept"]
        return self.modality.stim == utl.str_transfo(self.decision(dist))

    def print_dist(self):
        """
        Prints information about the percept distribution (used at the end of a simulation)

        :param dist_name: The name of the distribution to be printed.
        :return: a string with information about the distribution.
        """
        if self.modality.enabled:
            return f'percept {self.modality.mod}, {self.decision()}, {[round(np.max(i), 4) for i in self.dist["percept"]]}'
        return ''

    def get_used_words(self):
        """
        A dictionary of the words used for decoding when attention landed on a specific position.

        :return: A dictionary of the used words in the format { position : [used_words] }
        """
        if self.modality.mod == "ortho":
            return {key: self.modality.model.phono.lexical.get_names(
                list(dict.fromkeys(self.used_idx[key] + self.used_idx_tmp))) for key, value in self.used_idx.items() if
                len(value) > 0}
        else:
            return {key: self.modality.model.ortho.lexical.get_names(
                list(dict.fromkeys(self.used_idx[key] + self.used_idx_tmp))) for key, value in self.used_idx.items() if
                len(value) > 0}

    def get_dz_pos(self, threshold=0.5):
        if 'percept' not in self.modality.percept.dist:
            return self.modality.M
        decision = self.modality.percept.decision()
        count=0
        for pos in range(len(decision)):
            if decision[pos] == '!' and max(self.modality.percept.dist["percept"][pos]) > threshold and count !=2:
                dz_pos = pos
                count+=1
        if count <2:
            dz_pos = len(decision)
        return dz_pos


class perceptOrtho(percept):
    def __init__(self, modality, top_down=True, gamma_ratio_p=1e-4, leak=1e-5,
                 decoding_ratio_op=0.005, decoding_ratio_po=0.001, **modality_args):
        self.used_idx, self.used_idx_tmp, self.used_mask = {key: [] for key in range(30)}, [], []
        # if modality.model.input_type == "visual":
        #     decoding_ratio_op = 0.0035
        #     decoding_ratio_po = 0.00175
        # else:
        #     decoding_ratio_op = 0.001
        #     decoding_ratio_po = 0.005

        if modality.model.input_type == "visual":
            decoding_ratio_op = decoding_ratio_op
            decoding_ratio_po = decoding_ratio_po
        else:
            decoding_ratio_op = decoding_ratio_op
            decoding_ratio_po = decoding_ratio_po

        super().__init__(modality=modality, top_down=top_down, gamma_ratio_p=gamma_ratio_p, leak=leak,
                         decoding_ratio_op=decoding_ratio_op, decoding_ratio_po=decoding_ratio_po, **modality_args)


class perceptPhono(percept):
    def __init__(self, modality, top_down=True, gamma_ratio_p=1e-4, leak=5e-6, placement_auto=True,
                 decoding_ratio_op=0.005, decoding_ratio_po=0.001, **modality_args):
        # if modality.model.input_type == "visual":
        #     decoding_ratio_op = 0.0035
        #     decoding_ratio_po = 0.00175
        # else:
        #     decoding_ratio_op = 0.00175
        #     decoding_ratio_po = 0.005

        if modality.model.input_type == "visual":
            decoding_ratio_op = decoding_ratio_op
            decoding_ratio_po = decoding_ratio_po
        else:
            decoding_ratio_op = decoding_ratio_op
            decoding_ratio_po = decoding_ratio_po

        # print(gamma_ratio)

        super().__init__(modality=modality, top_down=top_down, gamma_ratio_p=gamma_ratio_p, leak=leak,
                         decoding_ratio_op=decoding_ratio_op, decoding_ratio_po=decoding_ratio_po, **modality_args)
        self.use_word = False
        self._pos = -1
        self.used_idx, self.used_idx_tmp, self.used_mask = {key: [] for key in range(30)}, [], []
        # better results with True, would like to be able to put it at False
        self.placement_auto = placement_auto

    #################################
    #### INFERENCES #################
    #################################

    def filter_att(self, dist=None, att=None):
        """
        Calculates the perceptual distribution filtered by attention.

        :param dist: the percept distribution. If None, takes the model current percept distribution.
        :param att: the attention profile, which is a list of floats between 0 and 1. If None, takes the model current attentional distribution.
        :return: The filtered percept distribution
        """
        att = att if att is not None else self.modality.attention.dist
        dist = dist if dist is not None else self.dist["percept"]
        return np.array([i * a + (1 - a) / self.modality.n for (i, a) in zip(dist, att)])



    #################################
    #### INFO #######################
    #################################

    def psi_score(self, stim=None):
        """
        Computes the cosine similarity between the representation of the stimulus and the representation of the phonological percept distribution.

        :return: The mean of the cosine similarity between the representation of the stimulus and the representation of the percept.
        """
        if stim is None:
            stim = self.modality.stim

        if len(stim) > 0:
            n = len(utl.str_transfo(stim))
            p = self.dist["percept"][:n]
            # on compare pas avec la représentation stockée mais avec une représentation adulte
            # comme ça la référence est toujours la même
            wds_idx = self.modality.lexical.get_repr_indices([stim])
            repr = utl.create_repr(wds_idx, self.modality.n, self.modality.lexical.eps)[0]
            scal = np.einsum('ij,ij->i', p, repr)
            norm = np.sqrt(np.einsum('ij,ij->i', repr, repr)) * np.sqrt(np.einsum('ij,ij->i', p, p))
            return np.mean(scal / norm) if sum(norm) > 0 else 0
        return -1
