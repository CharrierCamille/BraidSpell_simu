import pdb

import numpy as np
import pandas as pd
from numpy import linalg as LA
import braidpy.utilities as utl
import matplotlib.pyplot as plt
from scipy.stats import entropy
import logging


class _Word:
    """
    The _Word class is an inner class of the modality class and represents the word perceptual submodel, either in the orthographic or phonological modality.

        :param leakW : int. Calibrated parameter for decline in the word distribution
        :param L2_L_division : boolean, division by L2 norm of L distribution ?
        :param ld_thr : float. Threshold for lexical novelty detection.
        :param word_reading : boolean. if True, phono DL is considered equal to 1 for the lexical feedback
    """

    def __init__(self, modality, gamma_ratio_w, top_down, leakW=1250, L2_L_division=False, ld_thr=0.9,
                 word_reading=False, new_ld=False, ld_weight=0.15):
        self.modality = modality
        self.gamma_ratio_w = gamma_ratio_w
        self.top_down = top_down
        self.leakW = leakW
        self.L2_L_division = L2_L_division
        self.weightLDYES = self.weightLDNO = ld_weight
        self.ld_trans_mat = np.array([[1 - self.weightLDYES, self.weightLDYES], [self.weightLDNO, 1 - self.weightLDNO]])
        self.ld_thr = ld_thr  # more corrections than needed, but misses almost no case where correction is needed
        self.word_reading = word_reading
        self.new_ld = new_ld
        self.PM = True
        self.chosen_word = ""
        self.dist = {}
        self.distsim = None
        self.distsimatt = None

    #################################
    #### INFERENCES #################
    #################################

    ############## Bottom-up #######

    def build_word_inferences(self):
        """
        Builds bottom-up distributions by building the similarity, the word and the ld distributions
        """
        if self.modality.enabled:
            self.build_similarity()
            self.build_word()
            self.build_ld()

    # @utl.abstractmethod
    def build_similarity(self):
        """
        Updates the similarity between the percept and the lexicon
        """
        logging.debug(f"build word sim {self.modality.mod}")
        # print(f"build word sim--------------------- {self.modality.mod}")

        att = self.modality.attention.Q * self.modality.attention.dist
        att[att > 1] = 1
        # no markov chain on wsim_mask, otherwise the prior will reinforce himself

        # print(f"lexical repr size: {self.modality.lexical.repr.shape}")
        tdistatt = utl.word_similarity_aligned(
            self.modality.lexical.repr[self.modality.lexical.sample_idx], self.modality.percept.dist["percept"], att)

        tdist_noatt = utl.word_similarity_aligned(
            self.modality.lexical.repr[self.modality.lexical.sample_idx], self.modality.percept.dist["percept"])

        previous_distsim = self.distsim if self.distsim is not None else np.full((len(self.modality.lexical.repr), len(self.modality.percept.dist["percept"]) + 2), 1e-100)

        if len(self.modality.lexical.sample_idx) != len(self.modality.lexical.repr):
            self.distsim = utl.sampled_dist_sim(previous_distsim, tdistatt, self.modality.lexical.sample_idx,
                                                   len(self.modality.percept.dist["percept"]))

            self.distsimatt = utl.sampled_dist_sim(previous_distsim, tdistatt, self.modality.lexical.sample_idx,
                                                len(self.modality.percept.dist["percept"]))

            self.distsim = utl.sampled_dist_sim(previous_distsim, tdist_noatt, self.modality.lexical.sample_idx,
                                                   len(self.modality.percept.dist["percept"]))
            # print("sample")
        else:
            self.distsim = tdistatt

            self.distsimatt = tdistatt
            self.distsim = tdist_noatt



        # pdb.set_trace()

        self.dist["word_sim_att"] = np.zeros((len(self.distsimatt), 2))
        self.dist["word_sim_att"][:, 0] = utl.norm1D(self.distsimatt[:, 0])
        self.dist["word_sim_att"][:, 1] = self.distsimatt[:, 1]

        # self.dist["word_sim_att"] = np.zeros((len(self.distsim), 2))
        # self.dist["word_sim_att"][:, 0] = utl.norm1D(self.distsim[:, 0])
        # self.dist["word_sim_att"][:, 1] = self.distsim[:, 1]


        try:
            wtrans = utl.build_word_transition_vector(self.dist["word"],
                                                      self.modality.lexical.freq,
                                                      self.leakW)
        except:
            pdb.set_trace()

        # wsim = utl.word_similarity_aligned(self.modality.lexical.repr, self.modality.percept.dist["percept"])
        wsim_del = np.delete(self.distsim, 1, 1)
        self.dist["word_sim"] = np.array(wtrans)[:, np.newaxis] * wsim_del

        # diff = np.abs(self.dist["word_sim"][:, 0] - self.dist["word"])
        # print(f"max diff: {np.max(diff)}")



    def build_word(self):
        """
        Builds the word distribution according to the similarity
        """
        logging.debug(f"build word {self.modality.mod}")

        # print(f"build word---------------------{self.modality.mod}")

        self.dist["word"] = utl.norm1D(self.dist["word_sim"][:, 0])

        if self.modality.model.L2_L_division:
            # necessary to limit the jump at the beginning of the simulation: the division by the norm is only for
            # the DL, not for W since the representations are stored in a noramalized way, it's necessary to
            # 'unnormalize' by multiplicating by the norm
            self.dist["word"] = utl.norm1D(self.modality.lexical.repr_norm * self.dist["word"])

        # print(f"word  ---  {self.dist['word']}")

    def build_ld(self):
        """
        Builds the lexical decision distribution according to the similarity
        """
        logging.debug(f"build ld {self.modality.mod}")
        if self.new_ld:
            pba_yes = self.dist["sim_n"]
            pba_no = self.dist["err_n"]
            self.ld_trans_mat = np.array([[1 - 0.15, 0.15], [0.15, 1 - 0.15]])
            proba_ld_new = utl.norm1D([pba_yes, pba_no])
            ld_proba_trans_new = np.matmul(self.ld_trans_mat, self.dist["ld"]) * proba_ld_new  # Markov transition
            self.dist["ld"] = utl.norm1D(ld_proba_trans_new)
            print([pba_yes,pba_no],self.dist["ld"])
        else:
            proba_error_i = self.dist["word_sim"].sum(0)
            # equivalent calculation, but the kept version is faster.
            # proba_ok=[1]+[LA.norm(i)/LA.norm(1-i) for i in self.percept.dis["percept"]t]
            proba_ok = [1] + [1 / np.sqrt(1 + (self.modality.n - 2) / LA.norm(i) ** 2) for i in
                              self.modality.percept.dist["percept"]]
            proba_err = [i * j for i, j in zip(proba_ok, proba_error_i)]
            # TODO enlever print(self,utl.l_round(proba_error_i,4))
            # print(self,utl.l_round(proba_ok,4))
            proba_ld = np.array([proba_err[0], np.mean(proba_err[1:])])
            ld_proba_trans = np.matmul(self.ld_trans_mat, self.dist["ld"]) * proba_ld  # Markov transition
            self.dist["ld"] = utl.norm1D(ld_proba_trans)

    ############## Top-Down #######

    def plot_sigm(self):
        """
        Plots a sigmoid curve to test different gamma functions
        """
        x = [i / 100 for i in range(120)]
        y = [self.sigm(i) / 10 for i in x]
        plt.plot(x, y)
        plt.show()

    def gamma(self):
        """
        Updates the gamma coefficient for top-down lexical retroaction
        """
        self.dist["gamma"] = self.sigm(self.dist["ld"][0])

        pass

    def sigm(self, val):
        """
        Calculates the top-down lexical retroaction strength

        :param val: input value to the sigmoid function (generally the value of the lexical decision distribution)
        :return: the top-down lexical retroaction strength
        """
        # print("word gamma ratio " + str(self.gamma_ratio))
        return 2e-6 + 1 * (self.gamma_ratio_w / np.power((1. + np.exp(-(97 * val) + 95)), .3))

    def update_word(self, other_dist):
        """
        Updates the word distribution according to the word distribution in the other modality, modulated by the ld distribution
        """
        if self.modality.enabled:
            gamma = self.sigm(other_dist["ld"][0])
            TD_dist = other_dist["word"] * gamma + np.ones(len(other_dist["word"])) / len(other_dist["word"]) * (
                    1 - gamma)
            self.dist["word"] = utl.norm1D(TD_dist * self.dist["word"])

    def update_word_sem(self):
        """
        Updates the word distribution with the word-semantic distribution : not depending on a dl value
        """
        print("update word sem")

        self.dist["word"] = utl.norm1D(self.modality.model.semantic.dist["sem"] * self.dist["word"])

    #################################
    #### INFO #######################
    #################################

    def get_entropy(self):
        """
        Calculates the entropy of the word distribution

        :return: The entropy of the word distribution.
        """
        return entropy(self.dist["word"])

    #################################
    #### DECISIONS ##################
    #################################

    def PM_decision(self):
        """
        Makes the lexical novelty detection for this modality
        """
        self.PM = False if self.modality.enabled and not self.modality.lexical.force_app and \
                           (self.modality.lexical.force_update or self.dist["ld"][0] > self.ld_thr or self.decision(
                               dist_name="word") in self.modality.model.semantic.context_sem_words \
                            and self.modality.model.semantic.p_sem > 1 and self.modality.model.semantic.context_identification and not self.modality.model.detect_mismatch()) else True

    def decision(self, dist_name="word", dist=None, **kwargs):
        """
        The function takes in a distribution name or a distribution and returns the decision based on it.

        :param dist_name: the name of the distribution to be used for the decision, defaults to word (optional)
        :param dist: the probability distribution to use for the decision. If it's not set, the model state is used instead.
        :return: The decision is being returned.
        """
        dist = dist if dist is not None else self.dist[dist_name] if dist_name in ['word', 'ld', 'word_sim',
                                                                                   'word_sim_att'] else \
            self.dist["word"] if dist_name == "word_index" else None
        if self.modality.enabled:
            if dist_name == 'ld':
                return dist[0] > (self.ld_thr if "ld_thr" not in kwargs else kwargs["ld_thr"])
            elif dist_name in ['word', 'word_sim_att', 'word_sim']:
                dsort = np.argsort(dist)[::-1][:1]
                try:
                    return np.array(self.modality.lexical.get_names(dsort))[dsort.argsort()][0]
                except:
                    return ""
            elif dist_name == "word_index":
                return np.argmax(dist) if len(dist) > 0 else -1

    def print_dist(self, dist_name="ld"):
        """
        Prints information about a given distribution (used at the end of a simulation)

        :param dist_name: The name of the distribution to be printed.
        :return: a string with information about the distribution.
        """
        if self.modality.enabled:
            if dist_name == 'word':
                dist = self.dist["word"]
                idx = self.decision("word_index")
                if idx > -1:
                    wd = self.modality.lexical.get_word_entry(self.modality.lexical.get_name(index=idx))
                    return f' {dist_name} {self.modality.mod} {wd.name}, idx = {idx}, wmax = {round(dist[idx], 6)}, {len(dist)} words '
            elif dist_name == 'ld':
                return f'{dist_name} {self.modality.mod} {self.dist["ld"][0]}'
            elif dist_name == 'word_sim':
                return f' : {dist_name} {self.modality.mod} {self.decision(dist_name)}'
        return ''

    def evaluate_decision(self, dist=None):
        """
        Evaluates the decision taken by the model (function decision of this class), for the distribution "word" only.

        :param dist: the probability distribution to use for the decision. If it's not set, the model state is used instead.
        :return: a boolean value indicating whether the decision is correct.
        """
        # not every decision can be evaluated by the model. For example, for the dl, one must know if it's the first exposure to a word or not
        # -> evaluation not implemented here
        dist = dist if dist else self.dist["word"]
        idx = self.decision("word_index", dist)
        return self.modality.stim == self.modality.lexical.get_name(index=idx) if idx >= 0 else False


class _WordOrtho(_Word):
    def __init__(self, modality, top_down=True, gamma_ratio_w=1e-3, **modality_args):
        super().__init__(modality=modality, top_down=top_down, gamma_ratio_w=gamma_ratio_w, **modality_args)

    #################################
    #### INFERENCES #################
    #################################

    def update_word(self):
        other_mod = self.modality.model.phono
        if other_mod.enabled:
            super().update_word(other_mod.word.dist)

    #################################
    #### INFO #######################
    #################################


class _WordPhono(_Word):
    def __init__(self, modality, top_down=True, gamma_ratio_w=1e-3, ld_thr=0.9, **modality_args):
        super().__init__(modality=modality, top_down=top_down, gamma_ratio_w=gamma_ratio_w, ld_thr=ld_thr, **modality_args)

    #################################
    #### INFERENCES #################
    #################################

    def update_word(self):
        other_mod = self.modality.model.ortho
        if other_mod.enabled:
            super().update_word(other_mod.word.dist)
