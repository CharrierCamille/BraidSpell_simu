# -*- coding: utf-8 -*-
# General purpose libraries
import copy
import heapq
# Scientific/Numerical computing
import itertools
import math
import os
import pdb
import logging
import pickle as pkl
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

# BRAID utlities
import braidpy.braid as braid
import braidpy.utilities as utl
import braidpy.lexicon as lex

# on désactive le mode debug de numba et numpy
from braidpy import detect_errors


## décorateurs des simulations individuelles

def generic_simu(func):
    def wrapper(self, *args, **kwargs):
        self.begin_simu()
        res = func(self, *args, **kwargs)
        self.end_simu()  # formater les distributions dans un format facilement utilisable

    return wrapper


def learning(func):
    def wrapper(self, *arg, **kw):
        res = func(self, *arg, **kw)
        if self.model.ortho.lexical.learning or self.model.phono.lexical.learning:
            self.model.ortho.lexical.learn()
            self.model.phono.lexical.learn()
        return res

    return wrapper


class simu:
    """ Simulation context, includes model, simulation parameters and the outputs """

    def __init__(self, model_param=None, ortho_param=None, phono_param=None, semantic_param=None, simu_args=None,
                 level='simu',
                 build_prototype=False, max_iter=1000, t_min=100, simu_type="H", thr_expo=0.2,
                 stop_criterion_type="pMean", pos_init=-1, segment_reading=False,
                 word_sampling_frequency=50, word_sampling_size=1000, sampling=False, reading_unit="letter",
                 fixation_criterion_type="ortho",thr_fix=0.4):
        """
        Object constructor, simulation context initiator

        Args:
            model_param, ortho_param, phono_param : dict, parameters for inner classes
            simu_args : dictionary of optional parameters for the simulation
            level : string, level for the logging package, among simu,expe,debug
            build_prototype : boolean, if True, build the prototype from the simulation file
            max_iter: int, number of iterations
            t_min : int, minimum number of iterations for a fixation
            simu_type : string, simulation type among normal, threshold, H, change_pos
            thr_expo : float, threshold for the end of the exposure (mean entropy on letters)
                # thr_expo à 0.1 dans ma thèse, mais 0.2 suffirait
            stop_criterion_type : for simulations with a termination criterion, like simu_H, criterion type to end the exposure
            pos_init : float, initial position of the attention. if set to -1, let the model decide
            serial_reading : string, among "letter", "grapheme" or "None". If "None", next position is chosen according to entropy. If "letter" or "phoneme", the next unit in the word is chosen.
        """


        if model_param is None:
            model_param = {}
        self.model = braid.braid(ortho_param, phono_param, semantic_param, **model_param)
        self.simu_args = simu_args if simu_args is not None else {}
        logging.basicConfig(format='%(levelname)s - %(message)s')
        self.level = level
        if build_prototype:
            self.build_prototype()
        self.max_iter = max_iter
        self.t_min = t_min
        self.word_sampling_frequency = word_sampling_frequency
        self.word_sampling_size = word_sampling_size
        self.sampling = sampling
        self.fixation_criterion_type = fixation_criterion_type
        self.simu_type = simu_type
        self.thr_expo = thr_expo
        self.thr_expo_phono = 1.0 * thr_expo
        self.stop_criterion_type = stop_criterion_type
        self.pos_init = pos_init
        self.segment_reading = segment_reading
        self.reading_unit = reading_unit
        self.init_res()
        self.init_simu_attributes()
        self.real_time=-1
        self.thr_fix = thr_fix

    ############################
    ### Getters / Setters ######
    ############################

    def __setattr__(self, name, value):
        """
        This function allows for setting attributes in a model, ortho, or phono class.

        :param name: The name of the attribute being set
        :param value: The value that is being assigned to the attribute named "name". This is the value that will be stored in the attribute
        """

        if 'model' in self.__dict__ and ((name in self.model.__dict__ or '_' + name in self.model.__dict__) or
                                         ('ortho' in self.model.__dict__ and name in self.model.ortho_param_names) or
                                         ('phono' in self.model.__dict__ and name in self.model.phono_param_names)):
            self.model.__setattr__(name, value)
        # definition of simulation args here to facilitate the run of simulations with class expe
        elif name in ['thr_fix', 'alpha', 'n_choose_word']:
            self.simu_args[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """
        This function tries to get an attribute from the model class if not present in the simu class. Otherwise it raises an AttributeError.

        :param name: name of the attribute that is being accessed.
        :return: value of the attribute.
        """
        if 'model' in self.__dict__:
            return getattr(self.model, name)

    def __getstate__(self):
        """
        Redefinition of this function necessary to use the pkl package because getattr is overwritten (do not touch)
        """
        return self.__dict__

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value
        try:
            logging.getLogger().setLevel(getattr(logging, value.upper()))
        except:
            print("except")
            logging.basicConfig(level=getattr(logging, value.upper()), format='%(levelname)s - %(message)s')

    @property
    def segment_reading(self):
        return self._segment_reading

    @segment_reading.setter
    def segment_reading(self, value):
        self._segment_reading = value
        try:
            self.model.ortho.attention.segment_reading = value
            self.model.phono.attention.segment_reading = value

        except:
            print("Attentional modules undefined : cannot set property segment reading")

    @property
    def reading_unit(self):
        return self._reading_unit

    @reading_unit.setter
    def reading_unit(self, value):
        self._reading_unit = value
        try:
            self.model.ortho.attention.reading_unit = value
            self.model.phono.attention.reading_unit = value
        except:
            print("Attentional modules undefined : cannot set property serial reading")

    ############################
    ### Init of the simulation ##
    ############################

    def init_res(self):
        """
        Initializes the dictionary of results
        """
        self.res = {"ortho": {}, "phono": {}}
        for mod in ["phono", "ortho"]:
            modality = getattr(self.model, mod)
            perceptual_submodels = ['percept', 'word']
            for submodel_str in perceptual_submodels:
                submodel = getattr(modality, submodel_str)
                for key, value in submodel.dist.items():
                    self.res[mod][key] = []

    def init_pos(self):
        """
        Initializes the position of the model : if its initial value is -1, it's automatically set. Otherwise, value is kept.
            /!\ the position needs to be set at -1 at the end of a simulation for this to work
            otherwise the next simulation will start with the last value
        """
        print(self.pos_init)
        if self.pos_init != -1 & self.pos_init != -2:
            if self.model.phono.enabled:
                if self.model.input_type == "auditory":
                    self.model.ortho.pos = 0
                    self.model.phono.pos = self.pos_init
                else:
                    self.model.ortho.pos = self.pos_init
                    self.model.phono.pos = 0
        elif self.pos_init == -1:
            self.model.ortho.attention.init_pos_auto()
            if self.model.phono.enabled:
                self.model.phono.pos = 0
        elif self.pos_init == -2:
            self.model.ortho.pos = round(len(self.stim) / 2)
            if self.model.phono.enabled:
                self.model.phono.pos = round(self.model.phono.attention.len_phlen_relation(self.model.ortho.pos))


    def init_removal(self):
        """
        Removes a stimulus from the orthographic and phonological representations when needed
        """
        self.model.ortho.lexical.remove_stim_repr()
        self.model.phono.lexical.remove_stim_repr()
        self.model.removed_stim = self.model.stim

    def init_simu_attributes(self):
        """
        Initializes various attributes for a simulation.
        """
        self.n_simu = 0
        self.error_type = "None"
        self.corrected = "None"
        self.pron_sans_corr = ""
        self.t_tot = 0
        self.model.mismatch = False
        self.model.chosen_modality = None
        self.model.phono.missing_letter = False
        self.reset = {"mean": True, "gazePos": True, "dist": True, "lexicon": True}
        if self.simu_type != "change_pos":
            self.fix = {mod: {key: [] for key in ["t", "att", "pos", "sd", "err"]} for mod in ["ortho", "phono"]}
        if self.model.input_type == "visual":
            self.simu_stim = self.model.ortho.stim
        elif self.model.input_type == "auditory":
            self.simu_stim = self.model.phono.stim
        self.HPhonoDer = [1000 for _ in range(20)]
        self.HOrthoDer = [1000 for _ in range(20)]

    def begin_simu(self):
        """
        Beginning of simulation common to all of simulation types
        """
        self.real_time = time()
        self.init_pos()
        self.model.reset_model(self.reset)
        self.init_removal()
        self.init_res()
        self.init_simu_attributes()
        self.update_fix_info()
        self.complete_dist()

    ############################
    ### Results handling #######
    ############################

    def complete_dist(self):
        """
        Takes the current model distributions and adds them to the simulation results
        """
        for mod in ["phono", "ortho"]:
            modality = getattr(self.model, mod)
            perceptual_submodels = ['percept', 'word']
            for submodel_str in perceptual_submodels:
                submodel = getattr(modality, submodel_str)
                for key, value in submodel.dist.items():
                    try:
                        self.res[mod][key] += [submodel.dist[key]]
                    except:
                        pdb.set_trace()

    def update_fix_info(self):
        """
        Updates the fixation information (position, standard deviation, time, attention profile) for the orthographic and phonological modules
        """
        self.fix['ortho']["pos"].append(self.model.ortho.pos)
        self.fix['ortho']["sd"].append(self.model.ortho.attention.sd)
        self.fix['ortho']["t"].append(self.t_tot)
        self.fix['ortho']["att"].append(self.model.ortho.attention.dist)
        if self.phono.enabled:
            self.fix["phono"]["pos"].append(self.model.phono.pos)
            self.fix["phono"]["att"].append(self.model.phono.attention.dist)
            self.fix['phono']["sd"].append(self.model.phono.attention.sd)
            self.fix['phono']["t"].append(self.t_tot)
            try:
                self.fix["err_phono"].append(abs(int(self.model.gs[self.model.ortho.pos]) - self.model.phono.pos))
            except:
                pass

    def delete_fixation(self, t):
        """
        Removes the last fixation from the stored result

        :param t: the number of fixations to delete
        """
        for mod in ["phono", "ortho"]:
            for name in self.model.dist_names:
                self.res[mod][name] = self.res[mod][name][:-max(t, 1)]
                # the state of the model must be the same as before the deleted fixations
                getattr(self.model, mod).dist[name] = self.res[mod][name][-1]
            for k, val in self.fix.items():
                self.fix[mod][k] = val[:-1]
        self.t_tot -= t
        logging.simu(f"fixation supprimée : {t} iterations")

    def increase_n(self):
        """
        Increases the number of simulations by one
        """
        self.n_simu += 1

    def reset_n(self):
        """
        Resets the number of simulations to zero
        """
        self.n_simu = 0

    ############################
    ### End of the simulation #######
    ############################

    def stopCriterion(self):
        """
         Detects if the creterion for the end of the exposure has been met
        """
        if self.stop_criterion_type == "pMax":
            return max(self.model.ortho.percept.get_entropy()) < self.thr_expo * math.log(self.model.ortho.n, 2)
        if self.stop_criterion_type in ["pMax","pMean"]:
            # return np.mean(self.model.ortho.percept.get_entropy()) < self.thr_expo * math.log(self.model.ortho.n, 2)
            if self.model.ortho.enabled:
                dz_pos = self.model.ortho.percept.get_dz_pos()
                try:
                    Hunif = math.log(self.model.ortho.n)
                    self.HOrthoDer = self.HOrthoDer[1:] + [self.model.ortho.percept.max_HDer]
                    if "Mean" in self.stop_criterion_type:
                        c1 = dz_pos > 1 and np.mean(
                            self.model.ortho.percept.get_entropy()[:dz_pos]) < self.thr_expo * Hunif
                        # print(self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                    elif "Max" in self.stop_criterion_type:
                        c1 = dz_pos > 1 and np.max(
                            self.model.ortho.percept.get_entropy()[:dz_pos]) < self.thr_expo * Hunif
                        # print(self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                        # print("\t",self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                except:
                    pdb.set_trace()
                return c1

            return c0 or c1
        elif self.stop_criterion_type in ["phiMean", "pphiMean", "phiMax"]:
            if self.model.phono.enabled:
                dz_pos = self.model.phono.percept.get_dz_pos()
                try:
                    Hunif = math.log(self.model.phono.n)
                    self.HPhonoDer = self.HPhonoDer[1:] + [self.model.phono.percept.max_HDer]
                    if "Mean" in self.stop_criterion_type:
                        c1 = dz_pos > 1 and np.mean(
                            self.model.phono.percept.get_entropy()[:dz_pos]) < self.thr_expo * Hunif
                        # print(self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                    elif "Max" in self.stop_criterion_type:
                        c1 = dz_pos > 1 and np.max(
                            self.model.phono.percept.get_entropy()[:dz_pos]) < self.thr_expo * Hunif
                        # print(self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                        # print("\t",self.t_tot,dz_pos,self.model.phono.percept.get_entropy()[:dz_pos], self.thr_expo * Hunif)
                except:
                    pdb.set_trace()
                if self.stop_criterion_type in ["phiMean", "phiMax"]:
                    return c1
            c0 = np.mean(self.model.ortho.percept.get_entropy()) < 0.25 * self.thr_expo * math.log(self.model.ortho.n,
                                                                                                   2)
            return c0 or c1
        elif self.stop_criterion_type == "W":
            return np.max(self.model.ortho.word.dist["word"]) > 0.9
        elif self.stop_criterion_type == "WPhi":
            return np.max(self.model.phono.word.dist["word"]) > 0.9
        elif self.stop_criterion_type == "ld":
            ld = self.model.ortho.word.dist["ld"][0]
            return ld < 0.1 or ld > self.model.ortho.word.ld_thr
        elif self.stop_criterion_type == "ld_phi":
            ld = self.model.phono.word.dist["ld"][0]
            return ld < 0.1 or ld > self.model.phono.word.ld_thr
        elif self.stop_criterion_type == "phono":
            return max(self.model.phono.dist["percept"][0]) > self.thr_expo
        elif self.stop_criterion_type == "phono_pos":
            return max(self.model.phono.dist["percept"][self.model.phono.pos]) > self.thr_expo
        elif self.stop_criterion_type == "bothMean":
            dz_pos_ortho = self.model.ortho.percept.get_dz_pos()
            dz_pos_phono = self.model.phono.percept.get_dz_pos()
            try:
                h_unif_ortho = math.log(self.model.ortho.n,2)
                h_unif_phono = math.log(self.model.phono.n,2)
                h_ortho = np.mean(self.model.ortho.percept.get_entropy()[:dz_pos_ortho])
                h_phono = np.mean(self.model.phono.percept.get_entropy()[:dz_pos_phono])

                h_r_ortho = h_ortho / h_unif_ortho
                h_r_phono = h_phono / h_unif_phono

                return np.mean((h_r_ortho, h_r_phono)) < self.thr_expo
            except:
                pdb.set_trace()





    def PM_decisions(self):
        """
        This function makes novelty decisions for the orthographic and phonological branches of the model and then makes a global decision based on both decisions.
        """
        # decision in each modality
        self.model.ortho.word.PM_decision()
        self.model.phono.word.PM_decision()
        # amodal decision
        self.model.PM_decision_global()

    def reshape_results(self):
        """
        This function reshapes the results for easier use.
        """

        for mod in ["ortho", "phono"]:
            for name in self.res[mod].keys():
                try:
                    self.res[mod][name] = np.moveaxis(self.res[mod][name], 0, -1)
                except:
                    print("error in reshape_results")
                    pdb.set_trace()
                    pass

    def print_results(self):
        """
        Prints the results of the simulation. See the notebook one_word.ipynb for more information.
        """

        input = self.model.input_type
        logging.simu(f"modalité {self.model.input_type}")
        if input == "visual":
            ex = self.model.phono.enabled and len(self.model.phono.stim) > 0
            logging.simu(f"stimulus {self.simu_stim}, {self.model.phono.stim if ex else 'NO PHONO REPR'}")
            try:
                logging.simu(f"freq = " + str(self.model.ortho.lexical.df.loc[self.model.ortho.stim].freq))
            except:
                pass
        elif input == "auditory":
            ex = self.model.ortho.enabled and len(self.model.ortho.stim) > 0
            logging.simu(f"stimulus {self.simu_stim}, {self.model.ortho.stim if ex else 'NO ORTHO REPR'}")
            try:
                logging.simu(f"freq = " + str(self.model.phono.lexical.df.loc[self.model.phono.stim].freq))
            except:
                pass
        logging.simu(f"lexical status ortho: {'novel' if self.model.ortho.lexical.remove_stim else 'known'}")
        logging.simu(f"lexical status phono: {'novel' if self.model.phono.lexical.remove_stim else 'known'}")
        logging.simu(f"simulation duration : {self.t_tot}")
        if self.model.mixture_knowledge:
            if self.simu_stim in self.model.ortho:
                tp = self.model.ortho.lexical.df.loc[self.simu_stim].repr_type
                tp_str = 'expert' if tp == 0 else 'enfant' if tp == 1 else 'inconnu'
                logging.simu(f"type of ortho representation: {tp_str}")
        self.ortho.print_all_dists()
        self.phono.print_all_dists()
        logging.simu("\n IDENTIFICATION")
        logging.simu(f"chosen modality: {self.chosen_modality if self.chosen_modality is not None else 'None'}")
        if self.model.semantic.context_sem:
            if not self.PM:
                logging.simu(
                    f"Context decision: known word {self.model.chosen_modality} {self.model.ortho.word.chosen_word}")
            else:
                logging.simu(f"Context decision: novel word")
        ident_type = self.model.chosen_modality if self.model.chosen_modality is not None else 'fusion' if self.model.fusion else 'phono'
        logging.simu(f"Identification {ident_type}: /{self.model.phono.word.decision('word')}/")
        logging.simu(f"\n SUCCESS ")
        if self.model.phono.enabled:
            if self.model.phono.percept.evaluate_decision():
                logging.simu("Psi Ok")
            else:
                logging.simu("Erreur WPhi: " + self.error_type)
        logging.simu("WFusion Ok" if utl.str_transfo(
            self.model.ortho.word.chosen_word) == self.model.ortho.stim else "Erreur W Fusion")
        logging.simu("\n FIXATIONS")
        logging.simu(f"fixation times: {self.fix['ortho']['t']}")
        logging.simu(f"fixation positions : {self.fix['ortho']['pos']}")
        logging.simu(f"fixation dispersion : {self.fix['ortho']['sd']}")
        if self.model.phono.enabled:
            logging.simu(f"fixation phono positions : {self.fix['phono']['pos']}")
            logging.simu(f"errors phono positions : {self.fix['phono']['err']}")
        logging.simu(f"\n USED WORDS")
        if input == "visual":
            for key, value in self.model.phono.percept.get_used_words().items():
                logging.simu(f"position {key}, \n words {value}")
        elif input == "auditory":
            for key, value in self.model.ortho.percept.get_used_words().items():
                logging.simu(f"position {key}, \n words {value}")

        logging.simu("\n")

    def end_simu(self):
        """
        This function ends a simulation : it changes the shape of the results for easier use, makes decisions based on orthographic and phonological results,
        detects error types, resets eye position, restores stimulus representation, prints results
        """
        self.PM_decisions()
        self.reshape_results()
        self.model.ortho.pos = -1
        self.print_results()
        # self.detect_error_type()

    #############################
    #### Results generation #####
    #############################

    def getH(self):
        """
        Returns the entropy of the percept.
        """
        return [[entropy(i) for i in p] for p in np.moveaxis(self.model.ortho.percept.dist["percept"], -1, 0)]

    def one_res(self, typ):
        """
        Returns the result according to name

        :param typ: the type of result you want to get
        no explicit : get the evolution of the frequency of a word during learning + lexical decision
        ld : get lexical decision distribution over time
        ld_end : get lexical decision distribution at last iteration
        dirac : get the maximum of the L dstribution (quasi dirac for adults)
        sd : return attention dispersion chosen (sd at the end of the simulation)
        meanH : return the mean entropy of the letters
        sumH : return the sum entropy of the letters
        duree fix : return the durations of the fixations
        first phoneme : maximum of the first phoneme distribution
        """

        if typ == "entropy": pass
        if typ == "ld_ortho": return self.model.ortho.word.dist["ld"][0]
        if typ == "ld_phono": return self.model.phono.word.dist["ld"][0]
        if typ == "ld_ortho_all": return self.res["ortho"]["ld"][0]
        if typ == "ld_phono_all": return self.res["phono"]["ld"][0]
        if typ == "PM": return self.model.PM
        if typ == "PM_ortho": return self.model.ortho.PM
        if typ == "PM_phono": return self.model.phono.PM
        if typ == "dirac": return [self.model.get_dirac(self.simu_stim)]
        if typ == "sd": return [self.model.ortho.attention.sd]
        if typ == "meanH": return [np.mean(i) for i in self.getH()]
        if typ == "sumH": return [sum(i) for i in self.getH()]
        if typ == "t_tot": return self.t_tot
        if typ == "duree_fix": return np.diff(self.fix['ortho']["t"])
        if typ == "oculo": return [len(self.fix['ortho']["t"]), self.t_tot, self.get_res()[0][-1]]  # ne plus utiliser
        if typ == "parcours_visuel": return [len(self.fix['ortho']["t"]), self.t_tot,
                                             self.model.ortho.attention.sd]  # ne plus utiliser
        if typ == "fixations_visuelles": return self.fix['ortho']["pos"]
        if typ == "fixations_phono": return self.fix["phono"]["pos"]
        if typ == "first_phoneme": return np.max(self.model.phono.percept.dist["percept"][0])
        if typ == "first_phoneme75": return np.max(self.res['phono']['percept'][0, :, 75])
        if typ == "phi": return self.model.phono.percept.decision()
        if typ == "let": return self.model.ortho.percept.decision()
        if typ == "pron_sans_corr": return self.pron_sans_corr
        if typ == "wphi": return self.model.phono.word.decision("word")
        if typ == "wl": return self.model.ortho.word.decision("word")
        if typ == "wfusion": return self.model.ortho.word.chosen_word
        if typ == "maxwphi": return max(self.model.phono.word.dist["word"])
        if typ == "psi_score": return self.model.phono.psi_score(self.simu_stim)
        if typ == "instability": return self.max_der()
        if typ == "correction": return self.corrected
        if typ == "success_correction": return self.success_correction
        if typ == "id_sans_corr": return self.id_sans_corr
        if typ == "sum_err_pos_phono": return np.mean(self.fix['phono']["err"])
        if typ == "pmax_middle_letter": return np.max(self.model.ortho.percept.dist["percept"][1])
        if typ == "pmax_middle_letter_150": return np.max(self.res["ortho"]["percept"][1,:,150])
        if typ == "pmax_first_letter": return np.max(self.model.ortho.percept.dist["percept"][0])
        if typ == "dec_2nd_let": return self.model.ortho.percept.decision()[1]
        if typ == "dist_max_word_ortho": return self.res['ortho']['word'][np.argmax(self.model.ortho.word.dist["word"])]
        if typ == "let_all": return self.model.ortho.percept.dist["percept"]
        if typ == "phi_all": return self.model.phono.percept.dist["percept"]
        if typ == "wphi_all": return self.model.phono.word.dist["word"]
        if typ == "wlet_all": return self.model.ortho.word.dist["word"]
        if typ == "max_let": return np.max(self.model.ortho.percept.dist["percept"], axis=1)
        if typ == "distletatt_0" : return np.max(self.res["ortho"]["percept"],axis=1)[0].tolist()
        if typ == "distletatt_1": return np.max(self.res["ortho"]["percept"], axis=1)[1].tolist()
        if typ == "distletatt_2": return np.max(self.res["ortho"]["percept"], axis=1)[2].tolist()
        if typ == "distletatt_3": return np.max(self.res["ortho"]["percept"], axis=1)[3].tolist()
        if typ == "distletatt_4": return np.max(self.res["ortho"]["percept"], axis=1)[4].tolist()

        if typ == "simu_time": return time() - self.real_time

    #############################
    #### Error Analysis #####
    #############################

    def success(self, typ="phi"):
        """
        Returns a boolean value indicating whether the simulation was successful or not, depending on the measure considered

        :param typ: the type of success we want to measure, defaults to phi (optional)
        :return: The success of the simulation according to the measure considered.
        """
        if typ == "phi":
            return self.model.phono.percept.evaluate_decision()
        elif typ == "psi_score":
            return self.model.phono.psi_score(self.simu_stim) > 0.9
        elif typ == "let":
            return self.model.ortho.percept.evaluate_decision()
        elif typ == "wl":
            return self.model.ortho.word.evaluate_decision("word")
        elif typ == "wphi":
            return self.model.phono.word.evaluate_decision("word")
        elif typ == "wfusion":
            return utl.str_transfo(self.model.ortho.word.chosen_word) == self.model.ortho.stim
        elif "ortho" in typ or "phono" in typ:  # ld_phono/ortho ou PM_ortho/phono
            data = getattr(self.model, "ortho" if "ortho" in typ else "phono")
            is_PM = self.n_simu == 0 and (
                    data.lexical.remove_stim or not self.model.stim in self.model.ortho.lexical.df.index)
            return (data.word.dist["ld"][
                        0] < data.word.ld_thr) == is_PM if "ld" in typ else data.PM == is_PM if "PM" in typ else False
        elif typ == "PM":
            return (self.n_simu == 0 and self.model.PM and (
                    (self.model.ortho.remove_stim and self.model.phono.remove_stim) or (
                not self.model.stim in self.model.ortho.lexical.df.index))) \
                   or (self.n_simu > 0 and not self.model.PM)
        elif typ == "correction":
            return self.model.phono.percept.evaluate_decision()
        elif typ == "pron_sans_corr":
            return utl.str_eq(self.pron_sans_corr, self.model.phono.stim)
        elif typ == "duree_fix":
            return [True if self.model.PM else False] + [True] * len(np.diff(self.fix['ortho']["t"] + [self.t_tot]) - 1)

    def detect_lexicalisation_error(self):
        """
        Detects lexicalisation errors, which are defined as the case where the model's fusion word is not equal to the stimulus
        """
        return not self.model.PM and utl.str_transfo(self.model.ortho.word.chosen_word) != self.model.stim

    def detect_context_error(self):
        """
        Detects context errors, which are defined as the case where the pronunciation of the word is in the list of context semantic words
        """
        return self.model.phono.percept.decision() in self.model.semantic.context_sem_words_phono or self.model.ortho.detect_context_error() or self.model.phono.detect_context_error()

    def detect_missing_letter_error(self):
        """
        Detects error when no word used for decoding had some letter in stimulus
        """
        used_words = [i.split('_')[0] for j in self.model.phono.percept.get_used_words().values() for i in j]
        stim = self.model.ortho.stim
        for l in range(len(stim)):
            if not any([stim[l] == wd[l] for wd in used_words]):
                return True
        return False

    def detect_missing_bigram_error(self):
        """
        Detects error when no word used for decoding had some bigram in stimulus
        """
        used_words = [i.split('_')[0] for j in self.model.phono.percept.get_used_words().values() for i in j]
        stim = self.model.ortho.stim
        for l in range(len(stim) - 1):
            if not any([stim[l:l + 2] == wd[l:l + 2] for wd in used_words]):
                return True
        return False

    def detect_error_type(self):
        """
        Detects the type of error made by the model while decoding
        """
        self.error_type = "None"
        if self.model.phono.enabled:
            str1 = utl.str_transfo(self.model.phono.stim)
            str2 = utl.str_transfo(self.model.phono.percept.decision())
            if str1 != str2:
                self.error_type = "unknown"
                if detect_errors.detect_substitution_error(str1, str2):
                    self.error_type = "substitution"
                    if detect_errors.detect_end_error(str1, str2): self.error_type = "end substitution"
                    if detect_errors.detect_substitution_grapheme_error(str1,
                                                                        str2): self.error_type = "grapheme substitution"
                    if detect_errors.detect_substitution_schwa_error(str1, str2): self.error_type = "schwa substitution"
                    if detect_errors.detect_substitution_n_error(self.model.stim, str1,
                                                                 str2): self.error_type = "xnx substitution"
                    if detect_errors.detect_insertion_grapheme_error(str2): self.error_type = "grapheme insertion"
                if detect_errors.detect_insertion_error(str1, str2)[1]: self.error_type = "insertion"
                err = detect_errors.detect_deletion_error(str1, str2)
                if err != "":
                    self.error_type = err
                if self.detect_lexicalisation_error():
                    self.error_type = "lexicalisation"
                    if self.detect_context_error(): self.error_type = "context"
                if self.model.mismatch:
                    self.error_type = "mismatch detected"
                    if self.detect_context_error(): self.error_type = "context"
                if self.detect_missing_bigram_error(): self.error_type = "missing bigram"
                if self.detect_missing_letter_error(): self.error_type = "missing letter"
                ph = self.model.phono
                if len(ph.stim) > 0 and len(
                        ph.lexical.df.loc[ph.stim]) > 1 and str1 == str2: self.error_type = "homophone"

    #################################################################
    ###### Simulations corresponding to one exposure to one word ####
    ################################################################

    def run_simu_general(self):
        """
        Runs the simulation corresponding to it's name : normal, app, change_pos, grid_search, H
        """
        # print(self.simu_type)
        getattr(simu, "run_simu_" + self.simu_type)(self, **self.simu_args if self.simu_args is not None else {})

    def sample_words(self):
        """
        Samples words from the lexicon
        """


        if self.model.input_type == "visual" and self.word_sampling_size<len(self.model.ortho.lexical.sample_idx):
            self.model.ortho.lexical.build_sample(self.word_sampling_size)
            # print(f"sample words: {self.model.ortho.lexical.get_names(self.model.ortho.lexical.sample_idx)}")
            # print(f"sample size: {len(self.model.ortho.lexical.sample_idx)}")
        elif self.model.input_type == "auditory"and self.word_sampling_size<len(self.model.phono.lexical.sample_idx):
            self.model.phono.lexical.build_sample(self.word_sampling_size)
            # print(f"sample words: {self.model.phono.lexical.get_names(self.model.phono.lexical.sample_idx)}")
            # print(f"sample size: {len(self.model.phono.lexical.sample_idx)}")

    def one_step_general(self):
        """
        Runs one step corresponding to the type : normal, phono (others to come)
        """
        logging.debug(f"iteration {self.t_tot}")
        if self.sampling:
            if np.max(self.model.ortho.word.dist["word"])>0.5 or np.max(self.model.phono.word.dist["word"])>0.5:
                if self.t_tot % self.word_sampling_frequency == 0:
                    self.sample_words()
        self.model.one_iteration()
        self.complete_dist()
        self.t_tot += 1

    @generic_simu
    def run_simu_normal(self,thr_fix):
        """
        Runs the simulation for `max_iter` iterations, where each iteration is a call to the function `one_step_general`
        """
        for t in range(self.max_iter):
            self.one_step_general()

    @generic_simu
    def run_simu_stim_disappear(self,thr_fix):
        """
        Run the simulation for 150 it then the stimulus become a uniform over letters
        """
        for t in range(self.max_iter):
            if t==150:
                self.simu_stim="~~~"
                self.model.ortho.stim = "~~~"
            self.one_step_general()

    @learning
    def run_simu_app(self):
        """
        Runs the simulation in normal mode with learning at the end
        """
        self.run_simu_normal()

    @generic_simu
    def run_simu_threshold(self):
        """
        Runs the simulation until the stop criterion is met
        """
        for t in range(self.max_iter):
            if not self.stopCriterion():
                self.one_step_general()

    # @learning
    @generic_simu
    def run_simu_change_pos(self):
        """
        Runs a simulation where the times, ortho positions and attentional dispersions of all fixations are given in advance in the dictionary self.fix
        """
        if self.fix is None:
            self.fix = {"t": [], "pos": [], "pos_phono": [], "sd": []}
        if 0 not in self.fix['ortho']["t"]:  # si on renseigne pas 0, on le rajoute à la main dans les fix faites
            self.fix['ortho']["t"] = [0] + self.fix['ortho']["t"]
            self.fix['ortho']["pos"] = [self.model.ortho.pos] + self.fix['ortho']["pos"]
            if self.model.phono.enabled:
                self.fix["phono"]["pos"] = [self.model.phono.pos] + self.fix["phono"]["pos"]
            self.fix['ortho']["sd"] = [self.model.ortho.attention.sd] + self.fix['ortho']["sd"]
        self.fix['ortho']["att"] = [];
        if self.phono.enabled:
            self.fix["phono"]["att"] = [];
        self.fix['ortho']["sd"] = []
        for i in np.arange(0, self.max_iter):
            if i in self.fix['ortho']["t"]:
                idx = self.fix['ortho']["t"].index(i)
                self.model.pos = self.fix['ortho']["pos"][idx]
                self.fix['ortho']["att"] += [list(self.model.ortho.attention.dist)]
                if self.model.phono.enabled:
                    self.fix["phono"]["att"] += [list(self.model.phono.attention.dist)]
                self.fix['ortho']["sd"].append(self.model.ortho.attention.sd)
            self.one_step_general()

    ########################################
    ### subsidiary functions for simu_H ####
    ########################################

    def build_prototype(self):
        """
        Builds the entropy prototype + derivative prototype also
        """
        eng = "Eng" if self.model.langue == "en" else ""
        with open(os.path.realpath(
                os.path.dirname(__file__)) + '/../../codethese/ParcoursVisuel/pkl/HProto' + eng + 'M.pkl', 'rb') as f:
            [df, *rest] = pkl.load(f)
        df['len'] = df.word.str.len()
        df = df.groupby(['len', 't']).mean()
        dfH = df['value'].unstack(level=0)
        dfHDer = dfH.diff(periods=-1).dropna()
        with open(os.path.realpath(os.path.dirname(__file__)) + '/../resources/prototype/HProto' + eng + '.pkl',
                  'wb') as f:
            pkl.dump([dfH, dfHDer], f)

    def get_prototype(self, der=False):
        """
        Gets the simulation prototype
        """
        eng = "Eng" if self.model.langue == "en" else ""
        with open(os.path.realpath(os.path.dirname(__file__)) + '/../resources/prototype/HProto' + eng + '.pkl',
                  'rb') as f:
            [dfH, dfHDer] = pkl.load(f)
        if (self.model.input_type == "auditory") & (len(self.model.phono.stim) in dfHDer.keys()):
            return list(dfHDer[len(self.model.phono.stim)]) if der else list(dfH[len(self.model.phono.stim)])
        elif (self.model.input_type == "visual") & (len(self.model.ortho.stim) in dfHDer.keys()):
            return list(dfHDer[len(self.model.ortho.stim)]) if der else list(dfH[len(self.model.ortho.stim)])
        return None

    def run_correction(self, len_corr):
        """
        The function runs a correction by adjusting the gamma ratio to a high value and running a specified number of time steps in the model.

        :param len_corr: len_corr is an integer parameter that represents the number of correction steps to be performed by the model.
        """
        self.one_step_general()  # pour ne pas écraser le percept
        self.model.recalage_stim()
        if self.model.recalage:
            old_word_reading = self.model.word_reading
            self.model.word_reading = True
            self.model.phono.gamma_ratio *= 10
            for _ in range(len_corr):
                self.one_step_general()
            self.model.phono.gamma_ratio /= 10
            self.model.word_reading = old_word_reading

    def delete_correction(self, len_corr):
        """
        This function deletes the pronunciation correction if needed.

        :param len_corr: `len_corr` is an integer parameter that represents the duration of the pronunciation correction.
        """
        for mod in ["ortho", "phono"]:
            for name in self.model.dist_names:
                old_dist = self.res[mod][name][-(len_corr + 1)]
                for i in range(1, len_corr + 1):
                    self.res[mod][name][-i] = old_dist
                getattr(self.model, mod).dist[name] = old_dist

    def pronunciation_correction(self):
        """
        This function checks if a correction is needed for a pronunciation and performs the correction if necessary.
        """
        len_corr = 100
        if self.model.phono.enabled:
            self.pron_sans_corr = self.model.phono.percept.decision()
            self.id_sans_corr = self.model.phono.word.evaluate_decision() if len(
                self.model.ortho.lexical.repr) > 0 else False
            # necessity of correction
            if self.phono.enabled and self.model.semantic.context_sem:
                if not self.model.semantic.top_down and (self.model.word_reading or (
                        self.model.semantic.context_sem and self.model.p_sem > 1)) and not self.model.phono.decision(
                    "ld", ld_thr=0.85):
                    logging.simu(f"succès de l'identification avant correction : {self.id_sans_corr}")
                    logging.simu(f"CORRECTION A {self.t_tot}")
                    logging.simu(f"percept avant correction : {self.pron_sans_corr}")
                    logging.simu(f"percept après correction : {self.model.phono.decision('percept')}")
                    self.run_correction(len_corr)
                    self.success_correction = self.model.phono.evaluate_decision("percept")
                    logging.simu(f"succès de la correction : {self.success_correction}")
                    if (self.model.phono.decision(
                            dist_name="word") in self.model.semantic.context_sem_words_phono and self.model.p_sem > 1 and not self.model.detect_mismatch()) or self.model.word_reading:
                        self.corrected = "kept"
                        logging.simu(f"CORRECTION GARDÉE")
                    else:
                        self.corrected = "deleted"
                        logging.simu("CORRECTION SUPPRIMÉE")
                        self.correction = False
                        self.delete_correction(len_corr)

    def update_sigma(self, Hinit, H):
        """
        This function updates the standard deviation of the attention distribution based on the speed of perceptual information accumulation compared to a prototype

        :param Hinit: Hinit is a list of initial entropies for each position. It is used to check the validity of the prototype.
        :param H: H is a list of current entropies for each position.
        """
        print("UPDATING")
        proto = self.get_prototype()  # prototype entropy
        sd_list = [3, 2.5, 2, 1.75, 1.5, 1.25, 1, 0.9, 0.8, 0.7, 0.6, 0.5]
        rapport_list = [20, 5, 2, 1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -10]
        rapport = (sum(Hinit) - sum(H)) / (sum(Hinit) - proto[self.t_tot - 1])
        if rapport > 0:
            sd = next(sd for sd, rapp in zip(sd_list, rapport_list) if rapport > rapp)
            self.model.ortho.attention.sd = sd

    def update_entropy(self, t, alpha, thr_fix):
        """
        This function updates all measures related to entropy : the letter entropy, a list of positional entropy values modulated by motor cost, and a criterion for stopping
        the fixation according to entropy values

        :param t: t is a variable representing time since the beginning of the fixation.
        :param alpha : motor cost
        :param thr_fix : entropy threshold to stop a fixation
        :return: three values: H, Hesp, and criterion1.
        """
        # print(HPrec,self.model.phono.percept.H,self.model.phono.percept.mean_HDer)

        HPhonoPrec = copy.copy(self.model.phono.percept.H)
        HPrec = copy.copy(self.model.ortho.percept.H)
        self.model.phono.percept.H = self.model.phono.percept.get_entropy()
        self.model.ortho.percept.H = self.model.ortho.percept.get_entropy()
        # self.model.phono.percept.max_HDer = np.max(
        #     HPhonoPrec[0:len(self.model.phono.percept.H)] - np.array(self.model.phono.percept.H))
        if self.fixation_criterion_type == "phono":
            dz_pos = self.model.phono.percept.get_dz_pos()
            Hesp = [(1 - alpha) * (h - self.model.phono.percept.H[self.model.phono.pos]) / np.max(
                self.model.phono.percept.H[:dz_pos]) - alpha * abs(i - self.model.phono.pos) - thr_fix for i, h in
                    enumerate(self.model.phono.percept.H[:dz_pos])]
            criterion1 = max(Hesp) > 0 and (t >= self.t_min)
            # surtout pas ça -> and max(HPhonoPrec - np.array(self.model.phono.percept.H)) < 1e-3)  # pour éviter H'<1e-6 à t=0
            self.HDerPhono = HPhonoPrec - np.array(self.model.phono.percept.H)
            # print(self.t_tot,criterion1,Hesp, max(HPhonoPrec - np.array(self.model.phono.percept.H)))
            # print(self.t_tot,criterion1,Hesp,self.model.ortho.pos,self.model.phono.pos)
        else:  # ortho
            Hesp = [((1 - alpha) * (h - self.model.ortho.percept.H[self.model.ortho.pos]) / np.max(
                self.model.ortho.percept.H) -
                     alpha * abs(i - self.model.ortho.pos)) - thr_fix for i, h in enumerate(self.model.ortho.percept.H)]
            # si on attend que ça se stabilise, impossible de changer d'avis après!
            # l'objectif c'est de pas rester trop longtemps sur la fixation
            # surtout pas de cotrainte de dérivée qui attend que ça se stabilise, sinon c'est trop stabilisé et on revient pas en arrière
            criterion1 = max(Hesp) > 0 and (t >= self.t_min)
            # surtout pas ça -> and (t >= self.t_min and max(HPrec - np.array(self.model.ortho.percept.H)) < 1e-2)  # pour éviter H'<1e-6 à t=0
            # if criterion1:
            #    print(self.t_tot,Hesp,np.mean(self.model.ortho.percept.H),"\n",[(h - self.model.ortho.percept.H[self.model.ortho.pos]) for i, h in enumerate(self.model.ortho.percept.H)],"\n\n")
        return Hesp, criterion1

    def update_position(self, Hesp):
        """
        This function updates the orthographic and phonological position of the model based on letter identity information on each position

        :param Hesp: Hesp is a list of floats representing the level of orthographic uncertainty of each position modulated by motor cost
        """
        pos = self.model.ortho.pos
        if self.segment_reading and (self.reading_unit == "letter" or self.reading_unit == "grapheme"):
            if self.reading_unit == "letter":
                self.model.ortho.pos = pos + 1 if pos < self.model.ortho.N - 1 else 0
                if self.model.phono.enabled:
                    self.model.phono.attention.calculate_attention_parameters()
            elif self.reading_unit == "grapheme":
                gs = self.model.gs
                next_grapheme = str(int(gs[pos]) + 1)
                self.model.ortho.pos = gs.index(next_grapheme) if next_grapheme in gs else 0
                if self.model.phono.enabled:
                    self.model.phono.attention.calculate_attention_parameters()
        else:
            if self.fixation_criterion_type == 'phono':
                pos_tmp_phono = np.argmax(Hesp)
                # oblige change position quand reste "coincé" sans gagner de l'info
                if pos_tmp_phono == self.model.phono.pos and abs(self.HPhonoDer[pos_tmp_phono]) < 1e-3:
                    pos_tmp_phono = Hesp.index(heapq.nlargest(2, Hesp)[-1])
                # we stop at the first # well perceived
                pos_tmp_phono = min(pos_tmp_phono, self.model.phono.percept.get_dz_pos() - 1)
                self.model.phono.pos = pos_tmp_phono
                self.model.ortho.attention.calculate_attention_parameters()
            elif self.fixation_criterion_type == 'ortho':
                pos_tmp = np.argmax(Hesp)
                # oblige change position
                if pos_tmp == pos:
                    try:
                        pos_tmp = self.model.phono.attention.len_phlen_relation(Hesp.index(heapq.nlargest(2, Hesp)[-1]))
                    except:
                        pdb.set_trace()
                self.model.ortho.pos = min(len(self.model.stim) - 1, pos_tmp)
                if self.model.phono.enabled:
                    self.model.phono.attention.calculate_attention_parameters(end_verif=True)
                # si on a dépassé la fin du mot, on revient en arrière
        self.update_fix_info()

    @learning
    @generic_simu
    def run_simu_H(self, alpha=0.1, thr_fix=0.25):
        """
        This function runs a simulation with visuo-attentional exploration of the orthographic stimulus based on entropy optimization

        :param alpha: float, motor cost (careful, if too high, typically 0.2, simulation can stuck in one position)
        :param thr_fix: float, threshold for a new fixation (entropy difference)
        """
        # print(f"Initial entropy")
        self.thr_fix = thr_fix
        self.model.ortho.percept.H = self.model.phono.percept.get_entropy()
        self.model.phono.percept.H = self.model.phono.percept.get_entropy()
        HUnifOrtho = self.model.ortho.percept.get_entropy(self.model.ortho.lexical.get_empty_percept())
        HUnifPhono = self.model.phono.percept.get_entropy(self.model.phono.lexical.get_empty_percept())
        Hinit = copy.copy(self.model.ortho.percept.H)
        # re une liste et pop et rajouter un élément avec un enumerate qui selectionne
        while self.t_tot < self.max_iter:
            for self.t_local in range(500):
                if self.t_tot < self.max_iter:
                    # print(self.t_local)
                    logging.debug('\n\n')
                    logging.debug(f"TIME {self.t_tot}")
                    self.one_step_general()
                    # print(f"Update entropy")
                    Hesp, criterion1 = self.update_entropy(self.t_local, alpha, thr_fix)
                    criterion2 = self.stopCriterion()
                    if (criterion1 and self.t_local > self.t_min) or criterion2 or self.t_tot == self.max_iter:
                        # end of fixation
                        logging.debug(
                            f"fin fixation,  time : {self.t_local}, crit Hdiff : {criterion1}, critHExpo : {criterion2}")
                        logging.debug(f"{self.model.ortho.percept.get_entropy()}")
                        break;
            if criterion2 or self.t_tot == self.max_iter or len(
                    self.fix['ortho']['t']) > 2 * self.model.ortho.N:  # end of exposure
                self.pronunciation_correction()
                return
            self.update_position(Hesp)  # all fixations : position update
            if False and len(self.fix['ortho']["t"]) == 1:  # first fixation : sigma update
                self.update_sigma(Hinit, H)
        return  # if t_tot > max_iter

    @generic_simu
    def run_simu_H_normal(self, **kwargs):
        """
        Runs a simulation (orthographic input) based on entropy optimization to determine the visual and phonological positions.
        :param kwargs: optional arguments for the run_simu_H function
        """
        # print(f"Initial entropy")
        alpha = 0.1
        thr_fix = 0.25
        self.thr_fix = thr_fix
        self.model.ortho.percept.H = self.model.phono.percept.get_entropy()
        self.model.phono.percept.H = self.model.phono.percept.get_entropy()
        HUnifOrtho = self.model.ortho.percept.get_entropy(self.model.ortho.lexical.get_empty_percept())
        HUnifPhono = self.model.phono.percept.get_entropy(self.model.phono.lexical.get_empty_percept())
        Hinit = copy.copy(self.model.ortho.percept.H)
        # re une liste et pop et rajouter un élément avec un enumerate qui selectionne
        while self.t_tot < self.max_iter:
            for self.t_local in range(500):
                if self.t_tot < self.max_iter:
                    # print(self.t_local)
                    logging.debug('\n\n')
                    logging.debug(f"TIME {self.t_tot}")
                    self.one_step_general()
                    # print(f"Update entropy")
                    Hesp, criterion1 = self.update_entropy(self.t_local, alpha, thr_fix)
                    # criterion2 = self.stopCriterion()
                    # if (criterion1 and self.t_local > self.t_min) or criterion2 or self.t_tot == self.max_iter:
                    if (criterion1 and self.t_local > self.t_min) or self.t_tot == self.max_iter:
                        # end of fixation
                        # logging.debug(
                        #     f"fin fixation,  time : {self.t_local}, crit Hdiff : {criterion1}, critHExpo : {criterion2}")
                        # logging.debug(f"{self.model.ortho.percept.get_entropy()}")
                        break;
            # if criterion2 or self.t_tot == self.max_iter or len(
            if self.t_tot == self.max_iter or len(
                    self.fix['ortho']['t']) > 2 * self.model.ortho.N:  # end of exposure
                self.pronunciation_correction()
                return
            self.update_position(Hesp)  # all fixations : position update
            if False and len(self.fix['ortho']["t"]) == 1:  # first fixation : sigma update
                self.update_sigma(Hinit, H)
        return  # if t_tot > max_iter

    def run_simu_att_phono_auto_continuous(self, **kwargs):
        """
        Runs a simulation (orthographic input) based on entropy optimization to determine the visual position, but the phonological position
        is automatically determined according to the graphemic segmentation.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'None'
        self.model.phono.attention.att_phono_auto = True
        self.model.phono.attention.segment_reading = False
        self.run_simu_H(**kwargs)

    def run_simu_att_phono_auto_segment(self, **kwargs):
        """
        Runs a simulation (orthographic input)based on entropy optimization to determine the visual position, but the phonological position
        is automatically determined according to the graphemic segmentation.
        The attention distribution is not a Gaussian but a Dirac on the good position.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'None'
        self.model.phono.attention.att_phono_auto = True
        self.model.phono.attention.segment_reading = self.model.ortho.attention.segment_reading = True
        self.run_simu_H(**kwargs)

    def run_simu_letter_continuous(self, **kwargs):
        """
        Runs a simulation (orthographic input) where the stimulus is processed letter-by-letter. The phonological position
        is calculated through an statistical approximation between number of letters/phonemes in a word.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'letter'
        self.model.phono.attention.att_phono_auto = False
        self.model.phono.attention.segment_reading = False
        self.run_simu_H(**kwargs)

    def run_simu_letter_segment(self, **kwargs):
        """
        Runs a simulation (orthographic input) where the stimulus is processed letter-by-letter. The phonological position
        is calculated through an statistical approximation between number of letters/phonemes in a word.
        The attention distribution is not a Gaussian but a Dirac on the good position.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'letter'
        self.model.phono.attention.att_phono_auto = False
        self.model.phono.attention.segment_reading = self.model.ortho.attention.segment_reading = True
        self.run_simu_H(**kwargs)

    def run_simu_graphemic_continuous(self, **kwargs):
        """
        Runs a simulation (orthographic input) where the stimulus is processed grapheme-by-grapheme. The phonological position
        is automatically calculated through the graphemic segmentation.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'grapheme'
        self.model.phono.attention.att_phono_auto = True
        self.model.phono.attention.segment_reading = False
        self.run_simu_H(**kwargs)

    def run_simu_graphemic_segment(self, **kwargs):
        """
        Runs a simulation (orthographic input) where the stimulus is processed grapheme-by-grapheme. The phonological position
        is automatically calculated through the graphemic segmentation.
        The attention distribution is not a Gaussian but a Dirac on the good position.
        :param kwargs: optional arguments for the run_simu_H function
        """
        self.serial_reading = 'grapheme'
        self.model.phono.attention.att_phono_auto = True
        self.model.phono.attention.segment_reading = self.model.ortho.attention.segment_reading = True
        self.run_simu_H(**kwargs)

    def run_simu_choose_word(self, n_choose_word=10, **kwargs):
        """
        Runs a simulation (orthographic input) where only a certain number of words (n_choose_word) are used for decoding,
        the words being externally selected by similarity with the stimulus.
        """
        self.model.ortho.lexical.build_all_repr()
        self.model.phono.lexical.build_all_repr()
        self.model.ortho.lexical.set_repr()
        self.model.phono.lexical.set_repr()
        p = utl.create_repr(np.array([[self.model.ortho.chars.index(i) for i in self.simu_stim]]), self.model.ortho.n,
                            self.model.ortho.eps)[0]
        sim = np.prod(utl.wsim(self.model.ortho.repr[:self.model.shift_begin], p), axis=1)
        idx = sim.argsort()[::-1][:n_choose_word]
        logging.simu(f"used words for decoding {self.model.ortho.get_names(idx)}")
        self.model.ortho.lexical.df.loc[~self.model.ortho.lexical.df.idx.isin(idx), 'ortho'] = False
        self.model.ortho.lexical.df.loc[self.model.ortho.lexical.df.idx.isin(idx), 'ortho'] = True
        self.model.ortho.all_repr[len(self.simu_stim) - 1] = self.model.ortho.build_repr(len(self.simu_stim))
        self.model.ortho.repr = self.model.ortho.all_repr[len(self.simu_stim)]
        self.run_simu_H(**kwargs)

    @generic_simu
    def run_simu_spelling_normal(self, thr_fix, phoneme_duration=75, loop_auditory=False, loop_phonology=True):
        length = len(self.simu_stim)
        stim = []
        for i in range(length + 1):
            tmp_stim = np.full(length, "~", dtype=str)
            if i < length:
                tmp_stim[i] = self.simu_stim[i]
            stim.append("".join(tmp_stim))
        loop = False
        end_reached = False  # Nouveau flag

        pos = 0
        if self.model.input_type == "auditory":
            while self.t_tot < self.max_iter:
                if self.t_tot % phoneme_duration == 0:
                    if not end_reached:
                        if pos < len(stim):
                            if pos < (len(stim) - 1):
                                if not loop:
                                    self.model.phono.stim = stim[pos]
                                self.model.phono.pos = pos
                                if self.model.ortho.enabled:
                                    self.model.ortho.attention.calculate_attention_parameters()
                                pos += 1
                                if pos == (len(stim) - 1) and not loop_auditory and not loop_phonology:
                                    reset = True
                            else:
                                if loop_auditory or loop_phonology:
                                    if loop_auditory:
                                        self.model.phono.stim = stim[0]
                                    else:
                                        self.model.phono.stim = stim[-1]
                                        loop = True
                                    pos = 0
                                    self.model.phono.pos = 0
                                    if self.model.ortho.enabled:
                                        self.model.ortho.attention.calculate_attention_parameters()
                                    pos += 1
                                else:
                                    if reset is True:
                                        self.model.phono.stim = stim[-1]
                                        reset = False
                                        end_reached = True  # On arrête les maj de fixation
                                        self.model.phono.pos = len(stim) - 1
                        if self.t_tot != 0:
                            self.update_fix_info()
                    else:
                        # On ne met plus à jour les fixations, on garde juste la position finale
                        self.model.phono.pos = len(stim) - 1
                        self.model.phono.stim = stim[-1]
                self.one_step_general()
            self.model.phono.stim = self.simu_stim
        else:
            raise ValueError("Invalid input type for spelling simulation, you should run simu_normal instead")

    @generic_simu
    def run_simu_spelling_H(self, thr_fix, phoneme_duration=75,loop_auditory=False,loop_phonology=True):
        """
        Runs a simulation where the input is auditory for 'maxiter' iterations.
        """
        # pdb.set_trace()
        length = len(self.simu_stim)
        stim = []
        for i in range(length + 1):
            tmp_stim = np.full(length, "~", dtype=str)
            if i < length:
                tmp_stim[i] = self.simu_stim[i]
            stim.append("".join(tmp_stim))
        loop = False

        self.model.ortho.percept.H = self.model.ortho.percept.get_entropy()
        self.model.phono.percept.H = self.model.phono.percept.get_entropy()
        Hinit = copy.copy(self.model.phono.percept.H)

        pos = 0
        if self.model.input_type == "auditory":
            while self.t_tot < self.max_iter:
                # print(self.t_tot)
                if self.t_tot % phoneme_duration == 0:
                    if pos < (len(stim)):
                        if pos < (len(stim) - 1):
                            # print("parcours de l'input")
                            if not loop:
                                self.model.phono.stim = stim[pos]
                            self.model.phono.pos = pos
                            if self.model.ortho.enabled:
                                self.model.ortho.attention.calculate_attention_parameters()
                                # self.update_fix_info()
                            pos += 1
                        else:
                            if loop_auditory or loop_phonology:
                                # print("end of loop, back to beginning")
                                if loop_auditory:
                                    self.model.phono.stim = stim[0]
                                else:
                                    self.model.phono.stim = stim[-1]
                                    loop = True
                                pos = 0
                                self.model.phono.pos = 0
                                if self.model.ortho.enabled:
                                    self.model.ortho.attention.calculate_attention_parameters()
                                pos += 1
                            else:
                                # print("on reste à la fin")
                                self.model.phono.stim = stim[-1]
                                self.model.phono.pos = pos - 1
                    # print(self.model.phono.stim)
                    if self.t_tot != 0: self.update_fix_info()
                self.one_step_general()
                criterion = self.stopCriterion()
                if criterion or self.t_tot == self.max_iter:
                    self.model.phono.stim = self.simu_stim
                    return
            pdb.set_trace()
            self.model.phono.stim = self.simu_stim
        else:
            raise "Invalid input type for spelling simulation, you should run simu_H instead"
