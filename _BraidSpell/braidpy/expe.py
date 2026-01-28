# -*- coding: utf-8 -*-
# General purpose libraries
import copy
# Scientific/Numerical computing

import itertools
# BRAID utlities
from braidpy.simu import simu
import logging
import os
import pdb
import pickle as pkl
import random
import gc
from time import time

import numpy as np
import pandas as pd

# on désactive le mode debug de numba et numpy
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
numpy_logger = logging.getLogger('numpy')
numpy_logger.setLevel(logging.WARNING)


class expe:
    """
    Experiment involving simulations from several stimuli, exposures, parameters
    """

    def __init__(self, simu=None, res_fct_name=None, path="./", basename="simu", test={},
                 n_expo=1, reinit=True, auto_lenphlen=True,
                  print_res=True, store_txt=False, store_simu=False):
        """
        :param simu: An instance of the simulation class
        :param res_fct_name: A string or a list of strings representing the name(s) of the function(s) that will be used to compute the results of the simulation.
            All the possibilities are listed in the function simu.one_res of class simu
        :param path: The root path.
        :param basename: The basename for the simulation files that will be saved, it will be completed according to simulation parameters (orthographically/phonologically novel ?)
        :param test: A dictionary containing the parameters that will be tested (keys), and the values that will be tested for each parameters (values). The cross-product of each combination of parameters will be tested.
            ex test={"Q": [1,2], "max_iter" :[500,1000]}
        :param n_expo: The number of exposures for each word, defaults to 1 (optional)
                           /!\ in the stored dataframe, t can be either a number of iterations or a number of exposures
                           but in the simulation it's possible to vary both the number of exposures and store one result per iteration simulated
                           the corresponding columns in the dataframe (t,num) will be affected according to the value of self.n_expo and length of the result
                           If self.n_expo>1, t in exposures, if you want a result in iterations it will be stored as num
                           If self.n_expo == 1 and len(res) > 20, we have a long result -> it is a number of iterations, every element of res will be stored with varying t values
                           If self.n_expo == 1 and we have len(res) < 21, it's a list of non - temporal data, every element of res will be stored with varying num values
        :param reinit: A boolean parameter that determines whether to reinitialize the lexicon at each simulation or not.
        :param print_res: A boolean parameter that determines whether or not to print the results of the simulation.
        :param store_txt: A boolean parameter that determines whether or not to store results in a readable form in a txt file.
        :param store_simu: A boolean parameter that determines whether to store the simulation object or not.
        """
        self.simu = simu
        self.res_fct_name = res_fct_name if isinstance(res_fct_name, list) else [res_fct_name]
        self.path = path
        self.basename = basename
        self.test = test
        self.n_expo = n_expo
        self.reinit = reinit
        self.print_res = print_res
        self.store_txt = store_txt
        self.store_simu = store_simu
        self.csv_name = self.txt_name = None
        self.copy_model = None
        self.dico = self.already_tested_words = None
        self.succ = True
        self.auto_lenphlen = auto_lenphlen

    def update(self, **kwargs):
        """ used to set several attributes at once"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    ########################
    ### Beginning of expe ##
    ########################

    def set_filename(self):
        """
        Sets the filenames for saving simulation data in pickle and csv formats.
        """
        # if not os.path.exists('pkl'):
        #    os.mkdir('pkl')
        if not os.path.exists('csv'):
            os.mkdir('csv')
        o = self.simu.model.ortho.lexical.remove_stim
        p = self.simu.model.phono.lexical.remove_stim
        name = self.basename + '_PM_' + ('X' if not (o or p) else 'O' * o + 'P' * p)
        self.csv_name = 'csv/' + name + '.csv'
        self.txt_name = 'txt/' + name + '.txt'
        logging.expe(f"{self.liste} \n {self.csv_name}")

    def load_existing_data(self):
        """
        Loads preliminary results if part of the simulation was already conducted.
        """
        self.initialize_data()
        try:
            df = pd.read_csv(self.csv_name)[self.dico.keys()]
            self.already_tested_words = list(set(df.word))
        except:
            self.already_tested_words = []
            if self.store_txt:
                file1 = open(self.txt_name, "w");
                file1.writelines("Beginning of simulation \n ");
                file1.close()
            pd.DataFrame.from_dict(self.dico).to_csv(self.csv_name, mode='w', index=False)

    def initialize_data(self):
        """
        This function initializes the dictionary that will contain the results.
        """
        self.dico = dict(**{'num': [], 't': [], 'word': [], 'value': [], 'success': [], 'error_type': []},
                         **{key: [] for key in self.test})

    def begin_expe(self):
        """
        Sets up various parameters and variables before the simulation begins.
        """
        print("begin expe")
        self.set_filename()
        self.load_existing_data()
        self.param_product = [dict(zip(self.test, x)) for x in itertools.product(*self.test.values())]
        self.simu.model.ortho.lexical.build_all_repr()
        self.simu.model.phono.lexical.build_all_repr()
        try:
            self.copy_model = copy.deepcopy(self.simu.model) if self.reinit else None
        except:
            pdb.set_trace()

    ##############################################################
    ### Result function (used by compare_param end RealSimulation)
    ##############################################################

    def res_fct(self):
        """
        This function concatenates the results of different function names and returns them.
        :return: a list of results obtained by calling the `one_res` method of the `simu` object for each name in the
        `res_fct_name` list. If there is only one result and it is a list with more than one element, it returns the list. If there is more than one
        result and they are not strings, it concatenates them.
        """
        res = [self.simu.one_res(i) for i in self.res_fct_name]
        # if len(self.res_fct_name)==1 and isinstance(res[0],list):
        #    res=res[0]
        if len(self.res_fct_name) > 1:
            if len(res) == 2 and not isinstance(res[0], list) and isinstance(res[1], list) and len(res[1]) > 1:
                return np.concatenate(([res[0]], res[1]))
            else:
                try:
                    return np.concatenate((res[0], res[1:]))
                except:
                    return res
        if len(self.res_fct_name)==1 and isinstance(res[0],np.ndarray):
            return res[0]
        return np.array(res)

    ####################
    ### Data storing ###
    ####################

    def store_res(self):
        """
        Stores the results in a pkl file
        """
        # pdb.set_trace()
        try:
            df = pd.DataFrame.from_dict(self.dico)
        except:
            logging.error("Erreur dans la création du dataframe")
            print(self.dico)
            pdb.set_trace()
        if self.store_simu:
            cp = copy.deepcopy(self.simu);
            # removes heavy part of the object before storing it
            cp.ortho.all_repr = {}
            cp.ortho.repr = []
            if self.simu.model.phono.enabled:
                cp.phono.all_repr = {}
            cp.phono.repr = []
            pkl.dump(cp, open(self.path + self.csv_name, 'wb'))
        df.to_csv(self.csv_name, mode='a', header=False, index=False)

    ####################
    ### Big simulations ###
    ####################

    def compare_param(self):
        """ Generic simulation to compare different values of parameters :
            for example max_iter, Q, leak ...
            it sets automatically the value of the parameter, given the name of the parameter (ex "Q","max_iter")
            and its possible values (ex [1,2],[250,500]) and test all combinations of parameters ex [1,2]x[250,500] -> 4
            /!\ In these simulations, all words are tested independantly

            If you want to run this kind of simulation with a new parameter, you have to :
                1/ choose/define a new name in the simu.one_res function to define how to get the result you're interested in.
                2/ define how you set the parameter in the set_attr function from class simu (if it's not automatic)
        """
        self.begin_expe()
        for iw, word in enumerate(self.liste):
            if not word in self.already_tested_words:
                logging.expe(f"{word} {iw}/{len(self.liste)}")
                self.initialize_data()
                for ip, indices in enumerate(self.param_product):
                    if self.reinit:
                        self.simu.model = copy.deepcopy(self.copy_model)
                    if self.simu.model.input_type == 'visual':
                        self.simu.model.ortho.stim = word
                        if self.auto_lenphlen:
                            setattr(self.simu.model.ortho, "lenMax", len(word))
                            setattr(self.simu.model.phono, "lenMax", len(word))
                    elif self.simu.model.input_type == 'auditory':
                        self.simu.model.phono.stim = word
                        if self.auto_lenphlen:
                            setattr(self.simu.model.ortho, "phlenMax", len(word))
                            setattr(self.simu.model.phono, "phlenMax", len(word))
                    else:
                        raise Exception("Invalid input_type. Should be either 'visual' or 'auditory'. No other type "
                                        "implemented.")
                    if self.auto_lenphlen:
                        self.simu.model.reset_model(self.simu.reset)

                    self.simu.reset_n()
                    for p, val in indices.items():
                        try:
                            setattr(self.simu.model.ortho, p, val)
                        except:
                            pass
                        try:
                            setattr(self.simu.model.phono, p, val)
                        except:
                            pass
                        try:
                            setattr(self.simu, p, val)
                        except:
                            pass
                    for t in range(self.n_expo):
                        self.simu.run_simu_general()
                        res = self.res_fct()
                        succ = [self.simu.success(r) for r in self.res_fct_name]
                        if isinstance(succ[0], list) and len(succ[0]) > 1:
                            succ = np.concatenate((succ[0], succ[1:])) if len(succ) > 1 else succ[0]
                        if self.print_res:
                            print(word, indices, [round(i, 4) if isinstance(i, float) else i for i in res])
                        if self.store_txt:
                            summary = " ".join(["\n", word, str(indices),
                                                str([round(i, 4) if isinstance(i, float) else i for i in res])])
                            file1 = open(self.txt_name, "a");
                            file1.write(summary);
                            file1.close()
                        self.simu.increase_n()
                        for ir, r in enumerate(res):  # 2D array with time as second dimension
                            if isinstance(r, np.ndarray):
                                for iit, it in enumerate(r):
                                    app = dict(**{'word': word, 't': iit, 'num': ir, 'value': it, 'success': succ[ir],
                                                  'error_type': self.simu.error_type}, **indices)
                                    for k, v in app.items():
                                        try:
                                            self.dico[k].append(v)
                                        except:
                                            pdb.set_trace()
                            else:
                                # /!\ t can be either a number of iterations or a number of exposures
                                # but in the simulation it's possible to vary both the number of exposures and store one result per iteration simulated
                                # If self.n_expo>1, time in exposures, if you want a result in iterations it will be stored as num
                                # If self.n_expo == 1 and len(res) > 20, we have a long result -> it is a number of iterations(time=ir, num=0)
                                # If self.n_expo == 1 and we have len(res) < 21, it's a list of non - temporal data(num=ir, time=t)
                                time = ir if (self.n_expo == 1 and len(res) > 20) else t
                                num = 0 if (self.n_expo == 1 and len(res) > 20) else ir
                                try:
                                    app = dict(**{'word': word, 't': time, 'num': num, 'value': r, 'success': succ[ir],
                                                  'error_type': self.simu.error_type}, **indices)
                                except:
                                    pdb.set_trace()
                                for k, v in app.items():
                                    self.dico[k].append(v)
            self.store_res()
            gc.collect()
