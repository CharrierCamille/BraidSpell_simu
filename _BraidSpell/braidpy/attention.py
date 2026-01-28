import logging
import math
import pdb

import braidpy.utilities as utl
import braidpy.lexicon as lex
import numpy as np


class attention:
    """
    The attention class is an inner class of the modality class and represents the attentional submodel, either in the orthographic or phonological modality.

    """

    def __init__(self, modality, Q, mean, sd, sdM=1000, segment_reading=False, reading_unit="None"):
        """
        Attention class constructor.

         :param mean : float, position of the attentional focus
         :param sd : float, attentional dispersion
         :param sdM : float, max value of the attentional dispersion
         :param segment_reading: boolean. If set to True, the attention distribution is not Gaussian but uniform on 1 or several letters corresponding to one phoneme.
        """
        self.coupling_a = None
        self.coupling_b = None
        self.modality = modality
        self.Q = Q
        self.mean = mean
        self.sd = sd
        self.sdM = sdM
        self.dist = None
        self.segment_reading = segment_reading
        self.reading_unit = reading_unit

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        """
        Sets attention position, starts at 0. Position should be set at -1 at the end of a simulation

        :param value: int, the position to be set.
        """
        if value < -1 or ('lexical' in self.__dict__ and self.modality.N is not None and value >= self.modality.N):
            logging.warning(f"bad mean position is trying to be set : {value}")
        self._mean = value

    @property
    def sd(self):
        return self._sd

    @sd.setter
    def sd(self, value):
        # print(f"sd {self.modality.mod} set to {value}")
        if value < 0:
            logging.warning("You're trying to set a negative value of sdA")
        sdM = self.__getattribute__("sdM") if hasattr(self, 'sdM') and self.sdM is not None else 10000
        self._sd = min(value, sdM)

    @property
    def sdM(self):
        return self._sdM

    @sdM.setter
    def sdM(self, value):
        if value < 0:
            logging.warning("You're trying to set a negative value of sdM")
        self._sdM = value
        self.__setattr__('sd', self.sd)

    @utl.abstractmethod
    def build_attention_distribution(self):
        """
        Builds the attention distribution according to the attentional position mean, the standard deviation sd,
        the attentional Quantity Q and the length of the stimulus N.
        """
        pass

    def set_regression(self):
        """
        Performs linear regression to find the relationship between orthographic and phonological length.
        Use for attention coupling (matching orthographic positions to phonological positions)
        """
        from sklearn import linear_model
        other_mod = self.modality.model.ortho if self.modality.mod == "phono" else self.modality.model.phono
        x = other_mod.lexical.df.len.values.reshape(-1,
                                                    1) if self.modality.mod == "phono" else other_mod.lexical.df.phlen.values.reshape(
            -1, 1)
        y = self.modality.lexical.df.len.values.reshape(-1,
                                                        1) if self.modality.mod == "ortho" else self.modality.lexical.df.phlen.values.reshape(
            -1, 1)
        reg = linear_model.LinearRegression()
        res = reg.fit(x, y)
        print(f"mod : {self.modality.mod} coupling_a : {res.coef_[0][0]}, coupling_b : {res.intercept_[0]}")
        self.coupling_a, self.coupling_b = res.coef_[0][0], res.intercept_[0]
        self.ratio_att=1/np.mean(x/y)
        print(f"mod : {self.modality.mod} ratio : {self.ratio_att}")

    def len_phlen_relation(self, x, rnd=1):
        """
        Calculates the predicted length as a function of the other modality length according to the linear regression.

        :param x: the other modality length
        :return: the predicted this modality length
        """
        # pdb.set_trace()
        # return utl.len_phlen_relation(x, self.coupling_a, self.coupling_b, rnd)
        return round(x*self.ratio_att/rnd)*rnd


class attentionOrtho(attention):
    def __init__(self, modality, Q=1, mean=-1, sd=1.75, sdM=1000, **modality_args):
        # if modality.model.input_type == "auditory":
        #     sd = 2
        # else :
        #     sd = 1

        super().__init__(modality=modality, Q=Q, mean=mean, sd=sd, **modality_args)
        self.sdM = sdM

    def init_pos_auto(self):
        """
        Automatically sets the visual attention position (and also the gaze position) at the beginning of the simulation
        """
        # if self.modality.model.input_type == "visual":
        #     if self.Q > 0.7 and self.sd > 1:
        #         self.modality.pos = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4][self.modality.N - 1]
        #     else:
        #         self.modality.pos = 0
        # else:
        #     self.modality.pos = 0

        self.modality.pos = 0

        # print(f"init {self.modality.mod} position : {self.modality.pos}")

    def build_attention_distribution(self):

        """
        Calculates the attention parameters and builds the attention distribution according to it.
        """

        if self.mean >= 0:
            if self.segment_reading:
                gs = self.modality.model.gs
                len_grapheme = gs.count(gs[self.mean])
                new_mean = gs.index(gs[self.mean])
                tmp = np.array(
                    [1 / len_grapheme if 0 <= i - new_mean < len_grapheme else 0 for i in range(self.modality.N)])
            else:
                tmp = utl.gaussian(self.mean, self.sd, self.modality.N_max)
            tmp = self.Q * tmp
            tmp[tmp > 1] = 1
            self.dist = tmp

    def calculate_attention_parameters(self, end_verif=False):
        """
        Chooses the next orthographic position according to statistical properties of the word or according to the graphemic segmentation (if enabled)
        """
        self.modality.pos = round(self.len_phlen_relation(self.modality.model.phono.pos)) if self.modality.model.phono.pos > 0 and self.coupling_a is not None else 0
        # if end_verif:
        # print(self.modality.pos)

        if self.modality.model.input_type == "auditory":
            self.sd = self.modality.model.phono.attention.sd
            self.sd = self.modality.model.phono.attention.sd
            self.sd = self.len_phlen_relation(self.modality.model.phono.attention.sd, rnd=0.25)
            # pass


class attentionPhono(attention):
    def __init__(self, modality, Q=1, mean=-1, sd=1.75, att_phono_auto=False, segment_reading=False, **modality_args):
        """
        :param att_phono_auto: boolean. if True, automatically sets the phonological attention according to the graphemic segmentation
        """
        # if modality.model.input_type == "auditory":
        #     sd = 2
        # else:
        #     sd = 1

        super().__init__(modality=modality, Q=Q, mean=mean, sd=sd, segment_reading=segment_reading, **modality_args)
        self.att_phono_auto = att_phono_auto

    def calculate_attention_parameters(self, end_verif=False):
        """
        Chooses the next phonological position according to statistical properties of the word or according to the graphemic segmentation (if enabled)
        """
        # pdb.set_trace()
        if self.att_phono_auto and self.modality.model.ortho.stim in self.modality.model.df_graphemic_segmentation.index:
            self.modality.pos = int(self.modality.model.gs[self.modality.model.ortho.pos])
        elif self.att_phono_auto:
            logging.simu("ATTENTION : segmentation graphémique absente!")
            pdb.set_trace()
        else:
            self.modality.pos = round(self.len_phlen_relation(
                self.modality.model.ortho.pos)) if self.modality.model.ortho.pos > 0 and self.coupling_a is not None else 0
        if self.modality.model.input_type == "visual":
            self.sd = self.modality.model.ortho.attention.sd
            self.sd = self.len_phlen_relation(self.modality.model.ortho.attention.sd, rnd=0.25)
            # pass

    def build_attention_distribution(self):
        """
        Calculates the attention parameters and builds the attention distribution according to it.
        """
        if self.mean >= 0:
            tmp = utl.gaussian(self.mean, self.sd, self.modality.N_max)
            tmp = self.Q * tmp
            tmp[tmp > 1] = 1
            self.dist = tmp
