import pdb
import numpy as np
import braidpy.utilities as utl
import logging
from random import randint,seed

class semantic:
    """
    The semantic class is an inner class of the modality class and allows to build the semantic distribution.
    Attributes:
    var_int (int): An integer.
    var_str (str): A string.
    """
    def __init__(self,model, context_sem=False,top_down=False, context_identification=True, N_sem=5, p_sem=10):
        """
         :param context_sem : boolean. if True, semantic context is used during and at the end of the simulation
         :param top_down : boolean. if True, there is "top-down support" from semantic to the percept (online pronunciation correction)
         :param context_identification : boolean. If True, contextual identification is enabled during learning.
         :param N_sem : int. Size of the context
         :param p_sem : float. Strength of the context
        """
        self.model=model
        self.context_sem = context_sem
        self.top_down=top_down
        self.context_identification = context_identification
        self.N_sem = N_sem
        self.p_sem = p_sem
        self.context_sem_words=None
        self.context_sem_words_phono=None
        self.dist={}
        self.seed = 1


    def build_context(self):
        """
        builds the semantic distribution :
        if there is no context, there is no word included in the context and the semantic distribution is uniform
        if there is some context :
        if the word is known in one of the 2 modalities, it's hypothised to be known semantically, it's included in the context. Rest of words are selected randomly.
        if the word is neither known orthographically and phonologically, it's hypothised to be not known semantically, it's not included in the context
        """
        self.seed+=1
        seed(self.seed)
        lex = self.model.ortho.lexical
        lex_p = self.model.phono.lexical
        we=lex.get_word_entries(check_store=True)
        we_p=lex_p.get_word_entries(check_store=True)
        # context + stim known in minimum one modality -> stim + random words selected
        stim_idx = list(set(list(we.idx.values if we is not None else [])+list(we_p.idx.values if we_p is not None else [])))
        n = len(lex.df)
        if self.context_sem and len(stim_idx)>0:
            self.idx_sem = stim_idx + [randint(0, n-1) for _ in range(self.N_sem-len(stim_idx))] if n>0 else []
        # no context = no word in the context
        elif not self.context_sem :
            self.idx_sem = []
        # context + stim unknown -> random words selected
        else:
            self.idx_sem = [randint(0, n-1) for _ in range(self.N_sem)] if n>0 else []
        self.dist['sem'] = utl.norm1D([self.p_sem if i in self.idx_sem else 1 for i in range(n)])
        self.context_sem_words=[lex.get_name(i) for i in self.idx_sem]
        self.context_sem_words_phono=[lex_p.get_name(i) for i in self.idx_sem]
        logging.simu(f"context words : {self.context_sem_words}")



# existence incertaine! Je les laisse pour l'instant mais bon ..
#class semanticOrtho(semantic):
#    def __init__(self, modality, **modality_args):
#        super().__init__(modality=modality,**modality_args)
#
#class semanticPhono(semantic):
#    def __init__(self, modality, **modality_args):
#        super().__init__(modality=modality,**modality_args)
#        logging.simu(f"context words : {self.context_sem_words}")

