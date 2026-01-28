import argparse
import logging
import time
import timeit
from copy import copy

from braidpy.simu import simu
from braidpy.GUI.gui import gui
import braidpy.utilities as utl
import pdb
import os
import pickle as pkl
from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.document import Document
from bokeh.io import curdoc,show
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import numpy as np
import cProfile
import pstats
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int, help='Number of simulations')
    parser.add_argument('-s', default=None, type=str, help='String to identify')
    parser.add_argument('-m', default=None, type=int, help='Max number of iterations')
    parser.add_argument('-l', default=None, type=str, help='language : en or fr')
    parser.add_argument('--lexicon', default=None, type=str, help='lexicon name')
    parser.add_argument('-f', default=None, type=str, help='fixed frequency of the stimulus')
    parser.add_argument('-t', default=None, type=str, help='Type of simulation')
    parser.add_argument('--fMin', default=None, type=float, help='min frequency to accept word in the lexicon')
    parser.add_argument('--maxItem', default=None, type=int, help='max word number in the lexicon')
    parser.add_argument('--maxItemLen', default=None, type=int, help='max word number per length in the lexicon')
    parser.add_argument('--fMinPhono', default=None, type=float, help='min frequency to accept word in the phonological lexicon')
    parser.add_argument('--fMinOrtho', default=None, type=float, help='min frequency to accept word in the orthographic lexicon')
    parser.add_argument('-v', default=None, type=str, help='Version of the lexical decision')
    parser.add_argument('-p', default=None, type=int, help='position of eye/attention')
    parser.add_argument('-g', action='store_true', help = 'graphic interface')
    parser.add_argument('--Qa', default=None, type=float, help='Value of Qa parameter')
    parser.add_argument('--sdA', default=None, type=float, help='Value of sdA parameter')
    parser.add_argument('--sdM', default=None, type=float, help='Value of sdM parameter')
    parser.add_argument('--thr_expo', default=None, type=float, help='exposition threshold during visual exploration')
    parser.add_argument('--stop', default=None, type=str, help='stop criterion for the simulation')
    parser.add_argument('--thr_fix', default=None, type=float, help='fixation threshold during visual exploration')
    parser.add_argument('--alpha', default=None, type=float, help='motor cost during visual exploration')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--TD', default=None, type=bool, help='orthographic top down retroaction')
    parser.add_argument('--TDPhono', default=None, type=bool, help='phonological top down retroaction')
    parser.add_argument('--phono', default=True,type=bool)
    parser.add_argument('--noContext', action='store_false', help = 'desactivation of context')
    parser.add_argument('--level', default=None, type=str, help='log level (info, debug)')
    parser.add_argument('--time', action='store_true',default=None, help = 'profiles time to execute')
    parser.add_argument('--fix_times', default=None, type=str, help='if simu_type=="change_pos", gives the times of fixations ex "0 300"')
    parser.add_argument('--fix_pos', default=None, type=str, help='if simu_type=="change_pos", gives the positions of fixations ex "0 2"')
    parser.add_argument('--remove_stim_ortho', action='store_true',default=None,help='remove the stimulus ortho from the lexicon')
    parser.add_argument('--remove_stim_phono', action='store_true',default=None,help='remove the stimulus phono from the lexicon')
    parser.add_argument('--remove_neighbors', action='store_true',default=None,help='remove the stim neighbors (ortho and phono) from the lexicon')
    parser.add_argument('--remove_lemma', action='store_true',default=None,help='remove the words that share lemma with the stim from the lexicon')
    parser.add_argument('--build_prototype', action='store_true',default=None,help='used if you want to build the prototype from the data file')
    parser.add_argument('--shift', action='store_true',default=None,help='comparison with left shift and right with length +1/-1')
    parser.add_argument('--explicit', action='store_true',default=None,help='explicit learning : choose to create/update at the right time, with right word updated')
    parser.add_argument('--semi-explicit', action='store_true',default=None,help='explicit learning : choose to create/update at the right time, with most probable word updated')
    parser.add_argument('--mixture_knowledge', action='store_true', help='uses a mixture of ortho representations : expert, child and unknown')
    parser.add_argument('--n_choose_word', default=None, type=int, help='Number of nearest words chosen')

    args = parser.parse_args()


    def get_param(dico,liste):
        return dict(**{value: getattr(args, key) for key, value in dico.items() if getattr(args, key) is not None},
                    **{value :getattr(args,value) for value in liste if getattr(args, value) is not None})

    simu_param = get_param({'m':'max_iter','t':'simu_type','stop':'stop_criterion_type'},
                            ['level','build_prototype','thr_expo'])
    simu_args = get_param({}, ['thr_fix','alpha','n_choose_word'])
    model_param = get_param({'lexicon':'lexicon_name','v':'version', 'l':'langue',},
                            ['shift'])#,'force_app','force_update','force_word'])
    ortho_param = get_param({'s':'stim','Qa':'Q','sdA':'sd','TD':'top_down','remove_stim_ortho':'remove_stim','explicit':'force_word','fMinOrtho':'fMin'},
                            ['sdM','fMinOrtho','remove_lemma','remove_neighbors'])
    phono_param = get_param({'TDPhono':'top_down','remove_stim_phono':'remove_stim','phono':'enabled','fMinPhono':'fMin'},
                            ['fMinPhono','remove_lemma','remove_neighbors'])
    semantic_param = get_param({'noContext':'context_sem'}, [])
    model_param['path']='F:/LPNC/_BraidAcq/'
    sim=simu(model_param, ortho_param, phono_param,semantic_param,simu_args, **simu_param)
    if sim.simu_type== "change_pos":
        times=[int(i) for i in args.fix_times.split()]
        pos = [int(i) for i in args.fix_pos.split()]
        sim.fix={"t":times,"pos":pos}
    if sim.simu_type== "segm":
        sim.stim_o =  args.stim_o
        sim.stim_p = args.stim_p
    if args.p is not None:
        sim.model.pos=args.p
    if args.f is not None:
        sim.model.change_freq(args.f,sim.stim)
    if args.g:
        def modify_doc(doc):
            GUI = gui(sim)
            for att in dir(doc):
                try:
                    setattr(doc,att,getattr(GUI.curdoc,att))
                except:
                    pass
        # Créez une instance de l'application Bokeh en utilisant la fonction de rappel
        io_loop = IOLoop.current()
        # Créez une instance du serveur Bokeh avec l'application Bokeh
        server = Server(applications={'/': Application(FunctionHandler(modify_doc))}, io_loop=io_loop, port=5001)
        server.start()
        server.show('/')
        io_loop.start()
    else:
        if args.time:
            pr=cProfile.Profile(); pr.enable()
            cProfile.runctx("for i in range(args.n): print('n=',i); sim.run_simu_general()",{'sim':sim,'args':args},{},'tmp.txt')
            print(pstats.Stats('tmp.txt').strip_dirs().sort_stats('time').print_stats(50))
        else:
            for i in range(args.n):
                print('n=', i)
                if args.explicit or args.semi_explicit :
                    if i==0 :
                        sim.model.ortho.force_app=True
                    else:
                        sim.model.ortho.force_app=False
                        sim.model.ortho.force_update = True
                sim.run_simu_general()
    if args.debug :
        pdb.set_trace()
