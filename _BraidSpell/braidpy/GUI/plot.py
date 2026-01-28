# /usr/bin/env python3

# -*-coding:utf-8-*-
import sys

sys.path.append("../")
braidPath = "../../braidpy/"
sys.path.append(braidPath)
from time import time
from braidpy.lexicon import *
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from numpy import unique
import math
import itertools
from matplotlib.ticker import FormatStrFormatter
from Levenshtein import distance
from functools import partial
import braidpy.utilities as utl

def update(func):
    def wrapper(self, *arg, **kwargs):
        kw={}
        for attr, value in kwargs.items():
            if hasattr(self,attr):
                setattr(self, attr, value)
            else :
                kw[attr]=value
        res = func(self, *arg, **kw)
        return res
    return wrapper

class plot:
    def __init__(self, filename="", PM=False, select_num=None, num_remove=None, fct_word=None, arg_word=None,
                 fct_time=None, arg_time=None, alpha=0.1, title="", labelx="", labely="",
                 lexicon_name="celex.csv", norm_freq=False, nMax=None, nMaxLen=None,
                 remove_param=None, select_param=None, remove_errors=False, per_letter=False, save_csv=False,
                 show=True, plot_med=True, plot_color_bar=True,value='value',logx=False,logy = False):
        [self.path, self.filename, self.save_csv, self.PM] = ['pkl/', filename, save_csv, PM]
        [self.title, self.subtitle, self.labelx, self.labely] = [title, "", labelx, labely]
        [self.width, self.height, self.bar_width, self.step] = [8, 8, 0.5, None]
        [self.xlim, self.ylim] = [None, None]
        # on ne conserve dans le dataframe que nMax stimuli, ou nMaxLen stimuli par classe de longueur
        [self.nMax, self.nMaxLen, self.tmax] = [nMax, nMaxLen, None]
        self.alpha = alpha
        # condition sur les mots à sélectionner pour le plot
        [self.condition, self.arg1, self.arg2] = [None, None, None]
        # colonne dont on va regarder les résultats, peut aussi regarder 'success' par exemple
        self.value=value
        # pour faire des plots en changeant l'axe
        self.logx,self.logy=logx,logy
        # fonctions à appliquer pour traiter les données
        # df a commme colonnes : (une colonne par paramètre testé) + word + t + num
        # word : mot testé
        # t : itération/exposition selon la simulation
        # num : on peut enregistrer n données (ex les fixations pour un mot) indicées par num

        # A/ Traitement sur les données à un pas de temps donné : [nbFix, durée] ou [l1,l2,l3,l4] ..
        # au niveau du dataframe, cette fonction sera appliquée après un groupby('word', 't')
        # 1/ on veut prendre le i-ème élmt : si on a [nbFix,durée], on veut nbFix  -> select_num=0
        # 2/ on veut résumer la liste (moyenne) : si on a [l1,l2,l3] (mot "yes")
        # fonctions définies dans le groupby : on met un string directement
        # fct_word = "count", "first", "last", "max", "mean", "median", "min", "size", "sum", "var"
        # ex : si veut le nombre d'élements (longueur de la liste) : fct_word = 'count'
        # si on veut une autre fonction que celles définies dans cette liste :
        # cette fonction doit renvoyer une valeur unique (fonction d'aggrégation)
        # dans ce cas on ne donne pas le string mais la fonction directement
        # pas garanti : devrait fonctionner si elle est vectorielle, à tester!
        # si veut enlever les n premières valeurs (ex [sigma]+fixations -> fixations), remove_num=1

        # B/ Traitement sur la chaîne temporelle des résultats (fct_time)
        # 2 cas particuliers :
        # 'derivative' va calculer la dérivée de la chaîne temporelle
        # 'threshold' va calculer le temps pour atteindre un critère d'arret (ex value>0.95) défini par arg_time
        # sinon :
        # au niveau du dataframe, cette fonction sera appliquée après un groupby('word')
        # et après le traitement précédent
        # on a obtenu une liste indexée par le temps, on peut réappliquer un traitement
        # fonctions applicalbles à un GroupBy, à appeler directement avec un string :
        # fct_time = "count", "first", "head", "last", "max", "mean",
        # "median", "min", "size", "sum", "var, "tail"
        # ex :
        # la valeur init : fct_time = 'first'
        # la dernière valeur : fct_time = 'last'
        # les 100 premières : fct_time = 'head', arg_time = 100
        # les 100 dernières : fct_time = 'tail', arg_time = 100
        # la durée totale : fct_time = 'count'
        # les autres fonctions devraient marcher si ce sont des fonctions d'aggrégation, à tester!
        [self.fct_word, self.fct_time, self.arg_word, self.arg_time] = [fct_word, fct_time, arg_word, arg_time]
        # select_param : on ne plot qu'un nom parmi les différents noms de paramètres possibles -> moyenne autres param
        # remove_param : on enlève un nom parmi les différents noms de paramètres possibles -> moyenne autres param
        # on n'affecte pas les 2 paramètres à la fois
        [self.remove_param, self.select_param] = [remove_param, select_param]
        # remove_num : on enlève un num précis (ex [sdA,fix1,fix2,fix3] -> remove_num=0 )
        # select_num : on garde un num précis (ex [sdA,fix1,fix2,fix3] -> selectt_num=0 )
        [self.remove_num, self.select_num, self.remove_errors] = [num_remove, select_num, remove_errors]
        self.per_letter,self.show=per_letter,show
        # pbas phonotactiques
        self.dicof = extractLexicon(path="../../braidpy/", lenMin=0, lenMax=12, lexicon_name=lexicon_name,
                                    return_df=True).to_dict()['freq']
        self.df_freq = pd.DataFrame.from_dict(self.dicof, orient='index', columns=['freq']) \
            .rename_axis('word').reset_index()
        if norm_freq:  # normalise les fréquences par classe de longueur
            self.df_freq["len"] = self.df_freq.word.str.len()
            for l in set(self.df_freq.len):
                tot = self.df_freq.loc[self.df_freq.len == l, "freq"].sum()
                self.df_freq.loc[self.df_freq.len == l, "freq"] /= tot * 1000000
        [self.LetterFreq, self.BiLetterFreq] = [None, None]
        [self.df_brut, self.df] = [None, None]
        # remplissage des courbes
        [self.plot_med, self.plot_color_bar, self.axs, self.ax] = [plot_med, plot_color_bar, None, None]
        # [self.plot_med,self.plot_color_bar,self.axs,self.ax]=[False,False,None,None]
        self.set_data()
        utl.matplotlib_style()



    # on ne force pas à réinitialiser les titres et alpha mais les fct et PM oui
    @update
    def set_data(self, **kwargs):
        # a priori self.dico est directement un dataframe, car on a fait la transformation
        # mais si on crois un fichier de l'ancien format, on le transforme en df
        self.title_list = [""]; self.test_brut=None;
        data = pkl.load(open(self.path + self.filename, 'rb'))
        if len(data) == 3:
            if isinstance(data[1],list):
                [self.dico, self.value_list, self.name_list] = data
            else:
                [self.dico, self.test_brut, self.title_list] = data
        elif isinstance(data[1],list):
            [self.dico, self.value_list, self.name_list, self.title_list] = data
        else:
            [self.dico, self.test_brut, self.title_list, self.model] = data
        if not isinstance(self.dico, dict):
            self.df_brut = self.dico
        else:
            self.create_df()
        self.wnt = ['word', 'num', 't']
        # name_list/value_list/n_param_brut chg x (df_brut), param_names/values/n_param change (df)
        self.param_tuples = [i for i in self.df_brut.columns if i not in ['len'] + self.wnt]
        if self.test_brut is None: # ancien format
            self.test_brut = {key: value for key, value in zip(self.name_list, self.value_list)}
            self.param2one_column()
        else:
            self.define_param()
        # exposures : general success, first and second exposure
        # iterations : success at one iteration, no need for a success at each iteration
        #if "success" in self.df_brut.columns and 1<len(list(set(self.df_brut.t)))<20:
        #    self.df_brut["s1"] = self.df_brut.groupby(['word'] + self.param_names).transform('first')["success"]
        #    self.df_brut["s2"] = self.df_brut.groupby(['word'] + self.param_names).transform('nth', 2)["success"]
        #    self.df_brut["success"] = self.df_brut.s1 & self.df_brut.s2
        #    self.df_brut.drop(['s1', 's2'], axis=1,inplace=True)

    def __str__(self):
        df = self.df if self.df is not None else self.df_brut
        print(df.pivot_table(columns=self.param_names, index=self.wnt).droplevel(0, axis=1).reset_index())

    def df2csv(self):
        if self.save_csv:
            df=self.df
            df["len"]=df.word.str.len()
            df.to_csv(self.path+self.filename[:-3]+'csv')

    def param2one_column(self):
        self.df_brut=self.df_brut[self.param_tuples].stack(level=0).reset_index(level=1).merge(
            self.df_brut[self.wnt],left_index=True,right_index=True ).set_index('level_1')
        tpl=([self.value_list[idx][val] for idx,val in enumerate(eval(i))] for i in self.df_brut.index)
        self.df_brut.index=pd.MultiIndex.from_tuples(tpl)
        self.df_brut=self.df_brut.reset_index()
        self.df_brut.columns=self.name_list+['value']+self.wnt
        self.define_param()

    def formate_title(self, tpl, title=True):
        if isinstance(tpl, tuple):
            return ', '.join([self.param_names[i]+' = '+str(t) for i,t in enumerate(tpl)]) if title \
                else str(tpl)
        return tpl

    def define_param(self, keys=None):
        self.test = self.test_brut.copy()
        if keys is not None:
            self.test = {k: v for k, v in self.test.items() if k in keys}
        self.param_names = list(self.test.keys())
        self.param_tuples = list(itertools.product(*list(self.test.values())))


    def create_len(self):
        self.df["len"] = self.df.word.str.len()

    #### Affecting condition to plot data

    def threshold(self,df):
        # on regarde les donénes à garder, donc avant d'avoir atteint le threshold
        # on est > 0 en somme cumulée donc le threshold a été atteint au moins une fois
        # donc df['th']==True correspond à "le threshold n'a jamais été atteint", donc les données à garder
        df["th"] = df[self.value] > self.arg_time
        df['th'] = df.expanding().sum()['th'] == 0
        return df

    def set_condition(self, condition=None, arg1=0, arg2=None):
        dico = {"length": self.length, "freq": self.freq, "pba": self.pba, "neighb": self.neighb,
                "neighbSize": self.neighbSize,
                "name": self.name}
        self.condition = dico[condition]
        self.arg1 = arg1
        self.arg2 = arg2

    def remove_condition(self):
        self.condition = self.arg1 = self.arg2 = None

    def select_condition(self):
        if self.arg2 is None:
            self.df = self.df[self.df.cond == self.arg1]
        else:
            self.df = self.df[(self.df.cond <= self.arg2) & (self.df.cond >= self.arg1)]
        self.df = self.df.drop('cond', axis=1)

    def length(self):
        self.df["cond"] = self.df.word.str.len()
        self.select_condition()

    def freq(self):  # remove key when introducing df
        try:
            df = self.df.merge(self.df_freq, on='word')
            self.df["cond"] = df["freq"]
            self.select_condition()
        except:
            raise ValueError("extraction de la fréquence impossible, checkez les lexiques")

    def pba(self):
        def get_pba(key):
            return WordWithPba(key, self.LetterFreq, self.BiLetterFreq)[-1] * 100

        if self.LetterFreq is None:  # calcul des fréquences positionnelles
            [self.LetterFreq, self.BiLetterFreq] = calculatePositionalSegmentFreq(self.dicof)
        self.df["cond"] = self.df.word.apply(get_pba)  # on crée une colonne pba dans le df
        self.select_condition()

    def neighb(self):
        def dist_to_lexicon(key):
            df = pd.DataFrame({"dist": self.df.word.apply(partial(distance, key))})
            return min(df.dist[df.dist > 0])

        self.df["cond"] = self.df.word.apply(dist_to_lexicon)  # on crée une colonne dist dans le df
        self.select_condition()

    def neighbSize(self):
        def nbNeigh(key):
            df = pd.DataFrame({"dist": self.df.word.apply(partial(distance, key))})
            return df.dist[df.dist == 1].count()

        self.df["cond"] = self.df.word.apply(nbNeigh)  # on crée une colonne dist dans le df
        self.select_condition()

    def name(self):
        if self.arg1 is not None:
            self.df = self.df[self.df.word == self.arg1]

    def round(self, x, n=1):
        return round(x, n - int(math.floor(math.log10(abs(x)))) - 1) if x != 0 else 0

    # trouver le nombre d'outliers en fin de simu en ce qui concerne les valeurs de DL essentiellement
    def printOutliers(self, seuil=0.2, a=None, verbose=False):
        comp = a if a is not None else 0 if self.PM else 1
        self.selection()
        df_end = self.df[self.df.t == self.tmax];
        df = df_end.drop(columns=['len'] + self.wnt, errors='ignore')
        [countTot, count] = [df.count()[0], df[abs(df - comp) > seuil].count()]
        for i in count.index:
            param = self.formate_title(i)
            print(param, " : ", float(count[count.index == i]), "outliers / ", countTot)
            if verbose or True:
                print(df_end[abs(df_end[i] - comp) > seuil].word.unique())

    def getSuccessPhonoDecoding(self):
        self.selection()
        lx=self.model.lexicon['pron'].reset_index()
        res=self.df[['word','value']+self.param_names]
        merge=res.merge(lx,on='word')
        merge['success']=merge.value==merge.pron
        df=merge.set_index('word')[self.param_names+['success']].pivot_table(columns=self.param_names,values='success',index='word').reset_index()
        df["len"]=df.word.str.len()
        #print(df.groupby('len').mean())
        #print(df.mean())
        return df

        # sol=pd.DataFrame.from_dict(d,orient='index')
        # sol.columns=['sol']
        # res=p.df_brut[['word','value','att_factor']].set_index('word')
        # res['value']=res['value'].str.replace('#','')
        # res=res.pivot_table(columns=['att_factor'],values='value',index='word',aggfunc='first')
        # merge=res.join(sol,on='word')
        # succ=merge.copy()
        # for i in range(10):
        #    succ[i]=merge[i]==merge['sol']
        # print(merge)
        # print(succ.mean())

    ############## General fonction for plotting

    def select_subdf(self, l, df=None):
        df = df if df is not None else self.df
        # si c'est un string on obtient string sans guillemets -> il faut l'entourer de " "
        return df.query(' & '.join(['{0}=={1}'.format(k, l[i] if not isinstance(l[i],str) else '"'+l[i]+'"' )
                                    for i, k in enumerate(self.test)]))

    def select_size(self):
        l = len(self.test)
        # nb de valeurs pour chaque param = max+1 car commence à 0
        df = self.df.max()[self.param_names]
        len_list = [len(i) for i in self.test.values()]
        if len(len_list) > 1:
            d = 100
            # on fait la découpe qui va minimiser la distance entre largeur et hauteur
            for i in range(1, l):
                w = np.prod([len_list[j] for j in range(i)])
                h = np.prod([len_list[j + 1] for j in range(l - i)])
                dtmp = abs(w - h)
                if dtmp < d:
                    [self.w, self.h, self.nw, d] = [w, h, i, dtmp]

        elif len(len_list)>0:
            if len_list[0] < 5:
                self.w = 1;
                self.h = len_list[0];
                self.nw = 1
            else:
                self.w = int(np.sqrt(len_list[0]))
                self.h = math.ceil(len_list[0] / self.w)
                self.nw = 1
        else:
            self.w=1
            self.h=1
            self.nw = 1



    def find_coordinates(self, l):
        l = [v.index(l[i]) for i, (p, v) in enumerate(list(self.test.items()))]
        if len(l) > 1:
            i = 0;
            prod = 1
            for n in range(self.nw - 1, -1, -1):
                i += l[n] * prod
                prod *= len(self.test[self.param_names[n]])
            j = 0;
            prod = 1
            for n in range(self.nw, len(self.param_names)):
                j += l[n] * prod
                prod *= len(self.test[self.param_names[n]])
            return (i, j)
        i = int(l[0] / self.h)
        j = int(l[0] % self.h)
        return (i, j)

    def find_coordinates_t(self, l):
        return tuple([l // self.h, l % self.h])

    def print_success_rate(self):
        self.df_brut["len"]=self.df_brut.word.str.len()
        print(self.df_brut.groupby(['word'] + self.param_names).first().reset_index().groupby(
            ['len'] +self.param_names).mean()['success'].unstack(level=0))

    @update
    def selection(self, **kwargs):
        # self.create_df()
        self.df = self.df_brut.copy()
        self.define_param()
        if self.fct_time!='threshold':
            if self.remove_errors and 'success' in self.df:
               self.df=self.df[self.df.success==True].drop('success',axis=1)
            #else:
            #    self.df.drop('success',axis=1,inplace=True,errors='ignore')
        if self.select_num is not None:
            self.fct_word = 'nth';
            self.arg_word = self.select_num
        if self.remove_num is not None:
            # +1 pour que ça commence à 0 : remove_num= 0 pour 1 élément récupéré
            self.df = self.df.drop(self.df.groupby(['word', 't'] + self.param_names).agg('head', self.remove_num + 1).index)
        if self.fct_word is not None:
            gb = self.df.groupby(['word', 't'] + self.param_names)
            self.df = gb.agg(self.fct_word, self.arg_word) if self.arg_word is not None else gb.agg(self.fct_word)
            self.df = self.df.reset_index() #TODO besoin ? non NewPM
            if isinstance(self.df[self.value].values[0], float):
                self.df[self.value]=self.df[self.value].astype(float)
            try:
                self.df[self.value] = self.df[self.value].astype(float)
            except:
                pass
        if self.fct_time is not None:
            gb = self.df.groupby( self.param_names+['word'] )
            if self.fct_time == "threshold": # on compte le temps pour atteindre le threshold
                df=gb.apply(lambda x : self.threshold(x))
                sel=df[df.th==True].groupby(self.param_names+['word'])
                self.df=sel.count()[self.value].reset_index()
                self.df['success']=sel.last()['success'].reset_index()['success']
                self.df['t']=0;self.df['num']=0
            elif self.fct_time=='derivative':
                self.df[self.value] = gb[self.value].transform(lambda x:x.diff(periods=-1))
            else:  # fonction qui n'agrège pas -> pas dans le groupby
                try:
                    self.df = gb.agg({self.value: (self.fct_time if not self.arg_time else lambda x : self.fct_time(self.arg_time)),'t':'first','num':'first'})
                except:
                    pdb.set_trace()
                self.df = self.df.reset_index()
            self.df = self.df.dropna()  # si la fonction crée des na
        if self.condition is not None:
            self.condition()
        self.tmax = self.df.t.max()
        if self.nMaxLen is not None:
            self.create_len()
            gp = self.df.groupby('len')
            l = [i for k in gp.groups.keys() for i in \
                 list(gp.get_group(k).groupby('word').groups.keys())[:self.nMaxLen]]
            self.df = self.df[self.df.word.isin(l)]
        self.df.drop('len', axis=1, errors='ignore')
        if self.nMax is not None:
            li = list(set(self.df.word))[:self.nMax]
            self.df = self.df[self.df.word.isin(li)]
        if self.remove_param is not None:  # on veut enlever un paramètre de nos plots
            self.define_param([i for i in self.test if i != self.remove_param])
            self.df = self.df.groupby(self.wnt + self.param_names).mean().drop(self.remove_param, axis=1).reset_index()
        if self.select_param is not None:  # on veut plotter seulement les données de 1 paramètre
            self.df = self.df.groupby(self.wnt + [self.select_param]).mean().reset_index()
            self.df = self.df.drop([i for i in self.param_names if i != self.select_param], axis=1)
            self.define_param(self.select_param)
        if self.per_letter:
            self.create_len()
            self.df[self.value]=self.df.astype({'value':'float'})[self.value]/self.df["len"]
            self.df.drop("len",axis=1,errors='ignore',inplace=True)
        self.select_size()
        if self.value=="success":
            self.df["success"]=self.df["success"].astype(float)
        self.df2csv()

    def set_titles(self, ax, ax_i, subtitle=""):
        if (isinstance(ax_i,bool) and ax_i) or ax_i[0] == self.w - 1:
            ax.set_xlabel(self.labelx,fontsize=24)
        if (isinstance(ax_i,bool) and ax_i) or ax_i[1] == 0:
            ax.set_ylabel(self.labely,fontsize=24)
        ax.tick_params(axis="both")
        ax.set_title(self.formate_title(subtitle),fontsize=23)
        # pier cogsci
        #ax.set_xticks([0,250,500,750,1000])
        #ax.set_yticks([0,0.25,0.5,0.75,1])
        #ax.margins(x=0)
        #dico={"simple":"$P_{inner}$","Ali":"constant division","L2_P":"$P_{proj}$",
        #      "L2_L":"$||L||$ division", "L2_PL":"$P_{cos}$"}
        #ax.set_title(dico[subtitle[0]], fontsize=25)
        #ax.tight_layout(rect=(0, 0, 1, 0.9))


    def set_lim(self):
        # set same limit for all subplots
        if isinstance(self.axs, np.ndarray):
            if self.xlim is None or self.ylim is None:
                lim = np.array([[ax.get_xlim(), ax.get_ylim()] for axs in self.axs for ax in axs]).swapaxes(0, 1)
                if self.xlim is None:
                    self.xlim = [min(ax[0] for ax in lim[0]), max(ax[1] for ax in lim[0])]
                if self.ylim is None:
                    df = self.df.drop(self.wnt, axis=1, errors='ignore')
                    # data entre 0 et 1 -> proba -> on affiche entre 0 et pour pas zoomer entre 0.9 et 1
                    [mini, maxi] = [df[self.value].min(), df[self.value].max()]
                    self.ylim = [min(ax[0] for ax in lim[1]), max(ax[1] for ax in lim[1])] if mini < 0 or maxi > 1 else [0,
        plt.setp(self.axs, xlim=self.xlim, ylim=self.ylim)]
        else:
            try:
                self.axs.set_xlim(self.xlim)
                self.axs.set_ylim(self.ylim)
            except: pass
        #if self.xlim is not None:
        #    plt.xlim(self.xlim)
        #if self.ylim is not None:
        #    pdb.set_trace()
        #    plt.ylim(self.ylim)

    def create_subplots(self):
        self.fig, self.axs = plt.subplots(self.w, self.h, sharex=True,  figsize=(5 * self.h, 3 * self.w))
        # pour pouvoir l'appeler par axs[(ax_i)]
        self.axs = np.array(self.axs)[np.newaxis, np.newaxis] if self.w == self.h == 1 else \
            np.array(self.axs)[np.newaxis] if self.w == 1 else np.array(self.axs)[:, np.newaxis] if self.h == 1 \
                else np.array(self.axs)

    def create_plot(self,figsize=(30,15)):
        if self.axs is None:
            self.fig, self.axs = plt.subplots(figsize=figsize)#,dpi=600)

    ##### Plot General

    def SubplotCurve(self, df, ax_i):
        ## plot des points représentant min, max et med + valeurs correspondantes sur les axes en rouge
        # si un seul t mais plusieurs num, on change la forme du df pour plotter
        if df.t.max() == 0 and df.num.max() > 0:
            df = df.drop('t', axis=1, errors='ignore')
            df = df.rename({'num': 't'}, axis='columns')
            self.plot_med = self.plot_color_bar = False
        # pour un affichage adapté aux valeurs mais avec des chiffres ronds : si marche pas, à définir à la main

        try:
            self.ylim = [math.floor(df[self.value].min()), math.ceil(df[self.value].max())]
        except:
            pdb.set_trace()
        diff = self.round(self.ylim[1] - self.ylim[0])
        self.ylim[0] = self.ylim[1] - diff
        if self.plot_med or self.plot_color_bar:
            df_end = df.groupby('word').agg('last').drop(['t', 'num'], axis=1, errors='ignore')
        if self.plot_med:
            M = [df_end.value.min(), df_end.value.max(), df_end.value.median()]
            for m, c in zip(M, ['pink', 'pink', 'red']):
                self.axs[ax_i].scatter(self.tmax + 1, m, c=c, s=50, marker='o', alpha=1, zorder=1000)
                self.axs[ax_i].scatter(self.tmax + 1, m, c=c, s=50, marker='o', alpha=1, zorder=1000)
            # ajoute min,med et max sur les valeurs de l'axe y
            L = M + [self.ylim[0] + t / 10 * (self.ylim[1] - self.ylim[0]) for t in range(11)]
            # on évite l'overlap dans l'affichage des valuers sur l'axe
            loc = [m for j, m in enumerate(L) if j in unique([int(round(x / (0.1 * diff))) * 0.1 * diff for x in L],
                                                             return_index=True)[1]]
            self.axs[ax_i].set_yticks(np.array(loc))
            for idx, m in enumerate(loc):
                self.axs[ax_i].get_yticklabels()[idx].set_color("red" if m in M else "black")
            self.axs[ax_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ## plot des courbes en elles-mêmes
        gpby = df.reset_index(drop=True).drop('num', axis=1, errors='ignore').groupby('word')
        for label, val in gpby:
            self.axs[ax_i].plot(val.t, val.value, alpha=self.alpha, color='b')
        df.groupby(['num','t']).median().reset_index(1).plot(x="t",y="value",color='r',ax=self.axs[ax_i],legend=False)
        ## scatterplot pour représenter la répartition des données
        if self.plot_color_bar:
            step = 20
            xsc = [self.tmax * 1.05] * step
            yl = self.ylim
            ysc = np.array(
                [[yl[0] + t / step * (yl[1] - yl[0]), yl[0] + (t + 1) / step * (yl[1] - yl[0])] for t in range(step)])
            R = [df_end[(df_end.value < b) & (df_end.value > a)].value.count() for a, b in ysc]
            repart = [r / sum(R) for r in R]
            self.axs[ax_i].scatter(xsc, (ysc[:, 0] + ysc[:, 1]) / 2, c=repart, s=200, cmap='Blues', marker='s',
                                   alpha=0.8)

    def SubplotHisto(self, df, ax_i):
        df = df[df.t == df.t.max()].drop(columns=self.wnt)
        self.xlim = [math.floor(df.min().min()), math.ceil(df.max().max())] if self.xlim is None else self.xlim
        df.hist(ax=self.axs[ax_i], bins=20, grid=False)

    def SubplotBar(self, df, ax_i):
        df = df[df.t == df.t.max()].drop(columns=self.wnt)
        df[self.value].value_counts().sort_index().plot(ax=self.axs[ax_i], kind='bar')

    # 100 expositions : évolution des connaissances lexicales
    @update
    def plotCurves(self, **kwargs):
        self.plotGeneral(self.SubplotCurve)

    # dernière exposition/itération : répartition en histogramme
    @update
    def plotHisto(self, **kwargs):
        self.plotGeneral(self.SubplotHisto)

    @update
    def plotBar(self, **kwargs):
        self.plotGeneral(self.SubplotBar)

    # labelx et labelx pas réinitialisés à chaque fois, mais title1 si
    def plotGeneral(self, function):
        # création de des plots
        self.selection()
        self.create_subplots()
        # remplissage des plots
        for l in self.param_tuples:
            print(l)
            ax_i = self.find_coordinates(l)
            # on sélectionne les bons paramètres
            df = self.select_subdf(l)
            function(df.drop(self.param_names, axis=1), ax_i)
            self.set_titles(self.axs[ax_i], ax_i, l)
        self.set_lim()
        plt.suptitle(self.title_list[self.select_num if self.select_num is not None and self.select_num < len(
         self.title_list) else 0] if self.title == "" else self.title, fontsize="xx-large")
        if self.show:
            plt.show()

    ########## Other plotting functions than plotGeneral #############

    @update
    def plotScatter(self, x="t", c=None, arg=None, **kwargs):
        """ gère 3 types de plots scatter :
        1/ x=="l" : subplots = différents paramètres comparés, x = len(stimulus) (valeur à t=-1 ou ds arg)
        2/ x=="t" :  subplots = différents paramètres comparés, x = t (nb expo/itérations)
            ici, on peut colorer les points en fonction de la fréquence/longueur en affectant c= "l" ou "f"
        3/ x=="p" :  subplots = t (besoin d'un petit nombre! donc expositions mais pas itérations), x=param
        4/ x=="f" :  subplots = différents paramètres comparés, x = freq (valeur à t=-1 ou ds arg)
        A compléter si veut d'autres plots scatter avec un X et un Subplot d'un autre type
        """
        self.selection()
        self.labelx = self.labelx if len(self.labelx) > 0 else "freq" if x == "f" else "len" if x == "l" \
            else self.param_names[0] if x == "p" else "t"
        if x == "l":
            t = min(self.df.t.max(), max(arg, 0)) if arg is not None else self.df.t.max()  # on choisit le t
            self.df = self.df[self.df.t == t]
            self.df["x"] = self.df.word.str.len()  # on crée la colonne len
            self.df = self.df.drop(columns=['t', 'word'])
        elif x == "t":
            if c == 'f':
                self.df = self.df.merge(self.df_freq, on='word')
                self.df["c"] = self.df.freq / self.df.freq.max() * 100
                self.df["c"] = self.df["c"].astype(int)
            self.df = self.df.drop(['freq', 'word'], axis=1, errors='ignore').rename({"t": "x"}, axis=1)
        elif x == "p":
            self.df.rename({"Qa": "x"},axis=1,inplace=True)
            # on transforme les données : on ne teste plus param mais t
            self.test={'t':list(set(self.df.t))}; self.param_names = ["t"]
            self.param_tuples = list(itertools.product(*list(self.test.values())))
            # on doit recalculer la taille des subplots (plus en fct de param mais de t)
            n = len(self.test["t"])
            self.w = math.ceil(math.sqrt(n))
            self.h = math.ceil(n / self.w)
        elif x == "f":
            t = min(self.df.t.max(), max(arg, 0)) if arg is not None else self.df.t.max()  # on choisit le t
            self.df = self.df[self.df.t == t]
            self.df = self.df.merge(self.df_freq, on='word').drop(['word', 't'], axis=1)
            self.df = self.df.rename({"freq": "x"}, axis=1)
        self.create_subplots()
        for l in self.param_tuples:
            df = self.select_subdf(l)
            ax_i = self.find_coordinates(l) #if x != 'p'# else self.find_coordinates_t(l)
            if c is not None:
                cmap = cm.get_cmap('viridis')
                self.df.plot.scatter(ax=self.axs[ax_i], x="x", y="value", c='c', cmap=cmap)
            else:
                self.df.plot.scatter(ax=self.axs[ax_i], x="x", y="value")
            self.set_titles(self.axs[ax_i], ax_i, " " if x=="" else l)
            if x == "f":
                self.axs[ax_i].set_xscale("log")
        self.set_lim()
        plt.suptitle(self.title_list[self.select_num if self.select_num is not None and self.select_num < len(
                self.title_list) else 0] if self.title == "" else self.title)
        if self.show:
            plt.show()

    @update
    def plotSummaryCurve(self, x="t", arg=None, fct_summary=np.mean, marker=True,error=None, min_index=None, figsize=(15,8), **kwargs):
        # on résume les données en utilisant une fonction aggrégeant les différents mots / même t
        # sélection sur les indices, la longueur etc
        # pour x=="p", doit avoir plusieurs paramètres
        # t : temps en abscisse
        # f : fréquence en abscisse
        self.selection()
        err,gb=None,None
        if x == "t":
            self.df = self.df.drop(['word', 'num'], axis=1, errors='ignore')
            gb = self.df.groupby(['t'] + self.param_names)
        elif x == "p":  # doit pas avoir de t
            self.df = self.df.drop(self.wnt, axis=1, errors='ignore')
            gb = self.df.groupby(self.param_names)
        else:
            t = min(self.tmax, max(arg, 0)) if arg is not None else self.tmax  # on choisit le t
            self.df = self.df[self.df.t == t]
            if x == "l":
                self.df["x"] = self.df.word.str.len()  # on crée la colonne len
                self.df = self.df.drop(columns=self.wnt, errors='ignore').set_index('x')
                gb = self.df.groupby(["x"] + self.param_names)
            elif x == "f":
                self.df = self.df.merge(self.df_freq, on='word').drop(['word', 't', 'num'], axis=1)  # freq en col x
                self.df["freq"] = pd.qcut(self.df["freq"], q=10).apply(lambda x: x.right)
                gb = self.df.groupby(["freq"] + self.param_names)
        if error is not None:
            err = gb.agg('std')[self.value].unstack(level=0 if x in ['p'] else 1)
        self.df = gb.agg(fct_summary)[self.value]
        self.df=self.df if len(self.test)==0 else self.df.unstack(level=0 if x in ['p'] else 1)
        # on change le nom des colonnes pour plotter
        self.create_plot(figsize=figsize)
        if min_index is not None:
            self.df.index=err.index=[i+min_index-self.df.index[0] for i in self.df.index]
        if error=="shift":
            columns=list(self.df.columns); step=columns[1]-columns[0]; n=len(columns)
            def decale_index(df): df.index = [ind+i*step/5 for ind in df.index]
            for i,col in enumerate(columns):
                val,err_val=self.df[col],err[col]
                decale_index(val); decale_index(err_val)
                val.plot(ax=self.axs,yerr=err_val,marker='o',legend=col)
                self.axs.legend(title=self.param_names[0])
        elif error=="ribbon":
            for i,col in enumerate(self.df.columns):
                self.df[col].plot(ax=self.axs)
                plt.fill_between(self.df.index, self.df[col]-err[col]/5,self.df[col]+err[col]/5, alpha=0.9)
        else: # contient barre d'erreur sans shift et pas de barre d'erreur
            self.axs = self.df.plot(ax=self.axs, marker='o',yerr=err) if marker else \
            self.df.plot(ax=self.axs,yerr=err,linewidth=2)
        self.set_lim()
        if x == "f" or self.logx:
            self.axs.set_xscale("log")
        if self.logy:
            self.axs.set_yscale("log")
        elif x in ["t","l"]:
            xticks = plt.xticks()[0]
            #if 1< xticks[-1] - xticks[0] < 20:
            #    self.axs.set_xticks(range(int(xticks[0]), math.ceil(xticks[-1])))
        elif x == "p":
            self.axs.set_xticks(self.df.index)
        self.axs.tick_params(axis="x", labelsize=25)
        self.axs.tick_params(axis="y", labelsize=25)
        self.labelx = self.labelx if self.labelx else "len" if x == "l" else "t" if x == "t" else "f"\
            if x == "f" else self.param_names[1]
        self.set_titles(self.axs, True, self.title)
        if self.show:
            plt.show(); self.axs=None


    # TODO redéfinir en termes de subplots en longueur, pas summary
    # summary c'est plutôt un paramètre qu'un type de graphe
    @update
    def plotSummaryLenCurve(self, arg=None, fct_summary='mean', marker=True, **kwargs):
        # un subplot par longueur de mot, et une courbe par paramètre au sein de chaque longueur
        # prérequis : on a un seul paramètre à plotter, sinon c'est la merde
        # on met le temps en abscisse
        # on résume les données en utilisant une fonction aggrégeant les différents mots / même t
        # sélection sur les indices, la longueur etc
        # on a soit un seul paramètre et plusieurs courbes, soit plusieurs courbes pour le
        # même paramètre
        self.selection()
        self.create_len()
        [lenMin, lenMax] = [self.df.len.min(), self.df.len.max()];
        diff = lenMax - lenMin + 1
        if fct_summary is not None:
            gb = self.df.drop(['num'], axis=1, errors='ignore').groupby(['len', 't', 'word'] + self.param_names)
            self.df = gb.agg(fct_summary).reset_index()
        self.w = 1 if diff < 5 else int(np.sqrt(diff))
        self.h = diff if diff < 5 else math.ceil(diff / self.w)
        self.create_subplots()
        for l in range(lenMin, lenMax + 1):
            ax_i = self.find_coordinates_t(l - lenMin)
            # df.columns = [self.formate_title(i) for i in df.columns]
            legend = 1 < len(self.param_tuples) < 10   # si trop de courbes, on enlève la légende pour visualiser
            df = self.df[self.df.len == l].drop('len', axis=1)
            if fct_summary is not None:
                df = df.groupby(['t']+self.param_names).mean().unstack().droplevel(0, axis=1)
                df.plot(ax=self.axs[ax_i], marker='o') if marker else \
                    df.plot(ax=self.axs[ax_i], legend=legend)
            else:
                gpby = df.groupby('word')#.plot(x="t",y=value, ax=self.axs[ax_i], legend=False,color="b")
                for label, val in gpby:
                    self.axs[ax_i].plot(val.t, val.value, alpha=self.alpha, color='b')
            xticks = self.axs[ax_i].get_xticks()
            if xticks[-1] - xticks[0] < 20 :
                self.axs[ax_i].set_xticks(range(int(xticks[0]), math.ceil(xticks[-1]) + 1))
            self.labelx='t'
            self.set_titles(self.axs[ax_i], True,  "len : " + str(l))
        plt.suptitle(self.title_list[self.select_num if self.select_num is not None and self.select_num < len(
            self.title_list) else 0] if self.title == "" else self.title)
        if self.show:
            plt.show()

    @update
    def plotHeatMap(self, dimx=None, dimy=None, x="p", etiquettes=False, select=True, **kwargs):
        # only with 2D parameters : mean of all values represented with color in 2D
        if select:
            self.selection()
        if dimx is None and dimy is None :
            dimx=self.param_names[0]; dimy=self.param_names[1]
        elif dimx is not None and dimy is None:
            dimy=self.param_names[1-self.param_names.index(dimx)]
        else:
            dimx=self.param_names[1-self.param_names.index(dimy)]
        if x == "t":  # doit avoir une seule dimension/ un seul param après le select
            gb = self.df.drop(['num', 'word'], axis=1, errors='ignore').groupby(self.param_names + ['t'])
            self.df = gb.mean().unstack(level=0).droplevel(0, axis=1)
            dimy = self.param_names[0]
        else:
            # si plusieurs t, on sélectionne le t maximal TODO
            self.df = self.df.drop(self.wnt, axis=1, errors='ignore')
            self.df = self.df.groupby(self.param_names).mean().unstack(level=dimx).droplevel(0, axis=1)
        # on est en dim3, il faut moyenner sur une dim
        if len(self.param_names) == 3:
            self.df = self.df.groupby(level=dimy if dimy < dimx else dimy - 1).mean()
        [mini, maxi] = [self.df.min().min(), self.df.max().max()]
        if x == "t":  # pour mettre le temps en abscisse
            self.df = self.df.T
        if mini > 0 and maxi <= 1:
            plt.imshow(self.df, cmap="Blues", vmin=0, vmax=1, origin='lower')
        else:
            plt.imshow(self.df, cmap="Blues", origin='lower')
        plt.colorbar()
        # étiquettes avec les valeurs
        if etiquettes:
            for i, p in enumerate(self.df.index):
                for j, c in enumerate(self.df.columns):
                    text = plt.text(j, i, round(self.df.loc[p, c], 3), ha="center", va="center", color="red")
        if x == "p":
            plt.xticks(range(len(self.test[dimx])), self.test[dimx])
        plt.yticks(range(len(self.test[dimy])), self.test[dimy])
        plt.xlabel("t" if x == "t" else dimx)
        plt.ylabel(dimy)
        plt.title(self.title_list[self.select_num if self.select_num is not None and self.select_num < len(
            self.title_list) else 0] if self.title == "" else self.title)
        if self.show:
            plt.show()

    @update
    def plotHeatMapT(self, etiquettes=False, **kwargs):
        ## only 1D parameter with index : selection is already done
        plt.imshow(self.df.T, cmap="Blues", vmin=0, origin='lower')
        plt.colorbar()
        # étiquettes avec les valeurs
        if etiquettes:
            for i, p in enumerate(self.df.index):
                for j, c in enumerate(self.df.columns):
                    text = plt.text(j, i, round(self.df.loc[p, c], 3), ha="center", va="center", color="red")
        plt.xlabel("nb expo")
        plt.ylabel(self.param_names[0])
        plt.yticks(range(len(self.value_list[0])), self.value_list[0])
        plt.title(self.title_list[self.select_num if self.select_num is not None and self.select_num < len(
            self.title_list) else 0] if self.title == "" else self.title)
        if self.show:
            plt.show()

    @update
    def plotHeatMapHisto(self,nbins=10, show=True, clim=None, **kwargs):
        self.selection()
        heatmap, xedges, yedges = np.histogram2d(self.df[self.param_names[0]], self.df[self.value], bins=nbins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        if clim is not None:
            plt.clim(clim[0],clim[1])
        plt.colorbar()
        plt.title(self.title)
        if show:
            plt.show()

    @update
    def plotScatterCorrelation(self, numx=0, numy=1, **kwargs):
        self.selection()
        dfx = self.df[self.df.num == numx]
        dfy = self.df[self.df.num == numy]
        self.create_subplots()
        for l in self.param_tuples:
            ax_i = self.find_coordinates(l)
            dfx_l = self.select_subdf(l,dfx)
            dfy_l = self.select_subdf(l,dfy)
            self.axs[ax_i].scatter(dfx_l[self.value], dfy_l[self.value], s=10, alpha=0.2)
            self.axs[ax_i].set_ylabel(self.title_list[numy] if len(self.title_list) > numy else "")
            self.axs[ax_i].set_xlabel(self.title_list[numx] if len(self.title_list) > numx else "")
            # self.set_titles(self.axs[ax_i], ax_i, labelx, labely, l)
        self.set_lim()
        if self.show:
            plt.show()

    @update
    def plotViolin(self, x="p",typ=False, **kwargs):
        # à la fin, il faut qu'il reste un seul t/num ou va en faire la moyenne
        self.selection()
        self.create_plot()
        if len(self.labely)==0:
            self.labely='value'
        self.df = self.df.rename({self.value: self.labely}, axis=1)
        if x == "p":
            if typ:
                self.df["type"] = "novel word" if self.PM else "known word"
                self.axs = sns.violinplot(x=self.param_names[0], y=self.labely, data=self.df,
                               palette="Paired", hue="type")
            else:
                self.axs = sns.violinplot(x=self.param_names[0], y=self.labely, data=self.df, palette="Paired")
        elif x == "l":
            self.create_len()
            self.axs = sns.violinplot(x="len", y=self.labely, data=self.df, palette="Paired", hue=self.param_names[0])
        elif x=='t':
            if type:
                self.axs = sns.violinplot(x="t",y=self.value,hue=self.param_names[0],data=self.df)
            else:
                self.axs = sns.violinplot(x="t", y=self.value, data=self.df)
        if self.show:
            plt.show();self.axs=None

    ######### Not used anymore

    def save_df(self, filename):
        if not os.path.exists('pkl_df'):
            os.mkdir('pkl_df')
        with open(filename, "wb") as f:
            pkl.dump([self.df_brut, self.test_brut, self.title_list], open('pkl_df/' + filename, 'wb'))

    def create_df(self):
        ## seulement pour les anciennes simu où on enregistrait un dico, pas un df
        word1 = list(self.dico.keys())[0]
        cond = list(self.dico[word1].keys());
        l = cond[0]
        if len(np.shape(list(self.dico.values())[0][l])) == 1:
            # si l'élement unique est une combinaison de résultats : on combine jamais plus de 3 résultats
            # c'est du bricolage, à changer!! mais série temporelle peut commencer à 4 expositions
            if len(list(self.dico.values())[0][l]) < 4:
                self.dico = {k: {key: [value] for key, value in v.items()} \
                             for k, v in self.dico.items()}
            else:
                self.dico = {k: {key: [[val] for val in value] for key, value in v.items()} \
                             for k, v in self.dico.items()}
            # si l'élement unique est une série temporelle
        y_list = [[lint for k, v in self.dico.items() for lext in v[c] for lint in lext] for c in cond]
        num_list = [i for k, v in self.dico.items() for lext in v[l] for i, lint in enumerate(lext)]
        time_list = [i for k, v in self.dico.items() for i, lext in enumerate(v[l]) for lint in lext]
        str_list = [k for k, v in self.dico.items() for i, lext in enumerate(v[l]) for lint in lext]
        d = {**{str(k): y_list[i] for i, k in enumerate(cond)}, **{"num": num_list, "t": time_list, "word": str_list}}
        self.df_brut = pd.DataFrame(d)
        self.df_brut.columns = [str(i) for i in self.df_brut.columns]

        def saveFig(self,name):
            self.axs.savefig(name,dpi=600)

## donner le nom pour plotter pour le fichier où ça n'est pas fait

# sim=simu(1000,"a","en",lexicon_name="lexiconBLP_SUBTLEX.csv",path=braidPath)
# for f in os.listdir('pkl/'):
#    print(f)
#    if "pkl" in f :
#        try:
#            p=plot(f)
#            if p.title_list==[""]:
#                print(p.df_brut)
#                title=input()
#                if title!="":
#                    sim_name=eval("sim.res_"+title)
#                    print(sim.res_titles(sim_name))
#                    pdb.set_trace()
#                    pkl.dump([p.df_brut,p.value_list,p.param_names,sim.res_titles(sim_name)],open('pkl/'+f, 'wb'))
#        except:
#            print("erreur",f)
#for f in os.listdir('pkl/'):
#    print(f)
#    if "pkl" in f:
#        try:
#            p = plot(f)
#        except:
#            print("erreur", f)
