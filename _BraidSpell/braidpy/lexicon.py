#!/usr/bin/env python3
# -*-coding:utf-8-*-
import sys, os
from collections import defaultdict, Counter

sys.path.append("../")
sys.path.append("../../braidpy")
import codecs
import pickle as pkl
import csv
import pandas as pd
import pdb
import numpy as np
import braidpy.utilities as utl
import re


def selectdf(df, lenMin, lenMax, maxItem, maxItemLen):
    """
    Filters the dataframe provided based on the provided parameters `lenMin`, `lenMax`, `maxItem`, and `maxItemLen`,

    :param df: The input dataframe
    :param lenMin: The minimum length of the 'len' column in the dataframe
    :param lenMax: The maximum length of the words to include in the dataframe
    :param maxItem: The maximum number of items to be selected from the dataframe based on the frequency column (most frequent words are kept)
    :param maxItemLen: The maximum number of items to keep for each length group.
    :return: a subset of the input dataframe `df` based on the provided conditions.
    """
    df['len'] = df['len'].astype(int)
    if 'phlen' in df.columns:
        df['phlen'] = df['phlen'].astype(int)
    if lenMin is not None and lenMax is not None:
        df = df[(df['len'] >= lenMin) & (df['len'] <= lenMax)].set_index('word')
    if maxItem is not None:
        df = df.nlargest(maxItem, 'freq')
    if maxItemLen is not None and (maxItem is None or maxItemLen < maxItem):
        df = df.groupby('len').apply(lambda x: x.sort_values(by='freq', ascending=False).head(maxItemLen)).reset_index(
            0, drop=True)
    # word as index
    return df.head(maxItem)


def extractLexicon(path="../", lexicon_name="BLP.csv", maxItem=None, maxItemLen=None, lenMin=None,
                   lenMax=None, fMin=0, fMinPhono=None, fMinOrtho=None, fMax=1000000, cat=None, phono=False, ortho=True,
                   return_df=False):
    """
    Extracts the lexicon accoring to some selection criteria.

    :param path: string. path to the lexicon.
    :param lexicon_name: string. lexicon name.
    :param maxItem: The maximum number of items to be selected from the dataframe based on the frequency column (most frequent words are kept)
    :param maxItemLen: The maximum number of items to keep for each length group.
    :param lenMin: The minimum length of the 'len' column in the dataframe
    :param lenMax: The maximum length of the words to include in the dataframe
    :param fMin: float. minimum frequency for the words.
    :param fMinPhono: float. minimum phono frequency for the words to be included in the phono lexicon
    :param fMinOrtho: float. minimum ortho frequency for the words to be included in the ortho lexicon
    :param fMax: float. maximum frequency for the words.
    :param cat: string. grammatical category.
    :param phono: boolean. Modality extracted or not ?
    :param ortho: boolean. Modality extracted or not ?
    :param return_df: boolean. Must the df be returned instead of a list of word ?
    :return: A list of word or a dataframe.
    """
    # careful, if change lenMax, it changes the chosen words, this parameters cannot be progressively raised during expe.
    # MaxItemLen can be raised progressively
    df = pd.read_csv(os.path.realpath(os.path.dirname(__file__)) + '/../resources/lexicon/' + lexicon_name,
                     keep_default_na=False)
    # grammatical category
    # pdb.set_trace()
    if cat is not None and 'cat' in df.columns:
        df = df[df.cat == cat]
    df["word"] = df.word.str.lower().replace("'", "").replace("-", "").replace(" ", "")
    if "pron" in df.columns:
        df["pron"] = df.pron.str.replace(" ", "")
        df["phlen"] = df.pron.str.len()
    col = ['len', 'word', 'freq'] + (['pron', 'phlen'] if phono and 'pron' in df.columns else [])
    # Homographs handling
    agg_fun = {key: 'first' for key in col};
    agg_fun['freq'] = 'sum';
    agg_fun['len'] = 'min'
    df = df.groupby('word').agg(agg_fun).reset_index(drop=True)
    df = df.assign(ortho=ortho).assign(phono=(phono & ('pron' in df.columns)))
    if fMin is not None:
        df = df[df['freq'] > fMin]
    if fMax is not None:
        df = df[df['freq'] < fMax]
    if fMinPhono is not None:
        df.loc[df['freq'] < fMinPhono, 'phono'] = False
    if fMinOrtho is not None:
        df.loc[df['freq'] < fMinOrtho, 'ortho'] = False
    df = selectdf(df, lenMin, lenMax, maxItem, maxItemLen)  # .to_dict()['freq']
    return df if return_df else list(df.reset_index().word)


def simplify_letters(df):
    """
    Simplifies the lexicon by merging some letters of the dataframe.

    :param df: input dataframe (lexicon)
    :return: the filtered dataframe
    """

    def f(val):
        a = {k: 'a' for k in ['ä', 'ã', 'á']}
        o = {k: 'o' for k in ['ö', 'ó']}
        u = {k: 'u' for k in ['ù', 'ü', 'ú']}
        i = {k: 'i' for k in ['í']}
        return dict(**{' ': '', ':': '', '?': ''}, **a, **o, **u, **i)[val[0]]

    # pour gérer quand word est en index
    index_name = df.index.name
    if index_name is not None:
        df = df.reset_index()
    for col in ['orthosyll', 'segm', 'gpmatch', 'word']:
        if col in df.columns:
            df[col] = df[col].str.replace('ã|ä|á|ö|ó|ù|ü|ú|í| |:|\?', f, regex=True)
    df.set_index(index_name if index_name is not None else 'word')
    return df


def simplify_phonemes(df):
    """
    Simplifies the lexicon by merging some phonemes of the dataframe.

    :param df: input dataframe (lexicon)
    :return: the filtered dataframe
    """

    def f(val):
        return {'O': 'o', 'E': 'e', '9': '2', '1': '5', '8': 'y', ' ': '', 'A': 'a', '*': '°', '§': '&', 'U': 'u'}[
            val[0]]

    for col in ['pron', 'syll', 'pseg', 'phono', 'gpmatch', 'pron_x', 'pron_y']:
        if col in df.columns and not df[col].dtype == 'bool':
            df[col] = df[col].str.replace('O|E|9|1| |A|\*|8|§', f, regex=True)
            # df = df[~df[col].str.contains('\'| |\.')]
    return df


def extract_spanish_lexicon(path):
    """
    Extracts the spanish lexicon from txt file and stores it as a csv file.

    :param path: path to the lexicon
    :return:
    """
    df = pd.read_csv(path + 'resources/lexicon/lexique_espagnol.txt', sep='\t', encoding='latin')
    freq = pd.read_csv(path + 'resources/lexicon/freq_espagnol.txt', sep='\t', encoding='latin')
    lex = df[['word', 'pron']].merge(freq, on='word')
    lex['pron'] = lex['pron'].str.replace('-', '')
    lex = simplify_letters(simplify_phonemes(lex))
    lex['len'] = lex.word.str.len()
    lex['phlen'] = lex.pron.str.len()
    lex.to_csv(path + 'resources/lexicon/lexique_espagnol.csv')


# extract_spanish_lexicon('~/braidpy/')


def extractPM(path="../", lexicon_name="pwords.txt", lenMin=3, lenMax=8, maxItem=None, maxItemLen=None):
    """
    Reads a lexicon file and returns a list of words that meet certain length and frequency criteria.

    :param path: Directory path where the lexicon file is located. By default, it is set to "../".
    :param lexicon_name: Name of the lexicon file. By default, it is set to "pwords.txt", defaults to pwords.txt (optional)
    :param maxItem: The maximum number of items to be selected from the dataframe based on the frequency column (most frequent words are kept)
    :param maxItemLen: The maximum number of items to keep for each length group.
    :param lenMin: The minimum length of the 'len' column in the dataframe
    :param lenMax: The maximum length of the words to include in the dataframe
    :return: a list of indices from a DataFrame.
    """
    df = pd.read_csv(path + "resources/lexicon/PM/" + lexicon_name, delimiter=' ', usecols=[1, 5])
    df.columns = ['freq', 'word']
    df['len'] = df['word'].str.len()
    return list(selectdf(df, lenMin, lenMax, maxItem, maxItemLen).index)


def extractPM_BLP(path="../", lexicon_name="pseudo-words-BLP.csv", lenMin=2, lenMax=8, maxItem=None, maxItemLen=None):
    """
    Reads a lexicon file and returns a list of words that meet certain length and frequency criteria. Should be the same format as the BLP format (some operations are needed to transform data).

    :param path: Directory path where the lexicon file is located. By default, it is set to "../".
    :param lexicon_name: Name of the lexicon file. By default, it is set to "pwords.txt", defaults to pwords.txt (optional)
    :param maxItem: The maximum number of items to be selected from the dataframe based on the frequency column (most frequent words are kept)
    :param maxItemLen: The maximum number of items to keep for each length group.
    :param lenMin: The minimum length of the 'len' column in the dataframe
    :param lenMax: The maximum length of the words to include in the dataframe
    :return: a list of indices from a DataFrame.
    """
    df = pd.read_csv(path + "resources/lexicon/PM/" + lexicon_name, delimiter=',', usecols=[0, 1])
    df = df[df.lexicality == 'N']
    df.columns = ['word', 'lexicality']
    df['word'] = df['word'].str.replace('_', '')
    df['len'] = df['word'].str.len()
    return list(selectdf(df, lenMin, lenMax, maxItem, maxItemLen).index)


def extractLetterFreq(lexicon, letters):
    """
    Calculates letter frequency based on words frequency of all words in the lexicon.

    :param lexicon: lexicon dictionary (word,freq)
    :param letters: letters dictionary
    :return:
    """
    letterFreq = {key: 0 for key in letters.keys()}
    for word, f in lexicon.items():
        for letter in word:
            letterFreq[letter] += f
    s = sum(letterFreq.values())
    return {key: value / s for key, value in letterFreq.items()}


def del_index(s, i):
    """
    Removes index i from a string.

    :param s: string
    :param i: index
    :return:  the string without the index i
    """
    if i == len(s) - 1:
        return s[:-1]
    if i < len(s) - 1:
        return s[:i] + s[i + 1:]
    return None


def is_consistent(r1, r2):
    """
    Calculates if 2 words are consistant with each other.
    definition consistance :
    MTM : A word is said to be consistent if its pronunciation agrees with those of similarly spelled words (its orthographic neighbors)
    Borleff2017 : consistency approach to measure transparency
    dichotomous approach : a word or smaller sized unit is regarded consistent when there is only one possible mapping and inconsistent when there are alternative mappings.
    gradual approach : the level of consistency is expressed as the proportion of dominant mappings over the total number of occurrences of the base unit analyzed.

    :param r1: first word
    :param r2: second word
    """

    # cas particulier des mots commençant avec une lettre muette
    def h_begin(r):
        if r.segm[0] == 'h' and len(r.segm.split('.')[0]) > 0:
            r.segm = r.segm[0] + '.' + r.segm[1:]
            r.pseg = '#.' + r.pseg
        return r

    r1 = h_begin(r1);
    r2 = h_begin(r2)
    o1 = r1.segm;
    o2 = r2.segm;
    # pas de la même longueur : on ne calcule pas la consistance
    # absents / absente : on en fait quoi ??? règle contextuelle, mais c'est inconsistant ??
    # que faire avec longueur phono différente ?? a priori c'est non consistant non ??
    if o1 == o2 or len(o1.replace('.', '')) != len(o2.replace('.', '')):
        return True
    pos1 = [i for i, l in enumerate(o1) if l != '.']
    pos2 = [i for i, l in enumerate(o2) if l != '.']
    diff = [i for i, (oi, oj) in enumerate(zip(o1.replace('.', ''), o2.replace('.', ''))) if oi != oj]
    # pas voisin : on ne calcule pas la consistance
    if len(diff) != 1:
        return True
    # on "enlève" le graphème avec une lettre différente
    p1 = r1.pseg
    p2 = r2.pseg
    p1_l = del_index(p1.split('.'), o1[:pos1[diff[0]]].count('.'))
    p2_l = del_index(p2.split('.'), o2[:pos2[diff[0]]].count('.'))
    if len(p1_l) != len(p2_l):
        return False
    for i, (pi, pj) in enumerate(zip(r1.pseg.split('.'), r2.pseg.split('.'))):
        if i != diff[0] and pi != pj:
            return False
    return True


def calculate_syllabic_consistency(name, df):
    """
    Calculates the syllabic consistency of a word.

    :param name: word we want the consistency
    :param df: lexicon
    :return: the syllabic consistency
    """
    df = df.groupby('word').first()

    def f(x1, x2):
        return '.'.join([i + '-' + j for i, j in zip(x1.split('-'), x2.split('-'))])

    df['gpmatch'] = df.apply(lambda x: f(x.ortho, x.phono), axis=1)
    return calculate_consistency(name, df)


def calculate_consistency(name, df):
    """
    Calculates consistency for the whole lexicon.

    :param name: name of the file to be generated.
    :param df: input dataframe
    :return: a dataframe with new columns
    """
    df = df.groupby('word').first()
    # on crée les consistances par graphèmes
    dico = defaultdict(list)
    dico_g = {}
    CGP = [i.split('-') for i in '.'.join(df.gpmatch.values).split('.') if len(i) > 2]
    for i, j in CGP:
        dico[i].append(j)
    # les occurrences de chaque phonème associé à un graphème
    for grapheme in dico:
        dico[grapheme] = Counter(dico[grapheme])
        s = sum(dico[grapheme].values())
        dico[grapheme] = {key: dico[grapheme][key] / s for key in dico[grapheme]}
        dico_g[grapheme] = s
    # on récupère la fréquence de chaque phonème associé à un graphème
    f = pd.DataFrame(list([[key1 + '-' + key2, value] for key1 in dico for key2, value in dico[key1].items()]),
                     columns=['gp', 'value'])
    # on crée les consistances pour chaque mot
    df['cons'] = df['gpmatch'].apply(
        lambda s: " ".join([str(round(f[f.gp == i].value.iloc[0], 5)) for i in s.split('.') if len(i) > 2]))
    # on calcule la consistance min
    df['cons-min'] = df['cons'].apply(lambda s: min([float(i) for i in s.split(' ') if len(i) > 0]))
    df['occ'] = df['gpmatch'].apply(
        lambda s: " ".join([str(dico_g[i.split('-')[0]]) for i in s.split('.') if len(i) > 0]))
    df['occ-min'] = df['occ'].apply(lambda s: min([int(i) for i in s.split(' ') if len(i) > 0]))
    df.to_csv(name[:-4] + '_cons.csv')
    return df
    # on calcule la fréquence d'occurrence de chaque graphème


def correct_graphemic_segmentation(df):
    """
    Corrects errors in the graphemic segmentation given by Manulex.

    :param df: dataframe with graphemic segmentation
    :return: a new dataframe
    """
    cons_phono = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'R', 's', 't', 'v', 'z', 'S', 'G', 'J', 'N', 'Z', 'j']

    def sp(s):
        return s.split('.')

    def transfo(x):
        # transformation pour obtenir des graphèmes plus élémentaires
        # ne fonctionne pas hyper bien pour les joyeux et autres mots compliqués en waj
        ortho = str(x.segm);
        phono = str(x.pseg).replace('*', '°');
        # les G sont transformés en ng
        if 'G' in phono:
            phono = phono.replace('G', 'n.g')
            ortho = ortho.replace('ng', 'n\.g')
        # on enlève les diphtongues en j
        l = [['ij', 'i'], ['i.j', 'i'], ['ji', 'i'], ['j.i', 'i']] + [['j.' + string, 'i.' + string] for string in
                                                                      ['a', 'e', 'o', 'y', '@', '&', '5', '2']]
        for i in l:
            if 'ill' not in x.word and 'y' not in ortho and i[0] in phono:
                phono = phono.replace(i[0], i[1])
        w = [['w.' + string, 'u.' + string] for string in ['a', 'e', 'o', 'y', '@', '&', '5', '2', 'i']]
        for i in w:
            if not re.search('oi|oy|w|qu', x.word) and i[0] in phono:
                phono = phono.replace(i[0], i[1])
        # on garde celles en w seulement si y a oi/oy/qu/w
        # on sépare les ill /ij/ etc pour faire des petits graphèmes
        if 'ill' in ortho and ('ij' in phono or 'il' in phono):
            ortho = ortho.replace('ill', 'i.ll')
            phono = phono.replace('ij', 'i.j').replace('il', 'i.l')
        if 'lli' in ortho and ('ji' in phono or 'lj' in phono):
            ortho = ortho.replace('lli', 'll.i')
            phono = phono.replace('ji', 'j.i').replace('lj', 'l.j')
        if 'ay' in ortho and ('ej' in phono or 'ei' in phono or 'aj' in phono):
            ortho = ortho.replace('ay', 'a.y')
            phono = phono.replace('ej', 'e.j').replace('ei', 'e.i').replace('aj', 'a.j')
        if 'enn' in ortho and ('en' in phono or '@n' in phono):
            ortho = ortho.replace('enn', 'e.nn')
            phono = phono.replace('en', 'e.n').replace('@n', '@.n')
        if ('emm' in ortho or 'um' in ortho) and ('am' in phono or 'om' in phono or '@m' in phono):
            ortho = ortho.replace('emm', 'e.mm').replace('um', 'u.m')
            phono = phono.replace('am', 'a.m').replace('om', 'o.m').replace('@m', '@.m')
        if 'er' in ortho and 'eR' in phono:
            ortho = ortho.replace('er', 'e.r')
            phono = phono.replace('eR', 'e.R')
        # les dièses du début sont supprimés, on regroupe les 2 graphèmes ortho
        if phono[0] == '#':
            phono = phono[2:]
            ortho = ortho.replace('.', '', 1)
        # les dièses du milieu sont transformés en schwa entre 2 consonnes, en rien sinon
        phono_sp = phono.split('.')
        if '#' in phono_sp and phono_sp.index('#') < len(phono_sp) - 1:
            idx_phono = phono_sp.index('#')  # i-ème point à supprimer
            # entre 2 consonnes, on rajoute un schwa
            if phono_sp[idx_phono - 1] in cons_phono and phono_sp[idx_phono + 1] in cons_phono:
                phono = re.sub(r'^(.*?(#.*?){0})#', r'\1°', phono)
            # si y a une voyelle, on met pas de schwa
            else:
                idx_ortho = [index for index, char in enumerate(ortho) if char == '.'][idx_phono - 1]
                ortho = ortho[:idx_ortho] + ortho[idx_ortho + 1:]
                idx = phono.index('#')
                phono = phono[:idx] + phono[idx + 2:]
        # les dièses de la fin sont supprimés
        # idx_pt = [i for i, l in enumerate(ortho) if l == '.']
        # idx_i = [l for i, l in enumerate(idx_pt) if i in idx]
        # x.segm = "".join([o for i, o in enumerate(ortho) if i not in idx_i])

        # n=phono.count('.'); phono=phono.replace('.#',''); n=n-phono.count('.')
        # ortho=''.join(ortho.rsplit('.',n))
        # on regroupe les dièses finaux en un seul phonème/graphème
        if '.#.#' in phono:
            cpt = phono.count('#')
            ortho = ortho[::-1].replace('.', '', cpt - 1)[::-1]
            phono = phono.replace('.#', '') + '.#'
        # on rajoute le schwa final si nécessaire
        if len(phono) > 0 and phono.replace('.#', '')[-1] in cons_phono and 'e' in ortho.split('.')[-1]:
            phono = phono.replace('.#', '') + '.°'
        muet = ['t', 'd', 'p', 'g', 'c', 's', 'e', 'es', 'ts', 'ent', 'x', 'ds', 'cs', 'gs', 'b', 'bs', 'f', 'h', 'ct',
                'hs', 'fs', 'ps', 'pt', 'gt', 'ls', 'th', 'l', 'cts', 'st', 'ht', 'rs']
        last = ortho.split('.')[-1]
        # colle les consonnes muettes finales à la dernière voyelle
        if '.' in ortho and last in muet and phono.split('.')[-1] == '#':
            ortho = ortho[:-(len(last) + 1)] + ortho[-len(last):]
            phono = phono[:-2]
        # bs devient ps même si ça commence par p : contrainte articulatoire
        if 'p.s' in phono and 'b' in ortho:
            phono = phono.replace('p.s', 'b.s')
        if 'p.t' in phono and 'b' in ortho:
            phono = phono.replace('p.t', 'b.t')
        # graphèmes à 2 lettres/2 phonèmes sont séparés en 2
        ortho_s = sp(ortho);
        phono_s = sp(phono)
        # db_graphemes = [i_db for i_db, (i, j, k, l) in enumerate(zip(map(len, ortho_s), map(len, phono_s), ortho_s, phono_s)) if i > 1 and j > 1]
        # if len(db_graphemes) > 0:
        #    for i in db_graphemes:
        #        ortho_s[i] = ortho_s[i][0] + '.' + ortho_s[i][1:];
        #        ortho = '.'.join(ortho_s)
        #        phono_s[i] = phono_s[i][0] + '.' + phono_s[i][1:];
        #        phono = '.'.join(phono_s)
        x.pseg = '.'.join(phono_s)
        x.segm = '.'.join(ortho_s)
        x.word = ''.join(ortho_s)
        x.phono = ''.join(phono_s).replace('#', '')
        x.gpmatch = '.'.join(['-'.join(i) for i in list(zip(ortho.split('.'), phono.split('.')))])
        return x

    df = simplify_letters(simplify_phonemes(df))
    df.apply(transfo, axis=1)
    # da = pd.read_csv('~/braidpy/resources/lexicon/lexique_fr.csv')[['word', 'pron']]
    # res = df.merge(da, on='word')
    # res2=res[res.phono!=res.pron]
    # reste quelques différences
    # homophones 'ent' @ pour sylviane, rien pour lexique
    # règle u/w pas clair
    #  j + schwa ?
    return df


def createManulexCP(path='../'):
    """ Pre-processes Manulex data in a usable format.

    :param path: path to the file
     """
    df = pd.read_csv(path + "resources/lexicon/Manulex.csv",
                     usecols=["FORMES ORTHOGRAPHIQUES", "NLET", "CP F"]).dropna().rename(
        columns={"FORMES ORTHOGRAPHIQUES": "word", "NLET": "len", "CP F": "freq"})
    df['freq'].astype(float)  # considerate freq as float
    # remove words with space
    df = df.drop(df[df["word"].str.contains(" ")].index)
    df = df.drop(df[df["word"].str.contains("'")].index)
    df = df.drop(df[df["word"].str.contains("-")].index)
    df = df.drop(df[df["word"].str.contains("1")].index)
    df = df.drop(df[df["word"].str.contains("œ")].index)
    # remove accents
    df["word"] = df["word"].str.replace("à", "a")
    df["word"] = df["word"].str.replace("é", "e")
    df["word"] = df["word"].str.replace("è", "e")
    df["word"] = df["word"].str.replace("ê", "e")
    df["word"] = df["word"].str.replace("ë", "e")
    df["word"] = df["word"].str.replace("ï", "i")
    df["word"] = df["word"].str.replace("î", "i")
    df["word"] = df["word"].str.replace("ô", "o")
    df["word"] = df["word"].str.replace("û", "u")
    df["word"] = df["word"].str.replace("ù", "u")
    df["word"] = df["word"].str.replace("ç", "c")
    df["word"] = df["word"].str.replace("â", "a")
    ## du coup on doit rassembler plusieurs mots
    agg_fun = {'freq': np.sum, 'len': 'first'}
    df = df.groupby('word').agg(agg_fun).reset_index()  # ajoute les fréquences des homographes
    df.to_csv(path + "resources/lexicon/ManulexCP.csv")


def neighbSize(df):
    """
    Calculates for each word of the dataframe the neighborhood size

    :param df: input dataframe
    :return: a dataframe with an added nb column
    """

    def nbNeigh(key):
        dfN = pd.DataFrame({"dist": df.word.apply(lambda x: distance(x, key))})
        return dfN.dist[dfN.dist == 1].count()

    df["nb"] = df.word.apply(nbNeigh)  # on crée une colonne dist dans le df
    return df


def neighbSizeLen(df):
    """
    Calculates for each word the neighborhood size, neighbors in the same length class

    :param df: input dataframe
    :return: a dataframe with an added nb column
    """

    def nbNeigh(key):
        dfa = df[df.len == len(key)]
        dfN = pd.DataFrame({"dist": dfa.word.apply(lambda x: distance(x, key))})
        return dfN.dist[dfN.dist == 1].count()

    df["len"] = df.word.str.len()
    df["nb"] = df.word.apply(nbNeigh)  # on crée une colonne dist dans le df
    return df


def neighbSizeLenList(df, l):
    """
    Calculate for each word in l the neighborhood size in the length class

    :param df: input dataframe
    :return: neighborhood size for one word.
    """

    def nbNeigh(key):
        dfa = df[df.len == len(key)]
        dfN = pd.DataFrame({"dist": dfa.word.apply(lambda x: distance(x, key))})
        return dfN.dist[dfN.dist == 1].count()

    df["len"] = df.word.str.len()
    nb = [nbNeigh(i) for i in l]
    dfRes = pd.DataFrame({"word": l, "nb": nb})
    return dfRes


def neighbSizeList(df, l):
    """
    Calculate for each word in l the neighborhood size

    :param df: input dataframe
    :return: neighborhood size for one word.
    """

    def nbNeigh(key):
        dfa = df[df.len == len(key)]
        dfN = pd.DataFrame({"dist": dfa.word.apply(lambda x: distance(x, key))})
        return dfN.dist[dfN.dist == 1].count()

    df["len"] = df.word.str.len()
    nb = [nbNeigh(i) for i in l]
    dfRes = pd.DataFrame({"word": l, "nb": nb})
    return dfRes


def removeNeigh(df, l):
    """
    Removes neighbors of the word l from the dataframe df.

    :param df: input dataframe
    :param l: word
    :return: the new dataframe
    """
    df = df.reset_index()
    for i in l:
        df["dist"] = df.word.apply(lambda x: distance(x, i))
        df = df[df.dist != 1]
    return df.drop("dist", axis=1)
