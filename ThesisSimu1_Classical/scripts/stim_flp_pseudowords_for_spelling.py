import pdb

import pandas as pd

# Charger les données à partir des fichiers CSV
lexique_infra = pd.read_csv(r"..\\data\\external\\Lexique_infra.csv")
pw = pd.read_csv(r"..\\data\\processed\\flp_pseudowords_graphemic.csv")


col = ['pron','gpmatch']
lexique_infra[col] = lexique_infra[col].replace('8', 'y')
lexique_infra[col] = lexique_infra[col].replace('§', '&')
# Fonction pour extraire les couples graphème-phonème de gpmatch
def extract_grapheme_phoneme_pairs(gpmatch):
    return [pair.split("-") for pair in gpmatch.split(".") if "-" in pair]


# Appliquer cette extraction au dataframe
lexique_infra['grapheme_phoneme_pairs'] = lexique_infra['gpmatch'].apply(extract_grapheme_phoneme_pairs)


# Fonction pour isoler les premiers, derniers et intermédiaires couples graphème-phonème
def isolate_positions(grapheme_phoneme_pairs):
    if len(grapheme_phoneme_pairs) == 1:
        return grapheme_phoneme_pairs[0], None, None
    elif len(grapheme_phoneme_pairs) == 2:
        return grapheme_phoneme_pairs[0], None, grapheme_phoneme_pairs[1]
    else:
        return (grapheme_phoneme_pairs[0],
                grapheme_phoneme_pairs[1:-1],
                grapheme_phoneme_pairs[-1])


# Appliquer cette fonction au dataframe pour séparer les positions
lexique_infra[['first', 'middle', 'last']] = lexique_infra['grapheme_phoneme_pairs'].apply(
    lambda pairs: pd.Series(isolate_positions(pairs))
)

# Calculer les correspondances graphème-phonème par position
# Créer une liste pour chaque position
correspondances_first = []
correspondances_middle = []
correspondances_last = []

# Remplir les correspondances pour chaque position
for index, row in lexique_infra.iterrows():
    if row['first']:
        correspondances_first.append((row['first'][0], row['first'][1]))  # graphème, phonème
    if row['middle']:
        for middle_pair in row['middle']:
            correspondances_middle.append((middle_pair[0], middle_pair[1]))  # graphème, phonème
    if row['last']:
        correspondances_last.append((row['last'][0], row['last'][1]))  # graphème, phonème

# Créer des dataframes pour chaque position et calculer les fréquences
df_first = pd.DataFrame(correspondances_first, columns=['Grapheme', 'Phoneme'])
df_middle = pd.DataFrame(correspondances_middle, columns=['Grapheme', 'Phoneme'])
df_last = pd.DataFrame(correspondances_last, columns=['Grapheme', 'Phoneme'])

# Calculer les fréquences des correspondances pour chaque position
df_first_freq = df_first.groupby(['Grapheme', 'Phoneme']).size().reset_index(name='freq_first')
df_middle_freq = df_middle.groupby(['Grapheme', 'Phoneme']).size().reset_index(name='freq_middle')
df_last_freq = df_last.groupby(['Grapheme', 'Phoneme']).size().reset_index(name='freq_last')

# Calculer les probabilités pour chaque position
df_first_freq['prob_first'] = df_first_freq['freq_first'] / df_first_freq['freq_first'].sum() * 100
df_middle_freq['prob_middle'] = df_middle_freq['freq_middle'] / df_middle_freq['freq_middle'].sum() * 100
df_last_freq['prob_last'] = df_last_freq['freq_last'] / df_last_freq['freq_last'].sum() * 100

# Fusionner les résultats
df_correspondances = pd.merge(df_first_freq, df_middle_freq, on=['Grapheme','Phoneme'], how='outer')
df_correspondances = pd.merge(df_correspondances, df_last_freq, on=['Grapheme','Phoneme'], how='outer')

# Remplacer les NaN par 0
df_correspondances.fillna(0, inplace=True)

# Fonction pour obtenir la correspondance la plus probable en fonction de la position
def get_most_probable_phoneme(grapheme, position, correspondances):
    if position == "first":
        cor = correspondances.loc[correspondances.groupby('Grapheme')['prob_first'].idxmax()]
        row = cor[cor['Grapheme'] == grapheme]
    elif position == "last":
        cor = correspondances.loc[correspondances.groupby('Grapheme')['prob_last'].idxmax()]
        row = cor[cor['Grapheme'] == grapheme]
    else:
        cor = correspondances.loc[correspondances.groupby('Grapheme')['prob_middle'].idxmax()]
        row = cor[cor['Grapheme'] == grapheme]
    return row['Phoneme'].values[0] if not row.empty else None

# Fonction pour générer la prononciation la plus probable pour chaque mot
def generate_most_probable_pronunciation(graphemes, correspondances):
    positions = ["first"] + ["middle"] * (len(graphemes) - 2) + ["last"] if len(graphemes) > 1 else ["first"]
    most_probable_pronunciation = [get_most_probable_phoneme(g, pos, correspondances) for g, pos in zip(graphemes, positions)]
    if None in most_probable_pronunciation:
        return None  # Retourne None si une correspondance n'est pas trouvée

    return ''.join(map(str, most_probable_pronunciation))

# Appliquer les transformations sur les données filtrées
lexique_infra['graphemes'] = lexique_infra['gpmatch'].apply(lambda x: [pair.split("-")[0] for pair in x.split(".")])

pw['graphemes'] = pw['graphemic'].str.split(".")

pw['most_probable_pronunciation'] = pw.apply(
    lambda row: generate_most_probable_pronunciation(row['graphemes'], df_correspondances), axis=1)

pw['most_probable_pronunciation'] = pw['most_probable_pronunciation'].str.replace('#', '', regex=False)



# # Exporter le dataframe final en CSV
pw.to_csv(r"..\\data\\processed\\flp_pseudowords_most_probable.csv",
                    index=False)

# # Afficher un aperçu du dataframe
# print(lexique_infra[['word', 'gpmatch', 'most_probable_pronunciation']].head())
# print(df_correspondances)
