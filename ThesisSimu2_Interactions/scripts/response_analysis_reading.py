import pdb

import pandas as pd
import numpy as np
import itertools

# Charger les données à partir des fichiers CSV
lexique_infra = pd.read_csv("../data/external/Lexique_infra.csv", sep=",")
raw_data = pd.read_csv("../data/processed/full_stim.csv", sep=",")

# Filtrer le lexique_infra pour ne conserver que les mots présents dans raw_data["item"]
lexique_infra_filtered = lexique_infra[lexique_infra['word'].isin(raw_data['word'])]
lexique_infra_filtered = lexique_infra_filtered.drop_duplicates(subset=['word', 'pron'], keep='first')

# Fonction pour extraire les graphèmes à partir de gpmatch
def extract_graphemes(gpmatch):
    gpmatch_pairs = gpmatch.split(".")
    graphemes = [pair.split("-")[0] for pair in gpmatch_pairs]
    return graphemes

# Fonction pour extraire les correspondances graphème-phonème avec position
def extract_positional_correspondences(lexique):
    records = []
    for gpmatch in lexique['gpmatch']:
        pairs = gpmatch.split('.')
        for i, pair in enumerate(pairs):
            if '-' in pair:
                g, p = pair.split('-')
                if i == 0:
                    pos = 'initial'
                elif i == len(pairs) - 1:
                    pos = 'final'
                else:
                    pos = 'middle'
                records.append((g, p, pos))
    df = pd.DataFrame(records, columns=['Grapheme', 'Phoneme', 'Position'])
    df['freq'] = 1
    # Probas conditionnelles par position
    prob_df = df.groupby(['Grapheme', 'Phoneme', 'Position']).agg({'freq': 'sum'}).reset_index()
    prob_df['total'] = prob_df.groupby(['Grapheme', 'Position'])['freq'].transform('sum')
    prob_df['prob'] = prob_df['freq'] / prob_df['total']
    return prob_df

# Fonction pour obtenir les phonèmes possibles à une position donnée
def get_phoneme_options_pos(grapheme, position, df_corresp):
    return df_corresp[(df_corresp['Grapheme'] == grapheme) & (df_corresp['Position'] == position)][['Phoneme', 'prob']].values

def has_consecutive_phonemes(phonemes):
    for i in range(1, len(phonemes)):
        if phonemes[i] == phonemes[i - 1]:
            return True
    return False

# Fonction modifiée pour générer les prononciations et leurs scores
def generate_pronunciations_with_scores(graphemes, df_corresp):
    positions = []
    for i in range(len(graphemes)):
        if i == 0:
            positions.append('initial')
        elif i == len(graphemes) - 1:
            positions.append('final')
        else:
            positions.append('middle')

    phoneme_options = []
    for g, pos in zip(graphemes, positions):
        options = get_phoneme_options_pos(g, pos, df_corresp)
        if len(options) == 0:
            return [], []
        phoneme_options.append(options)

    all_combinations = list(itertools.product(*phoneme_options))
    results = []
    for combo in all_combinations:
        phonemes = [p for p, _ in combo if p != '#']

        # Filtrer : ne garder que les prononciations sans phonèmes consécutifs identiques
        if has_consecutive_phonemes(phonemes):
            continue

        score = np.prod([prob for _, prob in combo])
        results.append((''.join(phonemes), score))

    results.sort(key=lambda x: -x[1])
    top_results = results[:10]
    return zip(*top_results) if top_results else ([], [])

# Extraire les correspondances positionnelles
positional_corresp = extract_positional_correspondences(lexique_infra)

# Extraire les graphèmes pour chaque mot
lexique_infra_filtered.loc[:, 'graphemes'] = lexique_infra_filtered['gpmatch'].apply(extract_graphemes)

# Générer les 10 meilleures prononciations et leurs scores
lexique_infra_filtered[['possible_pronunciations', 'scores']] = lexique_infra_filtered.apply(
    lambda row: pd.Series(generate_pronunciations_with_scores(row['graphemes'], positional_corresp)), axis=1
)

# Déplier le dataframe
lexique_infra_exploded = lexique_infra_filtered.explode(['possible_pronunciations', 'scores']).groupby('word')['possible_pronunciations'].apply(list).reset_index()

# Sauvegarder le résultat
lexique_infra_exploded.to_csv("../data/processed/consistency_reading_possiblepron.csv", index=False)

print("end")
