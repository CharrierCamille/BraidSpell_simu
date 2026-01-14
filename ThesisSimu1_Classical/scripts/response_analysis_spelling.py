import pandas as pd
import numpy as np
import itertools

# Charger les données
lexique_infra = pd.read_csv("../data/external/Lexique_infra.csv", sep=",")
raw_data = pd.read_csv("../data/external/chronolex_data.csv", sep=",")

# Filtrage
lexique_infra_filtered = lexique_infra[lexique_infra['word'].isin(raw_data['item'])]
lexique_infra_filtered = lexique_infra_filtered.drop_duplicates(subset=['word', 'pron'], keep='first')

# Extraire les phonèmes
def extract_phonemes(gpmatch):
    pairs = gpmatch.split(".")
    phonemes = [pair.split("-")[1] for pair in pairs if '-' in pair]
    return phonemes

# Extraction des correspondances avec positions
def extract_positional_correspondences_pg(lexique):
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
                records.append((p, g, pos))
    df = pd.DataFrame(records, columns=['Phoneme', 'Grapheme', 'Position'])
    df['freq'] = 1

    # Calcul des probabilités conditionnelles
    prob_df = df.groupby(['Phoneme', 'Grapheme', 'Position']).agg({'freq': 'sum'}).reset_index()
    prob_df['total'] = prob_df.groupby(['Phoneme', 'Position'])['freq'].transform('sum')
    prob_df['prob'] = prob_df['freq'] / prob_df['total']
    return prob_df

# Récupérer les options de graphèmes avec probas
def get_grapheme_options_pos(phoneme, position, df_corresp):
    return df_corresp[(df_corresp['Phoneme'] == phoneme) & (df_corresp['Position'] == position)][['Grapheme', 'prob']].values

# Fonction de génération des orthographes plausibles
def generate_spellings_with_scores(phonemes, df_corresp):
    positions = []
    for i in range(len(phonemes)):
        if i == 0:
            positions.append('initial')
        elif i == len(phonemes) - 1:
            positions.append('final')
        else:
            positions.append('middle')

    grapheme_options = []
    for p, pos in zip(phonemes, positions):
        options = get_grapheme_options_pos(p, pos, df_corresp)
        if len(options) == 0:
            return [], []
        grapheme_options.append(options)

    all_combinations = list(itertools.product(*grapheme_options))
    results = []
    for combo in all_combinations:
        graphemes = [g for g, _ in combo if g != '#']
        score = np.prod([prob for _, prob in combo])
        results.append((''.join(graphemes), score))

    results.sort(key=lambda x: -x[1])
    top_results = results[:10]
    return zip(*top_results) if top_results else ([], [])

# Appliquer les fonctions
positional_corresp_pg = extract_positional_correspondences_pg(lexique_infra_filtered)
lexique_infra_filtered.loc[:, 'phonemes'] = lexique_infra_filtered['gpmatch'].apply(extract_phonemes)
lexique_infra_filtered[['possible_spellings', 'scores']] = lexique_infra_filtered.apply(
    lambda row: pd.Series(generate_spellings_with_scores(row['phonemes'], positional_corresp_pg)), axis=1
)

# Dépliage et sauvegarde
lexique_infra_exploded = lexique_infra_filtered.explode(['possible_spellings', 'scores']).groupby('word')['possible_spellings'].apply(list).reset_index()
lexique_infra_exploded.to_csv("../data/processed/chronolex_spelling_possiblegraph.csv", index=False)

print("end")
