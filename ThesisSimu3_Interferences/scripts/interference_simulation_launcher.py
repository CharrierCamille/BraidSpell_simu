######################
### A - Imports ######
######################
import pdb

braidPath = "../../../_BraidSpell/"
import sys

sys.path.append(braidPath)

from braidpy.simu import simu
from braidpy.expe import expe

# specific libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(seed=240904)

#######################
### B - Model init ####
#######################

def simu_definition(input_type="visual", remove_stim_ortho=False, remove_stim_phono=False, stim='!partir!',
                    simu_type="H", criterion_type="both", sd=1.75):
    if input_type == "visual":
        ortho_param = {"learning": False, "remove_stim": remove_stim_ortho, 'stim': stim, 'sd': sd, 'Q': 1,
                       "decoding_ratio_po": 0.001}
        phono_param = {"learning": False, "remove_stim": remove_stim_phono, 'Q': 1,
                       "decoding_ratio_op": 0.005}
        if criterion_type == "ld":
            criterion = "ld"
        if criterion_type == "reco":
            criterion = "W"
    elif input_type == "auditory":
        ortho_param = {"learning": False, "remove_stim": remove_stim_ortho, "decoding_ratio_po": 0.005}
        phono_param = {"learning": False, "remove_stim": remove_stim_phono, 'stim': stim, 'sd':sd, 'Q': 1,
                        "decoding_ratio_op": 0.001}
        if criterion_type == "ld":
            criterion = "ld_phi"
        if criterion_type == "reco":
            criterion = "WPhi"

    if criterion_type == "both":
        criterion = "bothMean"

    # print(criterion)
    simu_param = {"level": "expe", "max_iter": 3000, "simu_type": simu_type,
                  "stop_criterion_type": criterion, "sampling": True, "pos_init":-2}
    simu_args = {}
    model_param = {"langue": "fr", "path": braidPath, "input_type": input_type}

    semantic_param = {"context_sem": False}
    return simu(model_param, ortho_param, phono_param, semantic_param, simu_args, **simu_param)


#########################
### C - Param definition ####
#########################

stimuli_reading = pd.read_csv("../data/processed/full_stim_v.csv", index_col=0).reset_index().target.tolist()
stimuli_spelling = pd.read_csv("../data/processed/full_stim_a.csv", index_col=0).reset_index().pron.tolist()

stimuli_spelling = [item.replace("1","5") for item in stimuli_spelling]
stimuli_reading = [f'!{word}!' for word in stimuli_reading]
stimuli_spelling = [f'!{word}!' for word in stimuli_spelling]
stimuli_reading = list(set(stimuli_reading))
stimuli_spelling = list(set(stimuli_spelling))

######################
### D - Expe init ####
######################

# simulation for visual known words
sim_reading_dl = simu_definition(input_type="visual", remove_stim_ortho=False, remove_stim_phono=False, criterion_type="ld", sd=1.75)

expe_param_reading_dl = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "inteference_dl_read_knownword",
              "n_expo": 1,
              "test": {"test":[1]},
              "liste": stimuli_reading}

exp_reading_dl = expe(simu=sim_reading_dl, **expe_param_reading_dl)

# simulation for visual known words
sim_reading_reco = simu_definition(input_type="visual", remove_stim_ortho=False, remove_stim_phono=False, criterion_type="reco", sd=1.75)

expe_param_reading_reco = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "inteference_reco_read_knownword",
              "n_expo": 1,
              "test": {"test":[1]},
              "liste": stimuli_reading}

exp_reading_reco = expe(simu=sim_reading_reco, **expe_param_reading_reco)


# simulation for auditory known words
sim_spelling_dl = simu_definition(input_type="auditory", remove_stim_ortho=False, remove_stim_phono=False, simu_type="spelling_H", criterion_type="ld", sd=1.75)

expe_param_spelling_dl = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "inteference_dl_spell_knownword",
              "n_expo": 1,
              "test": {"test":[1]},
              "liste": stimuli_spelling}

exp_spelling_dl = expe(simu=sim_spelling_dl, **expe_param_spelling_dl)

# simulation for auditory known words
sim_spelling_reco = simu_definition(input_type="auditory", remove_stim_ortho=False, remove_stim_phono=False, simu_type="spelling_H", criterion_type="reco", sd=1.75)

expe_param_spelling_reco = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "inteference_reco_spell_knownword",
              "n_expo": 1,
              "test": {"test":[1]},
              "liste": stimuli_spelling}

exp_spelling_reco = expe(simu=sim_spelling_reco, **expe_param_spelling_reco)

#######################
### E - Simulations ###
#######################

exp_reading_dl.compare_param()
exp_spelling_dl.compare_param()
exp_reading_reco.compare_param()
exp_spelling_reco.compare_param()


print('end')