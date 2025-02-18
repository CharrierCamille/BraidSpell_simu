######################
### A - Imports ######
######################
import pdb

braidPath = "../../../../_BraidSpell/"
import sys

sys.path.append(braidPath)

from braidpy.simu import simu
from braidpy.expe import expe

# specific libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(seed=231024)


#######################
### B - Model init ####
#######################

def simu_definition(input_type="visual", remove_stim_ortho=False, remove_stim_phono=False, stim='partir',
                    simu_type="normal", criterion_type="other_mod", sd=1):
    if input_type == "visual":
        ortho_param = {"learning": False, "remove_stim": remove_stim_ortho, 'stim': stim, 'sd': sd, 'Q': 1,
                       "decoding_ratio_po": 0.001}
        phono_param = {"learning": False, "remove_stim": remove_stim_phono, 'sd': sd, 'Q': 1,
                       "decoding_ratio_op": 0.006}
        if criterion_type == "this_mod":
            criterion = "pMean"
        else:
            criterion = "phiMean"
    elif input_type == "auditory":
        ortho_param = {"learning": False, "remove_stim": remove_stim_ortho, 'sd': sd, "decoding_ratio_po": 0.006}
        phono_param = {"learning": False, "remove_stim": remove_stim_phono, 'sd': sd, 'stim': stim, "decoding_ratio_op": 0.001}
        if criterion_type == "this_mod":
            criterion = "phiMean"
        else:
            criterion = "pMean"
    if criterion_type == "both":
        criterion = "bothMean"
    simu_param = {"level": "expe", "max_iter": 5000, "simu_type": simu_type,
                  "stop_criterion_type": criterion, "sampling": True}
    simu_args = {}
    model_param = {"langue": "fr", "path": braidPath, "input_type": input_type}

    semantic_param = {"context_sem": False}
    return simu(model_param, ortho_param, phono_param, semantic_param, simu_args, **simu_param)


#########################
### C - Param definition ####
#########################

stimuli_reading = pd.read_csv("01_Rstim_lengthcalib.csv", index_col=0).reset_index().word.tolist()
stimuli_spelling = pd.read_csv("01_Sstim_lengthcalib.csv", index_col=0).reset_index().pron.tolist()

# stimuli_reading = ["coût", "crampon", "fort", "geai", "hier", "lord", "ours", "parfum", "pouah", "sage",
#                    "session", "soir", "soupçon", "tien", "traînée", "érable"]
#
# stimuli_spelling = ["5fEkt", "Rwajal", "ZyRe", "akORde", "apsoly", "b&de", "balE", "byte", "dZin",
#                     "diREkt", "efOR", "epuv@te", "fele", "k&tRa", "k&tuRne", "k&vwa", "kORdjal", "pRete", "paRe",
#                     "petaR", "sepaRe", "tR@ble", "tRuble", "yile"]

stimuli_reading = [f'!{word}!' for word in stimuli_reading]
stimuli_spelling = [f'!{word}!' for word in stimuli_spelling]


decoding_ratio_ff = [0.006]
decoding_ratio_fb = [0.003]

n=30

# top_down = [i for i in np.random.normal(5e-4, 5e-4, n) if i > 0]
#
# while len(top_down) < n:
#     i = np.random.normal(5e-4, 5e-4, 1)[0]
#     if i > 0:
#         top_down.append(i)
#
# print(top_down)

# decoding_ratio_ff = [i for i in np.random.normal(5e-3, 2e-3, n) if i > 0]
#
# while len(decoding_ratio_ff) < n:
#     i = np.random.normal(5e-3, 2e-3, 1)[0]
#     if i > 0:
#         decoding_ratio_ff.append(i)
#
# print(decoding_ratio_ff)
#


decoding_ratio_fb = [i for i in np.random.normal(0.0025, 0.001, n) if i > 0]

while len(decoding_ratio_fb) < n:
    i = np.random.normal(0.0025, 0.001, 1)[0]
    if i > 0:
        decoding_ratio_fb.append(i)

print(decoding_ratio_fb)


######################
### D - Expe init ####
######################

# simulation for visual known words
sim_reading = simu_definition(input_type="visual", remove_stim_ortho=False, remove_stim_phono=False,
                                  simu_type="H", criterion_type="both", sd=1)

expe_param_reading_w = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "01_length_fb_wr",
              "n_expo": 1,

              "test": {"decoding_ratio_po": decoding_ratio_fb},
              "liste": stimuli_reading}

exp_reading_w = expe(simu=sim_reading, **expe_param_reading_w)
exp_reading_w.compare_param()

# simulation for visual known words
sim_reading_pw = simu_definition(input_type="visual", remove_stim_ortho=True, remove_stim_phono=True,
                                  simu_type="H", criterion_type="both", sd=1)

expe_param_reading_pw = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "01_length_fb_pwr",
              "n_expo": 1,

              "test": {"decoding_ratio_po": decoding_ratio_fb},
              "liste": stimuli_reading}

exp_reading_pw = expe(simu=sim_reading_pw, **expe_param_reading_pw)
exp_reading_pw.compare_param()

sim_spelling_w = simu_definition(input_type="auditory", remove_stim_ortho=False, remove_stim_phono=False,
                                  simu_type="spelling_H", criterion_type="both")

expe_param_spelling_w = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "01_length_fb_ws",
              "n_expo": 1,
              "test": {"decoding_ratio_op": decoding_ratio_fb},
              "liste": stimuli_spelling}

exp_spelling_w = expe(simu=sim_spelling_w, **expe_param_spelling_w)
exp_spelling_w.compare_param()

sim_spelling_pw = simu_definition(input_type="auditory", remove_stim_ortho=True, remove_stim_phono=True,
                                  simu_type="spelling_H", criterion_type="both")

expe_param_spelling_pw = {"res_fct_name": ["t_tot", "ld_ortho", "ld_phono", "phi", "let","simu_time"],
              "basename": "01_length_fb_pws",
              "n_expo": 1,
              "test": {"decoding_ratio_op": decoding_ratio_fb},
              "liste": stimuli_spelling}

exp_spelling_pw = expe(simu=sim_spelling_pw, **expe_param_spelling_pw)
exp_spelling_pw.compare_param()


