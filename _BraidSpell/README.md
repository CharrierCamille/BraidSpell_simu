# BraidPy
Python implementation of BRAID-Spell model.
"Bayesian word Recognition with Attention, Interference and Dynamics, with Spelling skills".

Reading, Spelling.
Visual and auditory word recognition and lexical Decision.

## Installation of the python package (optional)

In the root directory, run the following command :

```python
python setup.py install
```

If you want to modify the source code, use instead :

```python
python setup.py develop
```
## Dependencies (only if you don't use Conda)

Here is the needed packages to run the model and to plot the results:
install using Conda (either through anaconda GUI or command-line)

| Required 	| For fast  computation |        Data viz(optional) | 
| :------------ | -------------------   | ------------------------  |
| Numpy / Pandas|   Numba  	   	| mpl_toolkits		    |
| Matplotlib    |   Dask    		| bokeh			    |
| Seaborn       |   Joblib      	| 			    |

Useful packages but not required: 
unidecode,  Sklearn, StatsModels

## Conda Environment  
  
If you want to directly use the conda environment with the good Python/libraries version, you can use the environment file braid_env_18092023_minimal.yml
  
To create the braid environment, run the following command (after installing conda) in the braid root directory : 
```python
conda env create -f env/braid_env_18092023_minimal.yml
   
```
It will create a new environment named braid_env_1809_minimal.
  
To go in this environment, run the following command : 
```python
conda activate braid_env_1809_minimal
```

There is still some libraries to install :
```python
pip3 install Levenshtein sklearn 
```

If you want to create your own yml file, run the following command in the env/ directory: 
```python
./export_env.sh filename.yml
```

## Current Features
The current implementation performs simulations of:
- Letter and Phoneme perception 
- Visual and Auditory Word recognition
- Orthographic and Phonological Lexical Decision
- Word reading (decoding)
- Novel Word Acquisition
- Spelling

## Code architecture

There are several main classes :
- the "simu" class, which makes specific simulations (i.e. tasks) over a single word using the BRAID model and stores the result. It contains one inner class, the "braid" class.
- the "braid" class, which is the main model class. It contains 2 inner classes, the "ortho" and "phono" classes, that are derived from the "modality" class.
- the "modality" class, which implements calculations specific to one modality. It contains several inner classes, that correspond to the BRAID-Acq submodels : "sensor", "percept", "word", "lexical", "attention".
- the "semantic" class, which implements calculations related to the semantic context.

Aditionally, the "expe" class can be used to run simulations on word lists, and compare several model settings to design experiments.

The architecture of the code is illustrated on the following UML diagram:


![Architecture of the BRAID-Acq code](UML.png)

For more information, see the full documentation of the code.

## Usage

To learn how to conduct simulations with the model BRAID-Acq, look at the notebooks in the directory notebooks/ :


* command_line.ipynb : learn to simulate easily the reading of a single word using the command-line.
* first_word.ipynb : learn to write a Python code to configurate the model and simulate the reading or spelling of one word.
* first_simulations.ipynb : learn how to simulate an entire experiment (using the "expe" class) on several words. Shows the example of simulating the effect of length and lexicality.


## Class documentation

The class documentation is accessible at: build/html/modules.html.

If you modified the code and want to re-generate the doc, use the following command in the root directory:
```bash
make html
```

If you want to generate the doc from the start, you can look at the sphinx and auto-doc documentation documentation. You can also use these steps:

1. If they exist, delete source/ and build/ directories.
2. Initialize the documentation by the following command in the root directory:

```bash
sphinx-quickstart
```
3. Add the autodoc extension in the file conf.py in the source/ directory:

```bash
extensions = ['sphinx.ext.autodoc']
```
4. Automatically find modules in code

```bash
sphinx-apidoc -o source/ ../braidpy
```


5. Type the following command in the root directory:
```bash
make html
```


## Contact

email  
Camille Charrier: camille.charrier@univ-grenoble-alpes.fr
Alexandra Steinhilber: alexandra.st@free.fr / alexandra.steinhilber@univ-grenoble-alpes.fr  
Julien Diard: julien.diard@univ-grenoble-alpes.fr
