# solubility

An implementation of Delaney's ESOL method using the RDKit
```
ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure 
John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
https://pubs.acs.org/doi/abs/10.1021/ci034243x
```

There are 3 scripts in this repo. 

* esol.py - has the routines to calculate ESOL and few associated utility functions.
Depends on the RDKit, pandas and scikit learn, all of which are part of the RDKit distribution.
See the [RDKit site](https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md) for installation instructions. 

* solubility_comparison.py - compares ESOL calculation with 3 models calculated by DeepChem (Random Forest, Weave, 
Graph Convolution).  Requires a DeepChem installation. 
See the [DeepChem](https://github.com/deepchem/deepchem) site for installation instructions.

* evaluate_results.py - compares results calculated in solubility_comparison.py and generates a couple of plots. Requires matplotlib and and scipy. 
```
pip install matplotlib scipy
```
