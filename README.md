# solubility

An implementation of Delaney's ESOL method using the RDKit
```
ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure 
John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
https://pubs.acs.org/doi/abs/10.1021/ci034243x
```

There are 4 scripts in this repo. 

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

* confidence_interval_vs_correlation.py - generates a plot showing the relationship between correlation, number of
datapoints and confidence intervals. Same requirements as evaluate_results.py.

The repo also contains 2 datafiles. 

* delaney.csv - a csv file with data from the supporting material for [https://pubs.acs.org/doi/abs/10.1021/ci034243x](https://pubs.acs.org/doi/abs/10.1021/ci034243x)
* dls_100_unique.csv - csv file with solubility data for 56 compounds not in the delaney.csv.  Note that the original file from the
University of St Andrews [https://doi.org/10.17630/3a3a5abc-8458-4924-8e6c-b804347605e8](https://doi.org/10.17630/3a3a5abc-8458-4924-8e6c-b804347605e8) 
has 100 compounds, 44 of which are duplicated in delaney.csv.   
 

