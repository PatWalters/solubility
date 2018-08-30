#!/usr/bin/env python

import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import namedtuple


class ESOLCalculator:
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol_orig(self, mol):
        """
        Original parameters from the Delaney paper, just here for comparison
        :param mol: input molecule
        :return: predicted solubility
        """
        # just here as a reference don't use this!
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

    def calc_esol(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = -0.01216474
        coef = {"logp": -0.65685286, "mw": -0.00507685, "rotors": -0.01468901, "ap": -0.82973045}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol


def test_on_dls_100():
    """
    Test on the dls_100 dataset from the University of St Andrews
    https://doi.org/10.17630/3a3a5abc-8458-4924-8e6c-b804347605e8
    This function downloads the Excel spreadhsheet from St Andrews, does the
    :return: None
    """
    esol_calculator = ESOLCalculator()
    df = pd.read_excel("https://risweb.st-andrews.ac.uk/portal/files/251452157/dls_100.xlsx")
    df.dropna(inplace=True)
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
    res = []
    for mol, name, logS in df[["Molecule", "Chemical name", "LogS exp (mol/L)"]].values:
        res.append([name, logS, esol_calculator.calc_esol(mol)])
    df = pd.DataFrame(res, columns=["Name", "Experimental LogS", "ESol LogS"])
    df.to_csv("dls_100.csv", index=False)


def refit_esol():
    """
    Refit the parameters for ESOL using multiple linear regression
    :return: None
    """
    esol_calculator = ESOLCalculator()
    df = pd.read_csv("delaney.csv")
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
    result_list = []
    for name, mol in df[["Compound ID", "Molecule"]].values:
        result_list.append([name] + list(esol_calculator.calc_esol_descriptors(mol)))
    result_df = pd.DataFrame(result_list)
    descriptor_cols = ["LogP", "MW", "Rotors", "AP"]
    result_df.columns = ["Compound ID"] + descriptor_cols
    df = df.merge(result_df, on="Compound ID")
    x = df[descriptor_cols]
    y = df[["ESOL predicted log(solubility:mol/L)"]]

    model = LinearRegression()
    model.fit(x, y)
    coefficient_dict = {}
    for name, coef in zip(descriptor_cols, model.coef_[0]):
        coefficient_dict[name.lower()] = coef
    print("Intercept = ", model.intercept_[0])
    print("Coefficients =", coefficient_dict)


def demo():
    """
    Read the csv file from the Delaney supporting information, calculate ESOL using the data file from the
    supporting material using the original and refit coefficients.  Write a csv file comparing with experiment.
    :return:
    """
    esol_calculator = ESOLCalculator()
    df = pd.read_csv("delaney.csv")
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
    res = []
    for mol, val in df[["Molecule", "ESOL predicted log(solubility:mol/L)"]].values:
        res.append([val, esol_calculator.calc_esol(mol), esol_calculator.calc_esol_orig(mol)])
    output_df = pd.DataFrame(res, columns=["Experiment", "ESOL Current", "ESOL Original"])
    output_df.to_csv('validation.csv',index=False)


if __name__ == "__main__":
    test_on_dls_100()
