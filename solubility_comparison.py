#!/usr/bin/env python
import sys
import os
import deepchem
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models.sklearn_models import RandomForestRegressor, SklearnModel
import pandas as pd
from rdkit import Chem
import itertools
from esol import ESOLCalculator

# Turn off TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# -------- Utility Functions -----------------------------------#


def featurize_data(tasks, featurizer, normalize, dataset_file):
    loader = deepchem.data.CSVLoader(tasks=tasks, smiles_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    move_mean = True
    if normalize:
        transformers = [deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)]
    else:
        transformers = []
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset, featurizer, transformers


def generate_prediction(input_file_name, model, featurizer, transformers):
    df = pd.read_csv(input_file_name)
    mol_list = [Chem.MolFromSmiles(x) for x in df.SMILES]
    val_feats = featurizer.featurize(mol_list)
    res = model.predict_on_batch(val_feats, transformers)
    # kind of a hack
    # seems like some models return a list of lists and others (e.g. RF) return a list
    # check to see if the first element in the returned array is a list, if so, flatten the list
    if type(res[0]) is list:
        df["pred_vals"] = list(itertools.chain.from_iterable(*res))
    else:
        df["pred_vals"] = res
    return df


# ----------- Model Generator Functions --------------------------#


def generate_graph_conv_model():
    batch_size = 128
    model = GraphConvModel(1, batch_size=batch_size, mode='regression')
    return model


def generate_weave_model():
    batch_size = 64
    model = WeaveModel(1, batch_size=batch_size, learning_rate=1e-3, use_queue=False, mode='regression')
    return model


def generate_rf_model():
    model_dir = "."
    sklearn_model = RandomForestRegressor(n_estimators=500)
    return SklearnModel(sklearn_model, model_dir)


# ---------------- Function to Run Models ----------------------#


def run_model(model_func, task_list, featurizer, normalize, training_file_name, validation_file_name, nb_epoch):
    dataset, featurizer, transformers = featurize_data(task_list, featurizer, normalize, training_file_name)
    model = model_func()
    if nb_epoch > 0:
        model.fit(dataset, nb_epoch)
    else:
        model.fit(dataset)
    pred_df = generate_prediction(validation_file_name, model, featurizer, transformers)
    return pred_df


# ------------------ Function to Calculate ESOL ----------------------*

def calc_esol(input_file_name, smiles_col="SMILES"):
    df = pd.read_csv(input_file_name)
    esol_calculator = ESOLCalculator()
    res = []
    for smi in df[smiles_col].values:
        mol = Chem.MolFromSmiles(smi)
        res.append(esol_calculator.calc_esol(mol))
    df["pred_vals"] = res
    return df


# ----------------- main ---------------------------------------------*
def main():
    training_file_name = "delaney.csv"
    validation_file_name = "dls_100_unique.csv"
    task_list = ['measured log(solubility:mol/L)']

    print("=====ESOL=====")
    esol_df = calc_esol(validation_file_name)

    print("=====Random Forest=====")
    featurizer = deepchem.feat.fingerprints.CircularFingerprint(size=1024)
    model_func = generate_rf_model
    rf_df = run_model(model_func, task_list, featurizer, False, training_file_name, validation_file_name, nb_epoch=-1)

    print("=====Weave======")
    featurizer = deepchem.feat.WeaveFeaturizer()
    model_func = generate_weave_model
    weave_df = run_model(model_func, task_list, featurizer, True, training_file_name, validation_file_name, nb_epoch=30)

    print("=====Graph Convolution=====")
    featurizer = deepchem.feat.ConvMolFeaturizer()
    model_func = generate_graph_conv_model
    gc_df = run_model(model_func, task_list, featurizer, True, training_file_name, validation_file_name, nb_epoch=20)

    output_df = pd.DataFrame(rf_df[["SMILES", "Chemical name", "LogS exp (mol/L)"]])
    output_df["ESOL"] = esol_df["pred_vals"]
    output_df["RF"] = rf_df["pred_vals"]
    output_df["Weave"] = weave_df["pred_vals"]
    output_df["GC"] = gc_df["pred_vals"]
    output_df.to_csv("solubility_comparison.csv", index=False, float_format="%0.2f")


if __name__ == "__main__":
    main()
