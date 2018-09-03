#!/usr/bin/env python

import pandas as pd
from scipy.stats import norm, pearsonr
import math
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def pearson_confidence(r, num, interval=0.95):
    """
    Calculate upper and lower 95% CI for a Pearson r (not R**2)
    Inspired by https://stats.stackexchange.com/questions/18887
    :param r: Pearson's R
    :param num: number of data points
    :param interval: confidence interval (0-1.0)
    :return: lower bound, upper bound
    """
    stderr = 1.0 / math.sqrt(num - 3)
    z_score = norm.ppf(interval)
    delta = z_score * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return lower, upper


def calc_pearson(pred, truth):
    """
    Calculate a Pearson R**2 + confidence intervals for a set of predicted and true values
    :param pred: predicted values
    :param truth: true (experimental values)
    :return: Pearson R**2, lower and upper 95% confidence intervals
    """
    pearson_r_val = pearsonr(truth, pred)[0]
    lower, upper = pearson_confidence(pearson_r_val, len(pred))
    return [x ** 2 for x in [pearson_r_val, lower, upper]]


def eval_results(infile_name):
    """
    Read a csv file with solubility results
    Columns are SMILES, Name, True values, all subsequent columns are predictions
    Generates png files with R**2 and R**2 + confidence intervals
    :param infile_name: input file name
    :return: None
    """
    df = pd.read_csv(infile_name)
    truth_col = df.columns[2]
    res = []
    for col in df.columns[3:]:
        r, lower, upper = calc_pearson(df[truth_col].values, df[col].values)
        res.append([col, r, lower, upper])
    df = pd.DataFrame(res, columns=["Method", "R**2", "Lower", "Upper"])
    print(df)
    draw_histogram(df, "Solubility Comparison", "solubility_comparison.png", "Method", "R**2")
    draw_histogram(df, "Solubility Comparison", "solubility_comparison_error.png", "Method", "R**2", "Lower", "Upper")


def draw_histogram(input_df, title, outfile_name, name_col, val_col, lb_col=None, ub_col=None):
    """
    Draw a histogram, with error bars
    :param input_df: input dataframe
    :param title: Title for the plot
    :param outfile_name: output file name
    :param name_col: name to put on the x-axis
    :param val_col: value column to plot as the histogram
    :param ub_col: upper bound column
    :param lb_col: lower bound column
    :return: None
    """
    fig, ax = plt.subplots()

    xlab = input_df[name_col]
    y_val = input_df[val_col]
    if ub_col and lb_col:
        lb = input_df[lb_col].values
        ub = input_df[ub_col].values
        error = ub - lb
    else:
        error = None
    x_pos = np.arange(len(xlab))

    ax.set_title(title)
    ax.bar(x_pos, y_val, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(val_col)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlab)
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(outfile_name)


if __name__ == "__main__":
    eval_results("solubility_comparison.csv")
