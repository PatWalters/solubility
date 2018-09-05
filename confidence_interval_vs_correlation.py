#!/usr/bin/env python

from evaluate_results import pearson_confidence
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np

plt.switch_backend("agg")


# Generate a figure showing the relationship between correlation, number of datapoints and
# the confidence interval


def generate_points(r_squared, min_val=5, max_val=100):
    """
    Generate points for an upper and lower confidence interval curve for a given Pearson R**2
    :param r_squared: Pearson R**2
    :param min_val: Minimum x value
    :param max_val: Maximum x value
    :return: arrays with x, lower bound, and upper bound
    """
    r = sqrt(r_squared)
    x_list = []
    lb_list = []
    ub_list = []
    for i in range(min_val, max_val):
        x_list.append(i)
        lb, ub = pearson_confidence(r, i)
        lb_list.append(lb)
        ub_list.append(ub)
    return np.array(x_list), np.array(lb_list) ** 2, np.array(ub_list) ** 2


def generate_confidence_plot(p, r_squared, x_lab=False, y_lab=False):
    """
    Generate a plot of curves for confidence interval upper and lower bound with a colored
    region between the two curves
    :param p: pyplot object
    :param r_squared: Pearson R**2
    :param x_lab: label the x axis?
    :param y_lab: label the y axis?
    :return:
    """
    x, y_lb, y_ub = generate_points(r_squared)
    p.title(r"R$^2$ = %0.1f" % r_squared)
    if x_lab:
        p.xlabel("Number of Points")
    if y_lab:
        p.ylabel("Pearson R$^2$")
    p.plot(x, y_lb, color="black")
    p.plot(x, y_ub, color="black")
    p.fill_between(x, y_lb, y_ub, color="lightgray")
    p.grid(True)


def main():
    plt.figure(1)
    plt.subplot(221)
    generate_confidence_plot(plt, 0.6, y_lab=True)
    plt.subplot(222)
    generate_confidence_plot(plt, 0.7)
    plt.subplot(223)
    generate_confidence_plot(plt, 0.8, x_lab=True, y_lab=True)
    plt.subplot(224)
    generate_confidence_plot(plt, 0.9, x_lab=True)

    plt.tight_layout()

    outfile_name = "confidence.png"
    plt.savefig(outfile_name)
    print("wrote", outfile_name)


main()
