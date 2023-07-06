"""Contains functions that plot mAP@15 for individual labels and label families."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

def plot_label_based_map_at_15(model_variation_search, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name, aggregation variation and search name and plots the label-based mAP@15 for it.
    That is, the mAP@15 is plotted for each individual label."""

    model, variation, search = model_variation_search
    default_fig_name = f"Model: {model} Aggregation: {variation} Search: {search}\n"\
                            f"{dataset_name} Labels mAP@15 "
    fig_name = fig_name if fig_name else default_fig_name

    # Get the path to the label-based mAP@15
    variation_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
    map_path = os.path.join(variation_dir, search, "labels_mAP@15.csv")
    # embedding_search = variation + "-" + search

    # Read the label-based mAP@15
    labels_map = pd.read_csv(map_path)
    # Get the labels and maps
    labels = labels_map["label"].to_list()
    maps = labels_map["map@15"].to_list()
    # Sort the labels and maps, they should be ordered already but just in case
    label_aps = sorted(zip(labels, maps), key=lambda x: x[1], reverse=True)

    # Determine some plot parameters
    N = 10 # Number of rows
    delta = len(maps) // N

    # Plot the label-based mAP@15
    fig, ax = plt.subplots(figsize=(18, 24), nrows=N, constrained_layout=True)
    fig.suptitle(fig_name, fontsize=16)
    for i in range(N):
        ax[i].bar([label.replace("_","\n") for label,_ in label_aps[i*delta:(i+1)*delta]], 
                  [prec for _,prec in label_aps[i*delta:(i+1)*delta]])
        ax[i].set_yticks(np.arange(0, 1.05, 0.2))
        ax[i].grid()
        ax[i].set_ylim([0, 1.05])
        ax[i].set_ylabel("mAP@15")
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "label_based_mAP@15.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        plt.close()

def plot_family_based_map_at_15(model_variation_search, eval_dir, dataset_name="FSD50K.eval_audio", fig_name="", save_fig=False, save_dir=""):
    """Takes a model name, aggregation variation and search name and plots the label_family-based mAP@15 for it.
    That is, the mAP@15 is plotted for each individual label family."""

    model, variation, search = model_variation_search

    default_fig_name = f"Model: {model} Aggregation: {variation} Search: {search}\n"\
                            f"{dataset_name} Label Families mAP@15 "
    fig_name = fig_name if fig_name else default_fig_name

    # Get the path to the family-based mAP@15
    variation_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
    map_path = os.path.join(variation_dir, search, "families_mAP@15.csv")
    #embedding_search = variation + "-" + search

    # Read the family-based mAP@15
    labels_map = pd.read_csv(map_path)
    # Get the families and maps
    families = labels_map["family"].to_list()
    maps = labels_map["map"].to_list()
    # Sort the labels and maps, they should be ordered already but just in case
    family_aps = sorted(zip(families, maps), key=lambda x: x[1], reverse=True)

    # Plot the family-based mAP@15
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("Page 1 Results", fontsize=15)
    ax.bar([f.replace("_", " ").title() for f,_ in family_aps], 
           [m for _,m in family_aps], 
           edgecolor='k')

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("Label Families", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_ylim([0,1])
    ax.grid()

    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "family_based_mAP@15.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        plt.close()