"""Contains functions that plot mAP@15 for individual labels and label families."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

from .utils import save_function
from ..directories import EVAL_DIR, DATASET_NAME

# TODO: find a way to wrap the text better
def plot_map_at_15_for_all_labels(model_variation_search, 
                               eval_dir=EVAL_DIR, dataset_name=DATASET_NAME,
                               use_fig_name=True, fig_name="",
                               save_fig=False, save_dir=""):
    """Takes a model name, aggregation variation and search name and plots 
    the label-based mAP@15 for it. That is, the mAP@15 is plotted for each 
    individual label."""

    model, variation, search = model_variation_search

    # Determine the figure name
    if search=="dot":
        _search = "Dot Product"
    elif search=="nn":
        _search = "Nearest Neighbors"
    default_fig_name = f"{model} mAP@15 Values for All Labels\n" \
                        f"({variation} Processing, {_search} Search)"
    fig_name = fig_name if fig_name else default_fig_name

    # Get the path to the label-based mAP@15
    variation_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
    map_path = os.path.join(variation_dir, search, "labels_mAP@15.csv")
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
    fig, ax = plt.subplots(figsize=(18, 26), nrows=N, constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=16)
    for i in range(N):
        ax[i].bar([label.replace("_","\n") for label,_ in label_aps[i*delta:(i+1)*delta]], 
                  [prec for _,prec in label_aps[i*delta:(i+1)*delta]])
        ax[i].set_yticks(np.arange(0, 1.05, 0.2))
        ax[i].grid()
        ax[i].set_ylim([0, 1])
        ax[i].set_xlim([-0.5, len(label_aps[i*delta:(i+1)*delta])-0.5])
        ax[i].set_ylabel("mAP@15")

    save_function(save_fig, save_dir, "label_based_mAP@15.png", fig)
    plt.show()

def plot_map_at_N_for_families(model_variation_search, N=15,
                                eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                use_fig_name=True, fig_name="", 
                                save_fig=False, save_dir=""):
    """Takes a model name, aggregation variation and search name and plots the 
    label_family-based mAP@N for it. That is, the mAP@N is plotted for each 
    individual label family."""

    model, variation, search = model_variation_search

    # Determine the figure name
    if search=="dot":
        _search = "Dot Product"
    elif search=="nn":
        _search = "Nearest Neighbors"
    default_fig_name = f"{model} mAP@{N} Values for Label Families\n" \
                        f"({variation} Processing, {_search} Search)"
    fig_name = fig_name if fig_name else default_fig_name

    # Get the path to the family-based mAP@N
    variation_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
    map_path = os.path.join(variation_dir, search, f"families_mAP@{N}.csv")

    # Read the family-based mAP@N
    labels_map = pd.read_csv(map_path)
    # Get the families and maps
    families = labels_map["family"].to_list()
    maps = labels_map["map"].to_list()
    # Sort the labels and maps, they should be ordered already but just in case
    family_aps = sorted(zip(families, maps), key=lambda x: x[1], reverse=True)

    # Plot the family-based mAP@N
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.bar([f.replace("_", " ").title() for f,_ in family_aps], 
           [m for _,m in family_aps], 
           edgecolor='k')
    for j, m in enumerate([m for _,m in family_aps]):
        ax.text(j, 
                m+0.01, 
                f"{m:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=11, 
                weight='bold',
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.10))
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.set_xlabel("Sound Families", fontsize=15)
    ax.set_ylabel(f"mAP@{N} (↑)", fontsize=15)
    ax.set_ylim([0,1])
    ax.grid()

    save_function(save_fig, save_dir, f"family_based_mAP@{N}.png", fig)
    plt.show()

def plot_map_at_15_150_for_families(model_variation_search,
                                eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                use_fig_name=True, fig_name="", 
                                save_fig=False, save_dir=""):

    model, variation, search = model_variation_search

    # Determine the figure name
    if search=="dot":
        _search = "Dot Product"
    elif search=="nn":
        _search = "Nearest Neighbors"
    default_fig_name = f"{model} mAP@15 and mAP@150 Values for Label Families\n" \
                        f"({variation} Processing, {_search} Search)"
    fig_name = fig_name if fig_name else default_fig_name

    # Get the path to the family-based mAP@N
    variation_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)

    maps_N = {}
    for N in [15, 150]:
        map_path = os.path.join(variation_dir, search, f"families_mAP@{N}.csv")

        # Read the family-based mAP@N
        labels_map = pd.read_csv(map_path)
        # Get the families and maps
        families = labels_map["family"].to_list()
        maps = labels_map["map"].to_list()
        # Sort the labels and maps, they should be ordered already but just in case
        family_aps = sorted(zip(families, maps), key=lambda x: x[1], reverse=True)
        maps_N[N] = family_aps

    # Re-sort mAP150 to match the keys of mAP15
    _map150 = []
    for (f,_) in maps_N[15]:
        _map150.append((f, [m for (f_,m) in maps_N[150] if f_==f][0]))
    maps_N[150] = _map150

    # Plot the family-based mAP@N
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    for N, family_aps in maps_N.items():
        for i, (_, m) in enumerate(family_aps):
            loc = i - 0.2 if N==15 else i + 0.2
            ax.bar(loc,
                    m,
                    edgecolor='k',
                    color=COLORS[i],
                    hatch="//" if N==150 else None,
                    width=0.3,)
            ax.text(loc, 
                    m+0.01, 
                    f"{m:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=11, 
                    weight='bold',
                    )
    ax.set_xticks(np.arange(len(family_aps)), [f.replace("_", " ").title() for f,_ in maps_N[15]])

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.10))
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.set_xlabel("Sound Families", fontsize=15)
    ax.set_ylabel("mAP@N (↑)", fontsize=15)
    ax.set_ylim([0,1])
    ax.grid()

    save_function(save_fig, save_dir, f"family_based_mAP@15_mAP@150{N}.png", fig)
    plt.show()    