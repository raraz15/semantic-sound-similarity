""" This module contains functions for plotting the results of multiple models.
For each metric (MR1, mAP@k, etc.) there is a function that takes a list of [embeddings,search] 
and plots the results in the same plot for comparison.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

from ..directories import EVAL_DIR

DATASET_NAME = "FSD50K.eval_audio"

# TODO: how to encode variation and the search?

####################################################################################################
# mAP

def plot_micro_map_comparisons_multimodel(models, eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the Micro Averaged mAP@k in the same figure."""

    default_fig_name = f"Embedding Performances using Instance-Based mAP@15 (Micro-Averaged) on {dataset_name}"

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model, variation, search in models:
        map_path = os.path.join(eval_dir, dataset_name, model+"-"+variation, search, "micro_mAP@15.txt")
        with open(map_path, "r") as in_f:
            micro_map_at_15 = float(in_f.read())
        maps.append((model, variation, search, micro_map_at_15))

    # Plot the micro-mAP for each model
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("Page 1 Results", fontsize=15)
    for j,(model,variation,search,micro_map_at_15) in enumerate(maps):
        ax.bar(positions[j], 
                micro_map_at_15, 
                label=model,
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(positions[j], 
                micro_map_at_15+0.01, 
                f"{micro_map_at_15:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold'
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("Embedding, Search Combinations", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15) # TODO: change name?
    ax.set_ylim([0,1])
    ax.grid()
    ax.legend(loc="best", fontsize=10, title_fontsize=11, fancybox=True)
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "best_embeddings-micro_mAP@15-comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        txt_path = os.path.splitext(fig_path)[0]+".txt"
        with open(txt_path, "w") as infile:
            for model in models:
                infile.write(f"{model[0]}-{model[1]}\n")
    plt.show()

def plot_macro_map_comparisons_multimodel(models, eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the Macro Averaged mAP@15 in the same figure."""

    default_fig_name = f"Embedding Performances using Label-Based mAP@15 (Macro-Averaged) on {dataset_name}"

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model, variation, search in models:
        map_path = os.path.join(eval_dir, dataset_name, model+"-"+variation, search, "balanced_mAP@15.txt")
        with open(map_path, "r") as in_f:
            balanced_map_at_15 = float(in_f.read())
        maps.append((model, variation, search, balanced_map_at_15))

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("Page 1 Results", fontsize=15)
    for j,(model,variation,search,balanced_map_at_15) in enumerate(maps):
        ax.bar(positions[j], 
                balanced_map_at_15, 
                label=model,
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(positions[j], 
                balanced_map_at_15+0.01, 
                f"{balanced_map_at_15:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold'
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("Embedding, Search Combinations", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15) # TODO: change name?
    ax.set_ylim([0,1])
    ax.grid()
    ax.legend(loc="best", fontsize=10, title_fontsize=11, fancybox=True)
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "best_embeddings-macro_mAP@15-comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        txt_path = os.path.splitext(fig_path)[0]+".txt"
        with open(txt_path, "w") as infile:
            for model in models:
                infile.write(f"{model[0]}-{model[1]}\n")
    plt.show()

def plot_family_map_multimodel(models, eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the Macro Averaged mAP@15 in the same figure."""

    default_fig_name = f"Embedding-Search Performances over Label-Families Averaged mAP@15 on {dataset_name}"

    # Read the mAP for each model
    model_maps = defaultdict(list)
    for model, variation, search in models:
        map_path = os.path.join(eval_dir, dataset_name, model+"-"+variation, search, "families_mAP@15.csv")
        labels_map = pd.read_csv(map_path)
        families = labels_map["family"].to_list()
        maps = labels_map["map"].to_list()
        for family, family_map in zip(families, maps):
            model_maps[family].append((model, variation, search, family_map))
    
    fig,ax = plt.subplots(nrows=len(model_maps) ,figsize=(18,12), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    for i, (family, family_aps) in enumerate(model_maps.items()):
        for j,(model,variation,search,ap) in enumerate(family_aps):
            ax[i].bar(j, 
                    ap, 
                    label=model,
                    width=0.8, 
                    color=COLORS[j], 
                    edgecolor='k'
                    )
            ax[i].text(j, 
                    ap+0.01, 
                    f"{ap:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10, 
                    weight='bold'
                    )

        # Set the plot parameters
        ax[i].set_title(family.replace("_", " ").title(), fontsize=15)
        ax[i].set_yticks(np.arange(0,1.05,0.1))
        ax[i].tick_params(axis='x', which='major', labelsize=0)
        ax[i].tick_params(axis='y', which='major', labelsize=11)
        ax[i].set_ylabel("mAP@15 (↑)", fontsize=15)
        ax[i].set_ylim([0,1])
        ax[i].grid()
        if i==0:
            ax[i].legend(loc="upper center", fontsize=10, title_fontsize=11, fancybox=True, ncol=len(models))

    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "family_based_mAP@15-comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        plt.close()
    plt.show()


####################################################################################################
# MR1

def plot_mr1_comparisons_multimodel(models, eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Read the MR1s for each model
    mr1s = []
    for model, variation, search in models:
        mr1_path = os.path.join(eval_dir, dataset_name, model+"-"+variation, search, "MR1.txt")
        with open(mr1_path,"r") as infile:
            mr1 = float(infile.read())
        mr1s.append((model, variation, search, mr1))

    # Plot the MR1s
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else f"Embedding Performances using MR1 Evaluated on {dataset_name} Set"
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    #ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    for i,(model,search,mr1) in enumerate(mr1s):
            ax.bar(i, 
                mr1, 
                label=model,
                width=0.85, 
                color=COLORS[i], 
                edgecolor='k'
                )
            ax.text(i, 
                mr1+0.01, 
                f"{mr1:.2f}", 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                weight='bold'
                )

    # Set the plot parameters
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_yticks(np.arange(0,max([m[2] for m in mr1s])+1.0,0.5))
    ax.set_ylabel("MR1@90 (↓)", fontsize=15)
    #ax.set_title(models[0][0].split("-")[-1].replace("_"," "), fontsize=17)
    ax.grid()
    ax.legend(loc=4, fontsize=10, title="Embedding, Search Combinations", 
            title_fontsize=11, 
            fancybox=True)
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "best_embeddings-MR1_comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()