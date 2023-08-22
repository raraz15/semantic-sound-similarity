""" This module contains functions for plotting the results of multiple models.
For each metric (MR1, mAP@k, etc.) there is a function that takes a list of [embeddings,search] 
and plots the results in the same plot for comparison.
"""

import os
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

from .utils import save_function, sort_variation_paths, get_pca
from ..directories import EVAL_DIR, DATASET_NAME

###################################################################################
# mAP

def plot_map_comparisons_multimodel(models, map_type, 
                                    eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                    use_fig_name=True, fig_name="",
                                    legend=True,
                                    save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the mAP@15 in 
    the same figure. By setting map_type to "micro" or "macro" you can choose 
    between the two types of mAP.
    """

    # Determine the file name and figure name
    if map_type=="micro":
        file_name = "micro_mAP@15.txt"
        default_fig_name = "Sound Similarity Performances of Embeddings by "\
                        "Instance-Based mAP@15" #(Micro-Averaged)
        figure_save_name = "micro_mAP@15-comparison.png"
    elif map_type=="macro":
        file_name = "balanced_mAP@15.txt"
        default_fig_name = "Sound Similarity Performances of Embeddings by "\
                        "Label-Based mAP@15" # (Macro-Averaged)
        figure_save_name = "macro_mAP@15-comparison.png"
    else:
        raise("map_type must be one of 'micro', 'macro'")
    fig_name = fig_name if fig_name else default_fig_name

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model, variation, search in models:
        embedding_eval_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
        map_path = os.path.join(embedding_eval_dir, search, file_name)
        with open(map_path, "r") as in_f:
            map_at_15 = float(in_f.read())
        maps.append((model, variation, search, map_at_15))

    # Plot the mAP for each model in a single figure
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    for j,(model,variation,search,map_at_15) in enumerate(maps):
        ax.bar(positions[j], 
                map_at_15, 
                label=model,
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k',
                linewidth=1.6)
        ax.text(positions[j], 
                map_at_15+0.01, 
                f"{map_at_15:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                weight='bold')

    # Set the plot parameters
    # ax.set_title("Page 1 Results", fontsize=15)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_xlabel("Embeddings", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_ylim([0,1])
    ax.grid(alpha=0.5)
    if legend:
        ax.legend(loc="best", fontsize=11)

    # Save and show
    save_function(save_fig, save_dir, figure_save_name, fig)
    plt.show()

def plot_family_map_comparisons_multimodel(models, 
                                        eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                        use_fig_name=True, fig_name="",
                                        legend=True,
                                        save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the Family-based 
    mAP@15 in the same figure."""

    default_fig_name = "Sound Similarity Performances of Embeddings by " \
                        "Label-Family-Based mAP@15"
    fig_name = fig_name if fig_name else default_fig_name

    # Read the mAP for each model
    model_maps = defaultdict(list)
    for model, variation, search in models:
        embedding_eval_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
        map_path = os.path.join(embedding_eval_dir, search, "families_mAP@15.csv")
        labels_map = pd.read_csv(map_path)
        families = labels_map["family"].to_list()
        maps = labels_map["map"].to_list()
        for family, family_map in zip(families, maps):
            family = family.replace("_", " ").title()
            model_maps[family].append((model, variation, search, family_map))
    model_maps = dict(sorted(model_maps.items(), key=lambda x: x[0]))

    fig,ax = plt.subplots(nrows=len(model_maps) ,figsize=(18,12), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    for i, (family, family_aps) in enumerate(model_maps.items()):
        for j,(model,variation,search,ap) in enumerate(family_aps):
            ax[i].bar(j, 
                    ap, 
                    label=model,
                    width=0.8, 
                    color=COLORS[j], 
                    edgecolor='k',
                    linewidth=1.3)
            ax[i].text(j, 
                    ap+0.01, 
                    f"{ap:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    weight='bold')

        # Set the plot parameters
        ax[i].set_title(family, fontsize=15, weight='bold')
        ax[i].set_yticks(np.arange(0,1.05,0.1))
        ax[i].tick_params(axis='x', which='major', labelsize=0)
        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].set_ylabel("mAP@15 (↑)", fontsize=13)
        ax[i].set_ylim([0,1])
        ax[i].grid(alpha=0.5)
        if legend and i==0:
            ax[i].legend(loc="upper center", fontsize=12, 
                         fancybox=True, ncol=len(models))

    save_function(save_fig, save_dir, "family_based_mAP@15-comparison.png", fig)
    plt.show()

def plot_macro_map_pca_comparisons_multimodel(models,
                                        eval_dir=EVAL_DIR, dataset_name=DATASET_NAME,
                                        use_fig_name=True, fig_name="",                                     
                                        save_fig=False, save_dir=""):

    default_fig_name = "Effect of the Number of PCA Components on Sound " \
                        "Similarity Performace by Label-Averaged mAP@15"
    fig_name = fig_name if fig_name else default_fig_name

    fig, ax = plt.subplots(nrows=len(models), figsize=(18,12), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(default_fig_name, fontsize=20, weight='bold')
    
    for i,model_search in enumerate(models):

        model, agg, norm, search = model_search

        # Find all the variation_paths of the model
        if model=="fs-essentia-extractor_legacy":
            wildcard = f"{model}-PCA_*"
        else:
            wildcard = f"{model}-Agg_{agg}-PCA_*-Norm_{norm}"
        variation_paths = sorted(glob(os.path.join(eval_dir, dataset_name, wildcard)))
        # Sort by PCA components
        variation_paths = sort_variation_paths(model, variation_paths)

        # Read all the maps
        variations, maps = [], []
        for variation_path in variation_paths:
            map_path = os.path.join(variation_path, search, "balanced_mAP@15.txt")
            with open(map_path, "r") as in_f:
                balanced_map_at_15 = float(in_f.read())
            full_model_name = variation_path.split("/")[-1]
            if "fs-essentia-extractor_legacy" in full_model_name:
                variation = "-"+full_model_name.split("-")[-1]
            else:
                variation = "-".join(full_model_name.split("-")[-3:])
            maps.append(balanced_map_at_15)
            variations.append(variation)

        # Determine color for presentation
        if model=="fs-essentia-extractor_legacy":
            color = COLORS[0]
        elif model=="audioset-yamnet-1":
            color = COLORS[1]
        elif model=="fsd-sinet-vgg41-tlpf-1":
            color = COLORS[3]
        elif model=="fsd-sinet-vgg42-tlpf-1":
            color = COLORS[4]

        # Plot the maps
        xticks = []
        for j,(variation,balanced_mAP) in enumerate(zip(variations, maps)):
            ax[i].bar(j,
                    height=balanced_mAP, 
                    width=0.85,
                    color=color,
                    edgecolor='k',
                    linewidth=1.2)
            ax[i].text(j, 
                    balanced_mAP+0.01, 
                    f"{balanced_mAP:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=12)
            xticks.append(get_pca(variation))

        ax[i].set_title(f"{model}", fontsize=19, weight='bold')
        ax[i].tick_params(axis='y', which='major', labelsize=11)
        ax[i].tick_params(axis='x', which='major', labelsize=13)
        ax[i].set_xticks(np.arange(len(xticks)), xticks)
        ax[i].set_yticks(np.arange(0, 1.05, 0.1))
        ax[i].grid(alpha=0.5)
        ax[i].set_ylabel("mAP@15 (↑)", fontsize=13)
        ax[i].set_xlabel("Number of PCA Components", fontsize=13)
        ax[i].set_ylim([0,1])

    save_function(save_fig, save_dir, 
                  f"macro_map@15-PCA_comparisons-multi_models.png", fig)
    plt.show()

####################################################################################################
# MR1

def plot_mr1_comparisons_multimodel(models, mr1_type,
                                    eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                    use_fig_name=True, fig_name="",
                                    legend=True,          
                                    save_fig=False, save_dir=""):
    """ Takes a list of [(model,variation,search)] and plots all the MR1s in the same figure.
    mr1_type must be one of 'micro', 'macro'."""

    # Determine the file name and figure name
    if mr1_type=="micro":
        file_name = "micro_MR1.txt"
        default_fig_name = "Sound Similarity Performances of Embeddings by "\
                        "Instance-Based MR1" #(Micro-Averaged)
        figure_save_name = "micro_MR1-comparison.png"
    elif mr1_type=="macro":
        file_name = "balanced_MR1.txt"
        default_fig_name = "Sound Similarity Performances of Embeddings by "\
                        "Label-Based MR1" # (Macro-Averaged)
        figure_save_name = "macro_MR1-comparison.png"
    else:
        raise("mr1_type must be one of 'micro', 'macro'")
    fig_name = fig_name if fig_name else default_fig_name

    # Read the MR1s for each embedding-search combination
    mr1s = []
    for model, variation, search in models:
        embedding_eval_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
        mr1_path = os.path.join(embedding_eval_dir, search, file_name)
        with open(mr1_path,"r") as infile:
            mr1 = float(infile.read())
        mr1s.append((model, variation, search, mr1))

    # Determine the ytick params
    max_mr1 = max([m[3] for m in mr1s])
    if max_mr1<=10:
        delta_yticks = 0.5
    else:
        delta_yticks = 5

    # Plot all the MR1s in the same figure
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    for i,(model,variation,search,mr1) in enumerate(mr1s):
            ax.bar(i, 
                mr1, 
                label=model,
                width=0.8, 
                color=COLORS[i], 
                edgecolor='k',
                linewidth=1.3)
            ax.text(i, 
                mr1+0.01, 
                f"{mr1:.2f}", 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                weight='bold')

    # Set the plot parameters
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.set_yticks(np.arange(0, max_mr1+delta_yticks, delta_yticks))
    ax.set_xlabel("Embeddings", fontsize=15)
    ax.set_ylabel("MR1 (↓)", fontsize=15)
    ax.grid(alpha=0.5)
    if legend:
        ax.legend(loc=4, fontsize=11)

    save_function(save_fig, save_dir, figure_save_name, fig)
    plt.show()

def plot_family_mr1_comparisons_multimodel(models, 
                                           eval_dir=EVAL_DIR, dataset_name=DATASET_NAME, 
                                           use_fig_name=True, fig_name="",
                                           legend=True,
                                           save_fig=False, save_dir=""):
    """Takes a list of [(model,variation,search)] and plots all the Family-based 
    MR1 in the same figure."""

    default_fig_name = "Sound Similarity Performances of Embeddings by " \
                        "Label-Family-Based MR1"
    fig_name = fig_name if fig_name else default_fig_name

    # Read the MR1 for each embedding-search combination
    model_mr1s = defaultdict(list)
    max_mr1 = -100
    for model, variation, search in models:
        embedding_eval_dir = os.path.join(eval_dir, dataset_name, model+"-"+variation)
        mr1_path = os.path.join(embedding_eval_dir, search, "families_MR1.csv")
        families_mr1 = pd.read_csv(mr1_path)
        families = families_mr1["family"].to_list()
        mr1s = families_mr1["mr1"].to_list()
        for family, family_mr1 in zip(families, mr1s):
            family = family.replace("_", " ").title()
            if family_mr1>max_mr1:
                max_mr1 = family_mr1
            model_mr1s[family].append((model, variation, search, family_mr1))
    model_mr1s = dict(sorted(model_mr1s.items(), key=lambda x: x[0]))

    fig,ax = plt.subplots(nrows=len(model_mr1s) ,figsize=(18,12), constrained_layout=True)
    if use_fig_name:
        fig.suptitle(fig_name, fontsize=19, weight='bold')
    for i, (family, family_mr1s) in enumerate(model_mr1s.items()):
        for j,(model,variation,search,mr1) in enumerate(family_mr1s):
            ax[i].bar(j, 
                    mr1, 
                    label=model,
                    width=0.8, 
                    color=COLORS[j], 
                    edgecolor='k',
                    linewidth=1.3)
            ax[i].text(j, 
                    mr1+0.01, 
                    f"{mr1:.3f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    weight='bold')

        # Set the plot parameters
        ax[i].set_title(family, fontsize=15, weight='bold')
        #ax[i].set_yticks(np.arange(0,1.05,0.1))
        ax[i].tick_params(axis='x', which='major', labelsize=0)
        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].set_ylabel("MR1 (↓)", fontsize=13)
        ax[i].set_ylim([0, max_mr1+10])
        ax[i].grid(alpha=0.5)
        if legend and i==1:
            ax[i].legend(loc="upper center", fontsize=11, 
                         fancybox=True, ncol=len(models))

    save_function(save_fig, save_dir, 
                   "family_based_MR1-comparison.png", fig)
    plt.show()
