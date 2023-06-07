"""Contains functions for plotting single model performance."""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

###################################################################################################
# Micro map#k

def plot_map_at_all_k(model, eval_dir, dataset_name, fig_name="", n_cols=3, save_fig=False, save_dir=""):
    """Takes a model name and plots the mAP@k for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])
    # Read one map to get the k values
    map_path = os.path.join(variation_paths[0], searches[0], "mAP.csv")
    k_values = pd.read_csv(map_path).k.to_list()

    # Read all the maps
    map_dict = {}
    for k in k_values:
        map_dict[k] = {search: [] for search in searches}
        for search in searches:
            for model_dir in variation_paths:
                map_path = os.path.join(model_dir, search, "mAP.csv")
                map = pd.read_csv(map_path)
                full_model_name = model_dir.split("/")[-1]
                variation = "-".join(full_model_name.split("-")[-3:])
                map_dict[k][search].append((variation, map[map.k==k].mAP.to_numpy()[0]))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.20, 0.20, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1
    n_rows = len(map_dict.keys())//n_cols
    fig_name = fig_name if fig_name else f"Embedding Processing and Search Algorithm \
                Performances by mAP Values \n{model} Evaluated on {dataset_name}"

    # Plot the maps
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, 
                            figsize=(6*n_cols,6*n_rows), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    for n,k in enumerate(map_dict.keys()):

        row, col = n//n_cols, n%n_cols
        xticks = []
        for j,search in enumerate(map_dict[k].keys()):
            for z,(variation,map) in enumerate(map_dict[k][search]):
                if j==0:
                    xticks.append(variation.replace("-","\n"))
                if z==0:
                    if search=="dot":
                        label = "Dot Product"
                    elif search=="nn":
                        label = "Nearest Neighbors"
                else:
                    label = ""
                axs[row,col].bar(z+positions[j], 
                                height=map, 
                                width=delta*0.8, 
                                label=label, 
                                color=COLORS[j], 
                                edgecolor='k')
                axs[row,col].text(z+positions[j], 
                                map+0.01, 
                                f"{map:.3f}", 
                                ha='center', 
                                va='bottom', 
                                fontsize=10)

        axs[row,col].set_title(f"k={k} (Page {k//15})", fontsize=17, weight='bold')
        axs[row,col].tick_params(axis='y', which='major', labelsize=11)
        axs[row,col].tick_params(axis='x', which='major', labelsize=10)
        axs[row,col].set_xticks(np.arange(len(xticks)), xticks) #, rotation=20
        axs[row,col].set_yticks(np.arange(0,1.05,0.05))
        axs[row,col].grid()
        axs[row,col].set_ylabel("mAP@k (↑)", fontsize=15)
        axs[row,col].set_xlabel("Processing Parameters", fontsize=15)
        axs[row,col].legend(fontsize=11, loc=4, title="Search Algorithms", 
                            title_fontsize=12, fancybox=True)
        axs[row,col].set_ylim([0,1])
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-mAP.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_map_at_15(model, eval_dir, dataset_name, fig_name="", D=0.25, save_fig=False, save_dir=""):
    """Takes a model name and plots the mAP@k for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])
    # Read one map to get the k values
    map_path = os.path.join(variation_paths[0], searches[0], "mAP.csv")

    # Read all the maps
    map_dict = {search: [] for search in searches}
    for search in searches:
        for model_dir in variation_paths:
            map_path = os.path.join(model_dir, search, "mAP.csv")
            map = pd.read_csv(map_path)
            full_model_name = model_dir.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            map_dict[search].append((variation, map[map.k==15].mAP.to_numpy()[0]))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-D, D, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1
    fig_name = fig_name if fig_name else f"Embedding Processing and Search Algorithm \
                Performances by mAP Values \n{model} Evaluated on {dataset_name}"

    # Plot the maps
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    xticks = []
    for j,search in enumerate(map_dict.keys()):
        for z,(variation,map) in enumerate(map_dict[search]):
            if j==0:
                if "essentia" not in variation:
                    xticks.append(variation.replace("-","\n").replace("Agg_", ""))
                else:
                    xticks.append(variation.replace("essentia-extractor_legacy-", ""))
            if z==0:
                if search=="dot":
                    label = "Dot Product"
                elif search=="nn":
                    label = "Nearest Neighbors"
            else:
                label = ""
            ax.bar(z+positions[j], 
                            height=map, 
                            width=delta*0.6, 
                            label=label, 
                            color=COLORS[j], 
                            edgecolor='k')
            ax.text(z+positions[j], 
                            map+0.01, 
                            f"{map:.3f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=10)

    ax.set_title(f"Page 1 Results", fontsize=17)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.grid()
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Processing Parameters", fontsize=15)
    ax.legend(fontsize=10, loc=1, title="Search Algorithms", 
                        title_fontsize=10, fancybox=True)
    ax.set_ylim([0,1])
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-mAP@15.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

####################################################################################################
# Macro mAP@k

def plot_av_label_based_p_at_15(model, eval_dir, dataset_name, fig_name="", D=0.25, save_fig=False, save_dir=""):
    """Takes a model name and plots the mAP@k for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read all the maps
    map_dict = {search: [] for search in searches}
    for search in searches:
        for variation_path in variation_paths:
            map_path = os.path.join(variation_path, search, "macro_averaged_precision_at_15.txt")
            with open(map_path, "r") as f:
                map = float(f.read())
            full_model_name = variation_path.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            map_dict[search].append((variation, map))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-D, D, len(searches))
        delta = positions[1]-positions[0]
    else:
        positions = [0]
        delta = 1
    fig_name = fig_name if fig_name else f"Embedding Processing and Search Algorithm \
                Performances by Average Label-Based mAP\n{model} Evaluated on {dataset_name}"

    # Plot the maps
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    xticks = []
    for j,search in enumerate(map_dict.keys()):
        for z,(variation,map) in enumerate(map_dict[search]):
            if j==0:
                if "essentia" not in variation:
                    xticks.append(variation.replace("-","\n").replace("Agg_", ""))
                else:
                    xticks.append(variation.replace("essentia-extractor_legacy-", ""))
            if z==0:
                if search=="dot":
                    label = "Dot Product"
                elif search=="nn":
                    label = "Nearest Neighbors"
            else:
                label = ""
            ax.bar(z+positions[j], 
                            height=map, 
                            width=delta*0.6, 
                            label=label, 
                            color=COLORS[j], 
                            edgecolor='k')
            ax.text(z+positions[j], 
                            map+0.01, 
                            f"{map:.3f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=10)

    ax.set_title(f"Page 1 Results", fontsize=17)
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,1.05,0.05))
    ax.grid()
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_xlabel("Processing Parameters", fontsize=15)
    ax.legend(fontsize=10, loc=1, title="Search Algorithms", 
                        title_fontsize=10, fancybox=True)
    ax.set_ylim([0,1])
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-label_based_prec@15_comparisons.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_label_based_p_at_15(model, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes an embedding name and find all the variations of the model. For each varaiation,
    plots the label-based Precision@15 (Macro Averaged Precision@15)."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # For each variation of the model and the search path
    for variation_path in variation_paths:
        for search in searches:

            # Read the label-based mAP@15
            in_path = os.path.join(variation_path, search, "label_positive_rates.csv")
            label_rates = pd.read_csv(in_path)

            # Calculate the Precision@15 for each label
            label_rates["precision"] = label_rates["tp"] / (label_rates["tp"] + label_rates["fp"])
            # Separate the labels and maps
            labels = label_rates["label"].to_list()
            precisions = label_rates["precision"].to_list()
            # Sort the labels and maps
            label_precisions = sorted(zip(labels, precisions), key=lambda x: x[1], reverse=True)

            # Determine some plot parameters
            N = 10 # Number of rows
            delta = len(precisions) // N
            embedding_search = os.path.basename(variation_path) + "-" + search
            fig_name = fig_name if fig_name else f"{embedding_search} mAP Values of Individual Classes on {dataset_name}"

            # Plot the label-based Precision@15
            fig, ax = plt.subplots(figsize=(18, 24), nrows=N, constrained_layout=True)
            fig.suptitle(fig_name, fontsize=16)
            for i in range(N):
                ax[i].bar(
                        [label.replace("_","\n") for label,_ in label_precisions[i*delta:(i+1)*delta]], 
                        [prec for _,prec in label_precisions[i*delta:(i+1)*delta]]
                        )
                ax[i].set_yticks(np.arange(0, 1.05, 0.2))
                ax[i].grid()
                ax[i].set_ylim([0, 1.05])
                ax[i].set_ylabel("Precision@15")
            if save_fig:
                if save_dir == "":
                    print("Please provide a save directory if you want to save the figure.")
                    sys.exit(1)
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"{embedding_search}-label_based_prec@15.png")
                print(f"Saving figure to {fig_path}")
                fig.savefig(fig_path)
                plt.close()

###################################################################################################
# MR1

def plot_mr1(model, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes a model name and plots the MR1 for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir, dataset_name, f"{model}-*")))
    # Read one variation's folder to get the searches
    searches = os.listdir(variation_paths[0])

    # Read the MR1s
    mr1_dict = {}
    for search in searches:
        mr1_dict[search] = []
        for model_dir in variation_paths:
            mr1_path = os.path.join(model_dir, search, "MR1.txt")
            with open(mr1_path,"r") as infile:
                mr1 = float(infile.read())
            # Format the variation name for pprint
            full_model_name = model_dir.split("/")[-1]
            variation = "-".join(full_model_name.split("-")[-3:])
            mr1_dict[search].append((variation, mr1))

    # Determine some plot parameters
    if len(searches)>1:
        positions = np.linspace(-0.2, 0.2, len(searches))
    else:
        positions = [0]
    fig_name = fig_name if fig_name else f"Embedding Processing and Search Algorithm \
                Performances by MR1 Values\n{model} Evaluated on {dataset_name}"

    # Plot the MR1s
    fig, ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    xticks, max_val = [], []
    for i in range(len(variation_paths)):
        for j,search in enumerate(searches):
            variation, mr1 = mr1_dict[search][i]
            max_val += [mr1]
            if j%len(searches)==0:
                xticks.append(variation.replace("-","\n").replace("Agg_", ""))
            if i==0:
                if search=="dot":
                    label = "Dot Product"
                elif search=="nn":
                    label = "Nearest Neighbors"
            else:
                label = ""
            ax.bar(i+positions[j], 
                height=mr1, 
                width=0.35, 
                label=label, 
                color=COLORS[j], 
                edgecolor='k')
            ax.text(i+positions[j], 
                    mr1+0.01, 
                    f"{mr1:.2f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=10)

    # Set the plot parameters
    ax.tick_params(axis='y', which='major', labelsize=11)
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.set_xticks(np.arange(len(xticks)), xticks)
    ax.set_yticks(np.arange(0,max(max_val)+0.5,0.5))
     # TODO: read K
    ax.set_ylabel("MR1@90 (↓)", fontsize=15)
    ax.set_xlabel("Embedding Processing Parameters", fontsize=15)
    ax.legend(loc=4, fontsize=11, title="Search Algorithms", 
              title_fontsize=12, fancybox=True)
    ax.grid()
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{model}-MR1.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()