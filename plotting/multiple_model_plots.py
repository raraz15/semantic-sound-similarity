""" Multiple Model Plot Methods"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TABLEAU_COLORS
COLORS = list(TABLEAU_COLORS.values())

def get_model_name(full_name):
    return full_name.split("-PCA")[0].split("-Agg")[0]

# TODO: check MR1@90
def plot_mr1_comparisons_single_variation(models, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Read the MR1s for each model
    mr1s = []
    for model in models:
        model_dir = os.path.join(eval_dir, dataset_name, model[0])
        results_dir = os.path.join(model_dir, model[1])
        mr1_path = os.path.join(results_dir, "MR1.txt")
        with open(mr1_path,"r") as infile:
            mr1 = float(infile.read())
        mr1s.append((model[0], model[1], mr1))

    # Plot the MR1s
    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else f"Embedding Performances using MR1 values Evaluated on {dataset_name} Set"
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    for i,(model_name,search,mr1) in enumerate(mr1s):
            ax.bar(i, 
                mr1, 
                label=get_model_name(model_name),
                width=1*0.85, 
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
        names = "-".join([model[0] for model in models])
        fig_path = os.path.join(save_dir, f"{names}-MR1_comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_map_comparisons_single_variation(models, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of models and plots the mAP@k for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, dataset_name, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "mAP.csv")
        df = pd.read_csv(map_path)
        maps.append((model[0], model[1], df.mAP.to_numpy()))

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else f"Embedding Performances using mAP@k values Evaluated on {dataset_name} Set"
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    #ax.set_title("For each model, the best performing processing parameters are used", fontsize=15)
    ax.set_title("Page 1 Results", fontsize=15)
    #for i in range(len(K)):
    for j,(model_name,search,map) in enumerate(maps):
        ax.bar(0+positions[j], 
                map[0], 
                #label=model_name+f" {search}" if 0==0 else "", 
                label=get_model_name(model_name) if 0==0 else "",
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(0+positions[j], 
                map[0]+0.01, 
                f"{map[0]:.3f}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                weight='bold'
                )

    # Set the plot parameters
    ax.set_yticks(np.arange(0,1.05,0.05))
    #ax.set_xticks(np.arange(i+1), K)
    #ax.set_xticks(np.arange(1,i+2))
    ax.tick_params(axis='x', which='major', labelsize=0)
    ax.tick_params(axis='y', which='major', labelsize=11)
    #ax.set_xlabel("K (Similarity Rank)", fontsize=15)
    ax.set_xlabel("Embedding, Search Combinations", fontsize=15)
    #ax.set_xlabel("Page", fontsize=15)
    ax.set_ylabel("mAP@15 (↑)", fontsize=15)
    ax.set_ylim([0,1])
    #ax.set_title(models[0][0].split("-")[-1].replace("_"," "), fontsize=17)
    ax.grid()
    ax.legend(loc="best", fontsize=10, title_fontsize=11, fancybox=True)
    if save_fig:
        if save_dir == "":
            print("Please provide a save directory if you want to save the figure.")
            sys.exit(1)
        os.makedirs(save_dir, exist_ok=True)
        if fig_name == "":
            names = "-".join([model[0] for model in models])
            fig_path = os.path.join(save_dir, f"{names}-mAP_comparison_k15.png")
        else:
            fig_path = os.path.join(save_dir, fig_name+".png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
    plt.show()

def plot_micro_map_comparisons_multimodel(models, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of [(embedding,search)] and plots all the Micro Averaged mAP@k in the same figure."""

    default_fig_name = f"Embedding Performances using mAP@15 (Micro-Averaged) values on {dataset_name}"

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, dataset_name, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "micro_mAP.csv")
        map = pd.read_csv(map_path)
        map = map[map["k"]==15]["mAP"].values[0]
        maps.append((model[0], model[1], map))

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("Page 1 Results", fontsize=15)
    for j,(model_name,search,map) in enumerate(maps):
        ax.bar(0+positions[j], 
                map, 
                label=get_model_name(model_name),
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(0+positions[j], 
                map+0.01, 
                f"{map:.3f}", 
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
        fig_path = os.path.join(save_dir, f"best_embeddings-micro_mAP@15-comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        txt_path = os.path.splitext(fig_path)[0]+".txt"
        with open(txt_path, "w") as infile:
            for model in models:
                infile.write(f"{model[0]}-{model[1]}\n")
    plt.show()

def plot_macro_map_comparisons_multimodel(models, eval_dir, dataset_name, fig_name="", save_fig=False, save_dir=""):
    """Takes a list of [(embedding,search)] and plots all the Macro Averaged mAP@k in the same figure."""

    default_fig_name = f"Embedding Performances using Label-Based mAP@15 (Macro-Averaged) values on {dataset_name}"

    # Determine Some Parameters
    positions = np.linspace(-0.4, 0.4, len(models))
    delta = positions[1]-positions[0]

    # Read the mAP for each model
    maps = []
    for model in models:
        model_dir = os.path.join(eval_dir, dataset_name, model[0])
        results_dir = os.path.join(model_dir, model[1])
        map_path = os.path.join(results_dir, "macro_mAP@15.csv")
        map = pd.read_csv(map_path)["macro_map@15"].values[0]
        maps.append((model[0], model[1], map))

    fig,ax = plt.subplots(figsize=(18,6), constrained_layout=True)
    fig_name = fig_name if fig_name else default_fig_name
    fig.suptitle(fig_name, fontsize=19, weight='bold')
    ax.set_title("Page 1 Results", fontsize=15)
    for j,(model_name,search,map) in enumerate(maps):
        ax.bar(0+positions[j], 
                map, 
                label=get_model_name(model_name) if 0==0 else "",
                width=delta*0.80, 
                color=COLORS[j], 
                edgecolor='k'
                )
        ax.text(0+positions[j], 
                map+0.01, 
                f"{map:.3f}", 
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
        fig_path = os.path.join(save_dir, f"best_embeddings-macro_mAP@15-comparison.png")
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)
        txt_path = os.path.splitext(fig_path)[0]+".txt"
        with open(txt_path, "w") as infile:
            for model in models:
                infile.write(f"{model[0]}-{model[1]}\n")
    plt.show()