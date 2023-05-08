import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


colors = ["r", "g", "b", "y", "c", "m", "k"]

def plot_map(model, eval_dir, n_cols=3, save_fig=False ,save_dir="data/figures"):
    """Takes a model name and plots the MAP@K for all the variations of the model."""

    # Find all the variation_paths of the model
    variation_paths = sorted(glob.glob(os.path.join(eval_dir,f"{model}-*")))

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
                map_dict[k][search].append((variation, map[map.k==k].mAP.to_numpy()))
    

    # Plot the maps
    n_rows = len(k_values)//n_cols
    fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(6*n_cols,6*n_rows),constrained_layout=True)
    fig.suptitle(model, fontsize=20, weight='bold')
    for n,k in enumerate(k_values):
        axs[n//n_cols, n%n_cols].set_title(f"k={k}", fontsize=17, weight='bold')
        xticks = []
        for z in range(len(map_dict[k]["dot"])):
            variation = map_dict[k]["dot"][z][0]
            dot_val = map_dict[k]["dot"][z][1]
            nn_val = map_dict[k]["nn"][z][1]
            xticks.append(variation)
            if z==0:
                leg0,leg1 = "dot product", "nearest neighbors"
            else:
                leg0,leg1 = "", ""
            axs[n//n_cols, n%n_cols].bar(z-0.2, height=dot_val, width=0.35, label=leg0, color="g", edgecolor='k')
            axs[n//n_cols, n%n_cols].bar(z+0.2, height=nn_val, width=0.35 ,label=leg1, color="b", edgecolor='k')
            axs[n//n_cols, n%n_cols].text(z-0.2, dot_val+0.01, f"{dot_val[0]:.3f}", ha='center', va='bottom', fontsize=10)
            axs[n//n_cols, n%n_cols].text(z+0.2, nn_val+0.01, f"{nn_val[0]:.3f}", ha='center', va='bottom', fontsize=10)
        axs[n//n_cols, n%n_cols].tick_params(axis='y', which='major', labelsize=11)
        axs[n//n_cols, n%n_cols].tick_params(axis='x', which='major', labelsize=10)
        axs[n//n_cols, n%n_cols].set_xticks(np.arange(len(xticks)), xticks, rotation=20)
        axs[n//n_cols, n%n_cols].set_yticks(np.arange(0,1.05,0.05))
        axs[n//n_cols, n%n_cols].grid()
        axs[n//n_cols, n%n_cols].set_ylabel("MAP@K", fontsize=15)
        axs[n//n_cols, n%n_cols].set_xlabel("Processing Parameters", fontsize=15)
        axs[n//n_cols, n%n_cols].legend(fontsize=11, loc=4, title="Search Algorithms", title_fontsize=12, fancybox=True)
        axs[n//n_cols, n%n_cols].set_ylim([0,1])
    if save_fig:
        plt.savefig(save_dir, f"{model}.png")
    plt.show()

def plot_comparisons(models, eval_dir):
    """Takes a list of models and plots the MAP@K for all the variations of the model.
    Each model must be a tupple of (model_name, [variations], search_algorithm)"""

    # Determine Some Parameters
    positions = np.linspace(-0.25, 0.25, len(models))
    delta = positions[1]-positions[0]
    n_variations = len(models)

    fig,axs = plt.subplots(nrows=1, ncols=n_variations, 
                           figsize=(18,6), constrained_layout=True)
    fig.suptitle("Model Comparison", fontsize=20, weight='bold')
    for i in range(n_variations):

        # Read all the maps for all variations of model i
        maps = []
        for model in models:
            model_dir = os.path.join(eval_dir, f"{model[0]}-{model[1][i]}")
            results_dir = os.path.join(model_dir, model[2])
            map_path = os.path.join(results_dir, "mAP.csv")
            df = pd.read_csv(map_path)
            maps.append(df.mAP.to_numpy())
        K = df.k.to_numpy()

        # Plot the maps
        for j in range(len(K)):
            for l,map in enumerate(maps):
                if j==0:
                    label = models[l][0]
                else:
                    label = "" # Only dsiplay labels once
                axs[i].bar(j+positions[l], 
                        map[j], 
                        label=label, 
                        width=delta*0.85, 
                        color=colors[l], 
                        edgecolor='k'
                        )
                axs[i].text(j+positions[l], 
                            map[j]+0.01, 
                            f"{map[j]:.2f}", 
                            ha='center', 
                            va='bottom', 
                            fontsize=8, 
                            weight='bold')

        axs[i].set_yticks(np.arange(0,1.05,0.05))
        axs[i].set_xticks(np.arange(j+1), K)
        axs[i].tick_params(axis='x', which='major', labelsize=13)
        axs[i].tick_params(axis='y', which='major', labelsize=11)
        axs[i].set_xlabel("K", fontsize=15)
        axs[i].set_ylabel("MAP@K", fontsize=15)
        axs[i].set_ylim([0,1])
        if i==n_variations-1:
            axs[i].set_title("No PCA", fontsize=17)
        else:
            title = " ".join(models[1][1][i].split("PCA_")[1].split("-")[0].split("_") + ["Components"])
        axs[i].set_title(title, fontsize=17)
        axs[i].grid()
        axs[i].legend(fontsize=10, title="Models", title_fontsize=11, fancybox=True)