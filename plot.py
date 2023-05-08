import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_map(model, model_variations, eval_dir, n_cols=3, save_fig=False ,save_dir="data/figures"):

    # Read one variations's folder to get the searches
    model_dir = os.path.join(eval_dir,f"{model}-{model_variations[0]}")
    searches = os.listdir(model_dir)
    # Read one map to get the k values
    map_path = os.path.join(model_dir, searches[0], "mAP.csv")
    map = pd.read_csv(map_path)
    k_values = map.k.to_list()

    # Read all the maps
    map_dict = {}
    for k in k_values:
        map_dict[k] = {search: [] for search in searches}
        for search in searches:
            for variation in model_variations:
                model_dir = os.path.join(eval_dir, f"{model}-{variation}")
                results_dir = os.path.join(model_dir, search)
                map_path = os.path.join(results_dir, "mAP.csv")
                map = pd.read_csv(map_path)
                variation = "-".join([x.split("_")[1] for x in variation.split("-")])
                map_dict[k][search].append((variation, map[map.k==k].mAP.to_numpy()))

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
        axs[n//n_cols, n%n_cols].legend(fontsize=11, title="Search Algorithms", title_fontsize=12, fancybox=True)
        axs[n//n_cols, n%n_cols].set_ylim([0,1])
    if save_fig:
        plt.savefig(save_dir, f"{model}.png")
    plt.show()
