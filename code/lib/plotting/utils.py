import os

###################################################################################################
# Utility functions

def get_pca(variation):
    return int(variation.split("-")[1].split("_")[1])

def sort_variation_paths(model, variation_paths):
    # Sort based on the aggregation method, PCA, norm

    if model != "fs-essentia-extractor_legacy":

        all_agg = []
        for var_path in variation_paths:
            variation = var_path.split(model+"-")[1]
            agg = variation.split("-")[0]
            all_agg.append(agg)
        all_agg = sorted(list(set(all_agg)))

        new_sort = []
        for agg in all_agg:
            agg_sorted = []
            for var_path in variation_paths:
                variation = var_path.split(model+"-")[1]
                if agg in variation:
                    agg_sorted.append(var_path)
            agg_sorted = sorted(agg_sorted, key=lambda x: get_pca(x.split(model+"-")[1]))
            new_sort.extend(agg_sorted)
        return new_sort
    else:
        return sorted(variation_paths, key=lambda x: int(x.split("-PCA_")[1]))

def write_model_infos(fig_path, models):
    txt_path = os.path.splitext(fig_path)[0]+".txt"
    with open(txt_path, "w") as infile:
        for model in models:
            infile.write(f"{model[0]}-{model[1]}\n")

def save_function(save_fig, save_dir, default_name, fig):

    if save_fig:
        if save_dir == "":
            raise("Please provide a save directory if you want to save the figure.")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, default_name)
        print(f"Saving figure to {fig_path}")
        fig.savefig(fig_path)