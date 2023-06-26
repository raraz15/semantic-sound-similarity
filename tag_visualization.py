# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import re
import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lib_dir = os.path.dirname(os.getcwd())
sys.path.append(lib_dir)

DATASET = "FSD50K.eval_audio"
data_dir = os.path.join(lib_dir, "data")
eval_dir = os.path.join(data_dir, "evaluation_results", DATASET)
figures_dir = os.path.join(data_dir, "figures")

from directories import  AUDIO_DIR

GT_PATH = "../data/eval.csv"

from metrics import find_indices_containing_label

import IPython.display as ipd

# %% [markdown]
# ## FSD50K Eval Audio Ground Truth Labels

# %%
df = pd.read_csv(GT_PATH)
print(df.shape)

fsd_label_counter = Counter([label for labels in df["labels"].apply(lambda x: x.split(",")).to_list() for label in labels])
fsd_label_counter = {k: v for k,v in sorted(fsd_label_counter.items(), key=lambda x: x[1], reverse=True)}
fsd_labels = list(fsd_label_counter.keys())

# %%
df.head()

# %% [markdown]
# ### Plot Label Distribution

# %% [markdown]
# #### Sideways

# %%
fig, ax = plt.subplots(figsize=(18, 24), constrained_layout=True)
fig.suptitle("FSD50K.eval_audio Label Distribution", fontsize=20)
names = [label.replace("_", " ") for label in label_counter.keys()]
counts = [count for _, count in label_counter.items()]
ax.barh(names, counts, height=0.6, edgecolor="black")
ax.set_ylim([names[0], names[-1]])
ax.set_xscale("log")
ax.set_xlabel("Count (log)", fontsize=16)
ax.set_xticks([1, 10, 100, 1000, 3000])
ax.grid()
#fig.savefig(os.path.join(figures_dir, "FSD50K.eval_audio_label_distribution.png"), dpi=300)
plt.show()

# %% [markdown]
# #### Cool

# %%
N = 10
delta = len(label_counter) // N
fig, ax = plt.subplots(figsize=(18, 24), nrows=N, constrained_layout=True)
fig.suptitle("FSD50K Eval Audio Label Distribution", fontsize=16)
for i in range(N):
    ax[i].bar([label.replace("_","\n") for label in list(label_counter.keys())[i*delta:(i+1)*delta]], 
              [count for count in list(label_counter.values())[i*delta:(i+1)*delta]])
    ax[i].set_ylim([1, 10**4])
    ax[i].set_ylabel("Count (log)")
    ax[i].set_yscale("log")
    ax[i].set_yticks(ticks = [1, 10, 100,1000,10000])
    ax[i].grid()
plt.show()

# %% [markdown]
# ## Investigate FSD50K Taxonomy (After Creating It)

# %%
from anytree import Node, RenderTree, PreOrderIter
from anytree.importer import DictImporter, JsonImporter
from anytree.exporter import DictExporter, JsonExporter
from PrettyPrint import PrettyPrintTree

def get_full_name(node):
    full_name = []
    while node.name != "root":
        full_name.append(node.name)
        node = node.parent
    return full_name[::-1]

# %% [markdown]
# ### Load the FSD Taxonomy Tree

# %%
with open("../data/taxonomy/FSD50K_reduced_onthology-counts.json", "r") as f:
# with open("../data/taxonomy/fsd_taxonomy_tree.json", "r") as f:
    onthology_dict = json.load(f)
onthology_tree = DictImporter().import_(onthology_dict)
print(RenderTree(onthology_tree))

# %% [markdown]
# ### Follow Relationships

# %%
with open("../data/taxonomy/FSD50K_reduced_onthology.json", "r") as f:
    onthology_dict = json.load(f)
onthology_tree = DictImporter().import_(onthology_dict)
print(RenderTree(onthology_tree))

# %%
mapping_dict = {}
for leaf in list(PreOrderIter(onthology_tree, filter_=lambda node: node.name != "root")):
    mapping_dict[leaf.name] = {"path": get_full_name(leaf),
                               "siblings": [sibling.name for sibling in leaf.siblings],
                               "children": [child.name for child in leaf.children],
                               "parent": leaf.parent.name if leaf.parent else None,
                               "depth": leaf.depth,
                               }

# %%
mapping_dict["Conversation"]

# %%
removed_labels = [
            "Bell",
            "Church_bell",
            "Chime",
            "Wind_chime",
            "Chirp_and_tweet",
            "Hiss",
            "Doorbell",
            "Crack",
            "Crackle"
            ]

_df = df.copy()
remove_indices = find_indices_containing_label(removed_labels[0], _df)
for label in removed_labels[1:]:
    remove_indices = remove_indices | find_indices_containing_label(label, _df)
_df = _df[~remove_indices]
_df.shape

# %%
_df["branches"] = _df["labels"].apply(lambda x: len(set([mapping_dict[label]["path"][0] for label in x.split(",")])))
_df["branches"].apply(lambda x: x).value_counts()

# %%
_df = _df[_df["branches"] == 1]
low_count_labels = []
for label in fsd_labels:
    if label in removed_labels:
        continue
    n = len(_df[find_indices_containing_label(label, _df)])
    if n < 15:
        low_count_labels.append(label)
        print(label, n)
print(len(low_count_labels))
print(low_count_labels)

# %%
remove_indices = find_indices_containing_label(low_count_labels[0], _df)
for label in low_count_labels[1:]:
    remove_indices = remove_indices | find_indices_containing_label(label, _df)
_df = _df[~remove_indices]
_df.shape
_df.drop(columns=["branches"], inplace=True)
_df.head()

# %%
_df.to_csv("../data/eval_reduced.csv", index=False)

# %% [markdown]
# ['Knock', 
# 'Buzz', 
# 'Cowbell', 
# 'Tick', 
# 'Bicycle_bell', 
# 'Crow', 
# 'Boat_and_Water_vehicle', 
# 'Gull_and_seagull',
# ]

# %% [markdown]
# ### Plot Tree

# %%
def center_label(label):
    max_len = max([len(l) for l in label.split("_")])
    lines = []
    for line in label.split("_"):
        line = line.center(max_len)
        lines.append(line)
    return "\n".join(line for line in lines)

f1 = lambda node: node.children
f2 = lambda node: center_label(node.name)
pt = PrettyPrintTree(f1, f2, border=True)

# %% [markdown]
# #### Tree Branches

# %%
def plot_music_subsets(names, music_name, instrument_name, onthology_dict, pt):
    children = []
    for c in onthology_dict["children"][0]["children"][0]["children"]:
        if re.sub(r"_\[\d+\]", "", c["name"]) in names:
            if "children" in c:
                sub_dict = {"name": c["name"], "children": c["children"]}
            else:
                sub_dict = {"name": c["name"]}
            children.append(sub_dict)
    dct  = {"name": "root", "children": [{"name": music_name, "children": [{"name": instrument_name, "children": children}]}]}

    tree = DictImporter().import_(dct)
    pt(tree.children[0])

def plot_human_subset(names, onthology_dict, pt):

    children = []
    for c in onthology_dict["children"][1]["children"]:
        if re.sub(r"_\[\d+\]", "", c["name"]) in names:
            if "children" in c:
                sub_dict = {"name": c["name"], "children": c["children"]}
            else:
                sub_dict = {"name": c["name"]}
            children.append(sub_dict)
    dct  = {"name": "root", "children": [{"name": "Human_sounds_(DNE)", "children": children}]}

    tree = DictImporter().import_(dct)
    pt(tree.children[0])

def plot_things_subset(names, onthology_dict, pt):
    children = []
    for c in onthology_dict["children"][4]["children"]:
        if re.sub(r"_\[\d+\]", "", c["name"]) in names:
            if "children" in c:
                sub_dict = {"name": c["name"], "children": c["children"]}
            else:
                sub_dict = {"name": c["name"]}
            children.append(sub_dict)
    music1_dict  = {"name": "root", "children": [{"name": "Sounds_of_things_(DNE)", "children": children}]}

    music1 = DictImporter().import_(music1_dict)
    pt(music1.children[0])

# %%
main_nodes = [node.name for node in (onthology_tree).children]
for node in main_nodes:
    print(node)

# %% [markdown]
# ##### Music

# %%
music_nodes = [node.name for node in onthology_tree.children[0].children[0].children]
for node in sorted(music_nodes):
    print(node)
print(len(music_nodes))

# %%
names = [
        "Percussion",
        "Scratching_(performance_technique)", 
        ]
plot_music_subsets(names, 
                   main_nodes[0], 
                   onthology_tree.children[0].children[0].name, 
                   onthology_dict,
                   pt)

# %%
names = [
        "Brass_instrument", 
        "Harmonica", 
        "Wind_instrument_and_woodwind_instrument", 
        ]
plot_music_subsets(names, 
                   main_nodes[0], 
                   onthology_tree.children[0].children[0].name, 
                   onthology_dict,
                   pt)

# %%
names = [
        "Keyboard_(musical)",
        "Accordion", 
        ]
plot_music_subsets(names, 
                   main_nodes[0], 
                   onthology_tree.children[0].children[0].name, 
                   onthology_dict,
                   pt)

# %%
names = [
        "Harp", 
        "Bowed_string_instrument", 
        "Plucked_string_instrument", 
        ]
plot_music_subsets(names, 
                   main_nodes[0], 
                   onthology_tree.children[0].children[0].name, 
                   onthology_dict,
                   pt)

# %% [markdown]
# ##### Human Sounds

# %%
human_nodes = [node.name for node in onthology_tree.children[1].children]
for node in sorted(human_nodes):
    print(node)
print(len(human_nodes))

# %%
names = ["Human_voice"]
plot_human_subset(names, onthology_dict, pt)

# %%
names = [
        "Human_group_actions", 
        "Hands", 
        "Respiratory_sounds", 
        "Fart", 
        "Burping_and_eructation", 
        "Chewing_and_mastication", 
        "Run", 
        "Walk_and_footsteps"
        ]
plot_human_subset(names, onthology_dict, pt)

# %% [markdown]
# ##### Sounds of Things

# %%
things_nodes = [node.name for node in onthology_tree.children[4].children]
for node in sorted(things_nodes):
    print(node)
print(len(things_nodes))

# %%
names = [
        "Vehicle",
        "Engine",
        ]
plot_things_subset(names, onthology_dict, pt)

# %%
names = [
        "Thump_and_thud",
        "Glass",
        "Wood",
        "Explosion",
        ]
plot_things_subset(names, onthology_dict, pt)

# %%
names = [
        "Liquid",
        "Alarm",
        "Tools",
        "Mechanisms",
        ]
plot_things_subset(names, onthology_dict, pt)

# %%
names = [
        "Domestic_sounds_and_home_sounds",
        ]
plot_things_subset(names, onthology_dict, pt)

# %% [markdown]
# ##### Animal

# %%
animal_nodes = [node.name for node in onthology_tree.children[2].children]
for node in sorted(animal_nodes):
    print(node)
print(len(animal_nodes))

# %%
pt(onthology_tree.children[2], orientation=PrettyPrintTree.VERTICAL)

# %% [markdown]
# ##### Source-ambiguous_sounds

# %%
ambiguous_nodes = [node.name for node in onthology_tree.children[3].children]
for node in sorted(ambiguous_nodes):
    print(node)
print(len(ambiguous_nodes))

# %%
pt(onthology_tree.children[3], orientation=PrettyPrintTree.VERTICAL)

# %% [markdown]
# ##### Natural

# %%
natural_nodes = [node.name for node in onthology_tree.children[5].children]
for node in sorted(natural_nodes):
    print(node)
print(len(natural_nodes))

# %%
pt(onthology_tree.children[5], orientation=PrettyPrintTree.VERTICAL)

# %% [markdown]
# #### Full Plots

# %%
# Keep this and do not create a method for it
f1 = lambda node: node.children
f2 = lambda node: node.name.replace("_", "\n")
pt = PrettyPrintTree(f1, f2, border=True)

# %%
pt(onthology_tree.children[1], orientation=PrettyPrintTree.VERTICAL)

# %%
pt(onthology_tree.children[2], orientation=PrettyPrintTree.VERTICAL)

# %%
pt(onthology_tree.children[3], orientation=PrettyPrintTree.VERTICAL)

# %%
pt(onthology_tree.children[4], orientation=PrettyPrintTree.VERTICAL)

# %%
pt(onthology_tree.children[5], orientation=PrettyPrintTree.VERTICAL)

# %% [markdown]
# ### Listen To Removed Labels

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from essentia.standard import EasyLoader

# %%
removed_labels = [
            "Bell",
            "Church_bell",
            "Chime",
            "Wind_chime",
            "Chirp_and_tweet",
            "Hiss",
            "Doorbell",
            "Crack"
            ]
fnames_of_interest = dict()
for label in removed_labels:
    fnames_of_interest[label] = df[df["labels"].str.contains(label)]["fname"].to_list()
    print(f"{label}: {len(fnames_of_interest[label])}")

# %%
fname = fnames_of_interest["Doorbell"][0]
print(fname)
print(df[df["fname"]==fname]["labels"].to_list()[0].replace(",", "\n"))
audio_path = os.path.join(AUDIO_DIR, 
                          str(fname)+".wav")
ipd.Audio(audio_path)

# %% [markdown]
# ## Create Audioset Onthology and FSD50K Taxonomy Tree

# %% [markdown]
# #### Scrape Audioset

# %%
import requests
from bs4 import BeautifulSoup

# %%
URL = "https://research.google.com/audioset/ontology/"

def format_name(name):
    if name=="Dishes, pots, and pans":
        return "Dishes_and_pots_and_pans"
    elif name.lower()=='boat, water vehicle':
        return "Boat_and_Water_vehicle"
    else:
        name = name[0].capitalize()+name[1:].lower()
        name = name.replace(" ", "_")
        name = name.replace(",", "_and")
        return name

def get_branch(branch_url):
    branch_response = requests.get(branch_url)
    if branch_response.status_code != 200:
        raise Exception(f"Failed to load {branch_url}.")
    else:
        info = []
        branch_soup = BeautifulSoup(branch_response.text, "html.parser")
        for leaf in branch_soup.find_all("li", {"class": "B"}):
            leaf = leaf.find("a")
            leaf_url = URL+leaf.get('href')
            leaf_name = leaf.text
            info.append((leaf_name, leaf_url))
    return info

def parse_audioset(name_format=False, load=False, save=False):

    if load:
        importer = JsonImporter()
        with open("../data/taxonomy/Audioset_ontology.json", "r") as infile:
            tree_dict = json.load(infile)
        root = importer.import_(json.dumps(tree_dict))
        return root
    else:
        response = requests.get(URL+"index.html")
        if response.status_code != 200:
            raise Exception("Failed to load the ontology page.")
        else:
            print("Successfully loaded the ontology page.")
            soup = BeautifulSoup(response.text, "html.parser")

            root = Node("root")
            for column in soup.find("div", {"id": "branches"}).find_all("div", {"class": "Pb"}):

                branches = column.find_all("h2", {"class": "n"})
                l1_info = [(branch.text, URL+branch.a.get('href')) for branch in branches]

                # Level 1
                for branch_name, branch_url in l1_info:
                    if name_format:
                        branch_name = format_name(branch_name)
                    l1_node = Node(branch_name, parent=root)
                    l2_info = get_branch(branch_url)

                    # Level 2
                    for l2_name, l2_url in l2_info:
                        if name_format:
                            l2_name = format_name(l2_name)
                        l2_node = Node(l2_name, parent=l1_node)
                        l3_info = get_branch(l2_url)

                        # Level 3
                        for l3_name, l3_url in l3_info:
                            if name_format:
                                l3_name = format_name(l3_name)
                            l3_node = Node(l3_name, parent=l2_node)
                            l4_info = get_branch(l3_url)

                            # Level 4
                            for l4_name, l4_url in l4_info:
                                if name_format:
                                    l4_name = format_name(l4_name)
                                l4_node = Node(l4_name, parent=l3_node)
                                l5_info = get_branch(l4_url)

                                # Level 5
                                for l5_name, l5_url in l5_info:
                                    if name_format:
                                        l5_name = format_name(l5_name)
                                    l5_node = Node(l5_name, parent=l4_node)
                                    l6_info = get_branch(l5_url)

                                    # Level 6
                                    for l6_name, l6_url in l6_info:
                                        if name_format:
                                            l6_name = format_name(l6_name)
                                        l6_node = Node(l6_name, parent=l5_node)

            if save:
                exporter = DictExporter()
                tree_dict = exporter.export(root)
                with open("../data/taxonomy/Audioset_ontology.json", "w") as outfile:
                    json.dump(tree_dict, outfile, indent=4)

            return root

# %%
audioset_onthology = parse_audioset(load=True, name_format=True, save=False)
print(RenderTree(audioset_onthology))

# %% [markdown]
# #### Map FSD Labels to Audioset root

# %%
audioset_nodes = [leaf for leaf in list(PreOrderIter(audioset_onthology, filter_=lambda node: node.name != "root"))]
print(len(audioset_nodes))
audioset_leaves = [leaf for leaf in list(PreOrderIter(audioset_onthology, filter_=lambda node: node.is_leaf and node.name != "root"))]
print(len(audioset_leaves))

audioset_node_branches = [get_full_name(node) for node in audioset_nodes]

# %%
# Find the FSD labels that are not in the Audioset ontology while mapping the FSD labels to the Audioset labels
mappings = {}
for fsd_label in fsd_labels:
    found = False
    for audioset_node_branch in audioset_node_branches:
        if fsd_label==audioset_node_branch[-1]:
            found = True
            mappings[fsd_label] = audioset_node_branch
            break
    if not found:
        print(fsd_label, "not found")

# %%
print(json.dumps(mappings, indent=4))

# %%
# Follow the branches of FSD label through the Audioset ontology and keep only intermediate nodes that are in the FSD labels
reduced_mappings = {}
for fsd_label, audioset_branch in mappings.items():
    reduced_mappings[fsd_label] = [node for node in audioset_branch if node in fsd_labels]
print(json.dumps(reduced_mappings, indent=4))

# %% [markdown]
# This tree will not be good because FSD deletes a lot of main and intermediate nodes.

# %%
fsd_root = Node("root")
for fsd_branch in reduced_mappings.values():
    node = Node(fsd_branch[0], parent=fsd_root)
    for i in range(1, len(fsd_branch)):
        c_node = Node(fsd_branch[i], parent=node)
print(RenderTree(fsd_root))
if False:
    exporter = DictExporter()
    tree_dict = exporter.export(fsd_root)
    with open("../data/taxonomy/FSD50K_ontology.json", "w") as outfile:
        json.dump(tree_dict, outfile, indent=4)

# %% [markdown]
# ##### Solution is to Use your Own hard work

# %%
with open("../data/taxonomy/FSD_onthology.text", "r") as in_f:
    onthology_text = in_f.read().splitlines()
print(len(onthology_text))

# %%
n = 0
FSD_onthology_all_labels = []
for line in onthology_text:
    line = re.sub(r"\s*-\s", "", line)
    FSD_onthology_all_labels.append(line)
    if line not in fsd_labels:
        print(line)
        n += 1
print(n)

# %% [markdown]
# These are labels that exist in FSd50k but I removed them since they can belong to different families.

# %%
n = 0
for label in fsd_labels:
    if label not in FSD_onthology_all_labels:
        print(label)
        n += 1
print(n)

# %%

onthology_my = {"name": "root", "children": []}
for line in onthology_text:
    idx = line.index("- ")
    label = line[idx+2:]
    if idx==0 and label not in onthology_my:
        onthology_my["children"].append({"name": label, "children": []})
        l1_label = label
    elif idx==4:
        for child in onthology_my["children"]:
            if child["name"]==l1_label:
                child["children"].append({"name": label, "children": []})
                break
        l2_label = label
    elif idx==8:
        for child in onthology_my["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        child2["children"].append({"name": label, "children": []})
                        break
                break
        l3_label = label
    elif idx==12:
        for child in onthology_my["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        for child3 in child2["children"]:
                            if child3["name"]==l3_label:
                                child3["children"].append({"name": label, "children": []})
                                break
                        break
                break
        l4_label = label
    elif idx==16:
        for child in onthology_my["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        for child3 in child2["children"]:
                            if child3["name"]==l3_label:
                                for child4 in child3["children"]:
                                    if child4["name"]==l4_label:
                                        child4["children"].append({"name": label, "children": []})
                                        break
                                break
                        break
                break
        l5_label = label
importer = DictImporter()
my_root = importer.import_(onthology_my)
print(RenderTree(my_root))

# %% [markdown]
# 

# %%
with open("../data/taxonomy/FSD_onthology_reduced.text", "r") as in_f:
    onthology_text = in_f.read().splitlines()
print(len(onthology_text))

# %%
# Check for dublicates
my_lines = []
for line in onthology_text:
    line = re.sub(r"\s*-\s", "", line)
    if line not in my_lines:
        my_lines.append(line)
    else:
        print(line)

# Check for missing labels, collect labels
my_onthology_labels = []
for line in onthology_text:
    line = re.sub(r"\s*-\s", "", line)
    my_onthology_labels.append(line)
    if line not in fsd_labels:
        print(line)

print()
for label in fsd_labels:
    if label not in my_onthology_labels:
        print(label)

# %%
save = True

onthology_my_reduced = {"name": "root", "children": []}
for line in onthology_text:
    idx = line.index("- ")
    label = line[idx+2:]
    if idx==0 and label not in onthology_my_reduced:
        onthology_my_reduced["children"].append({"name": label, "children": []})
        l1_label = label
    elif idx==4:
        for child in onthology_my_reduced["children"]:
            if child["name"]==l1_label:
                child["children"].append({"name": label, "children": []})
                break
        l2_label = label
    elif idx==8:
        for child in onthology_my_reduced["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        child2["children"].append({"name": label, "children": []})
                        break
                break
        l3_label = label
    elif idx==12:
        for child in onthology_my_reduced["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        for child3 in child2["children"]:
                            if child3["name"]==l3_label:
                                child3["children"].append({"name": label, "children": []})
                                break
                        break
                break
        l4_label = label
    elif idx==16:
        for child in onthology_my_reduced["children"]:
            if child["name"]==l1_label:
                for child2 in child["children"]:
                    if child2["name"]==l2_label:
                        for child3 in child2["children"]:
                            if child3["name"]==l3_label:
                                for child4 in child3["children"]:
                                    if child4["name"]==l4_label:
                                        child4["children"].append({"name": label, "children": []})
                                        break
                                break
                        break
                break
        l5_label = label
# Convert to tree
importer = DictImporter()
my_root_reduced = importer.import_(onthology_my_reduced)

if save:
    exporter = JsonExporter(indent=4)
    with open("../data/taxonomy/FSD50K_reduced_onthology.json", "w") as out_f:
        exporter.write(my_root_reduced, out_f)

# %% [markdown]
# max([len(get_full_name(node)) for node in list(PreOrderIter(my_root_reduced, filter_=lambda node: node.name != "root"))])

# %% [markdown]
# #### Add Numbers

# %%
with open("../data/taxonomy/FSD50K_reduced_onthology.json", "r") as f:
    onthology_dict = json.load(f)
onthology_tree = DictImporter().import_(onthology_dict)
print(RenderTree(onthology_tree))

# %%
save = True
for child in onthology_tree.children:
    name = child.name
    name = re.sub(r"_\(DNE\)", "", name)
    n = len(df[find_indices_containing_label(name, df)])
    child.name = name + f'_[{n}]'
    for grandchild in child.children:
        name = grandchild.name
        n = len(df[find_indices_containing_label(name, df)])
        grandchild.name = name + f'_[{n}]'
        for greatgrandchild in grandchild.children:
            name = greatgrandchild.name
            n = len(df[find_indices_containing_label(name, df)])
            greatgrandchild.name = name + f'_[{n}]'
            for greatgreatgrandchild in greatgrandchild.children:
                name = greatgreatgrandchild.name
                n = len(df[find_indices_containing_label(name, df)])
                greatgreatgrandchild.name = name + f'_[{n}]'
                for greatgreatgreatgrandchild in greatgreatgrandchild.children:
                    name = greatgreatgreatgrandchild.name
                    n = len(df[find_indices_containing_label(name, df)])
                    greatgreatgreatgrandchild.name = name + f'_[{n}]'
print(RenderTree(onthology_tree))
if save:
    exporter = DictExporter()
    tree_dict = exporter.export(onthology_tree)
    with open("../data/taxonomy/FSD50K_reduced_onthology-counts.json", "w") as outfile:
        json.dump(tree_dict, outfile, indent=4)

# %% [markdown]
# onthology_dict = DictExporter().export(onthology_tree)
# print(json.dumps(onthology_dict, indent=4))

# %% [markdown]
# ## FSD50K Tags

# %%
import editdistance as ed

# %%
DATASET_DIR = "/data/FSD50K"

with open(f"{DATASET_DIR}/FSD50K.metadata/eval_clips_info_FSD50K.json" ,"r") as infile:
    metadata_dict = json.load(infile)
print(len(metadata_dict))

# %%
all_tags, no_tags = [], []
for clip_id,metadata in metadata_dict.items():
    all_tags.extend(metadata["tags"])
    no_tags += [len(metadata["tags"])]
counter = Counter(all_tags)
counter = {k: v for k,v in sorted(counter.items())}
tags = list(counter.keys())
print(len(counter))
tags = tags[275:] # remove numbers for now
tags = [tag for tag in tags if len(tag)>3] # Skip short tags
tags = [" ".join(tag.split("-")) for tag in tags] # Replace - with space
print(len(tags))

first_letters = [tag[0] for tag in tags]
alphabet = sorted(list(set(first_letters)))

# %%
first_letters.index("c")

# %%
groups = []

tags_subset = tags[:530] # tags starting with "a"
tags_subset = [tag for tag in tags_subset if len(tag)>2] # Skip short tags
comb = [(tag0,tag1) for tag0,tag1 in combinations(tags_subset, 2)] # All 2 combinations

for i,(tag0,tag1) in enumerate(comb):
    dist = ed.eval(tag0, tag1)
    if dist==1:
        ask_user = input(f"|{tag0}|{tag1}| Merge: y/N?")
        if ask_user=="y":
            tag0_in,tag1_in = False,False
            for j,group in enumerate(groups):
                if tag0 in group:
                    tag0_in = True
                    break
                if tag1 in group:
                    tag1_in = True
                    break
            if not (tag0_in or tag1_in):
                group.append(f"{tag0}|{tag1}")
            else:
                groups[j] += f"|{tag1}"

# %%
print(max(no_tags))
print(min(no_tags))
print(stats.mode(no_tags).mode)
print(np.median(no_tags))
print(np.mean(no_tags))

# %%
len(counter)/len(metadata_dict)

# %%
sorted(counter.items())


