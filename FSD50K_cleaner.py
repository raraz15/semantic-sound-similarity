import os
import json
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import editdistance as ed

if __name__=="__main__":

    DATASET_DIR = "/data/FSD50K"

    with open(f"{DATASET_DIR}/FSD50K.metadata/eval_clips_info_FSD50K.json" ,"r") as infile:
        metadata_dict = json.load(infile)
    print(len(metadata_dict))

    all_tags, no_tags = [], []
    for clip_id,metadata in metadata_dict.items():
        all_tags.extend(metadata["tags"])
        no_tags += [len(metadata["tags"])]
    counter = Counter(all_tags)
    counter = {k: v for k,v in sorted(counter.items())}
    tags = list(counter.keys())
    print(len(counter))
    tags = tags[275:] # remove numbers for now
    tags = [" ".join(tag.split("-")) for tag in tags] # Replace - with space
    print(len(tags))

    #first_letters = [tag[0] for tag in tags]
    #alphabet = sorted(list(set(first_letters)))

    tags_subset = tags[:530] # tags starting with "a"
    tags_subset = [tag for tag in tags_subset if len(tag)>3] # Skip short tags
    comb = [(tag0,tag1) for tag0,tag1 in combinations(tags_subset, 2)] # All 2 combinations

    groups,remove_indices = [],[]
    for i,(tag0,tag1) in enumerate(comb):
        computed = False
        for j,group in enumerate(groups): # Skip for already compared tags
            if tag0 in group and tag1 in group:
                computed = True
                break
        if computed:
            continue
        dist = ed.eval(tag0, tag1)
        if dist==1:
            if input(f"|{tag0}|{tag1}| Merge: y/N?" )=="y":
                remove_indices.append(i)
                tag0_in,tag1_in = False,False
                for j,group in enumerate(groups): # Search each group for both tags
                    if tag0 in group:
                        tag0_in = True
                    if tag1 in group:
                        tag1_in = True
                    if tag0_in or tag1_in:
                        break # A tag is found exit search
                if (not tag0_in) and (not tag1_in): # Neither tag exist in a group, create
                    groups.append(f"{tag0}|{tag1}")
                elif tag0_in and (not tag1_in): # Add tag1 to the group
                    groups[j] += f"|{tag1}"
                elif (not tag0_in) and tag1_in: # Add tag0 to the group
                    groups[j] += f"|{tag0}"
    print(groups)

    print(len(remove_indices))
    comb = [x for i,x in enumerate(comb) if i not in remove_indices]
    



    #with open("lol.json","w") as outfile:
    #    json.dump(rep_dict,outfile,indent=4)