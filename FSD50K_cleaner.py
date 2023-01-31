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

    first_letters = [tag[0] for tag in tags]
    alphabet = sorted(list(set(first_letters)))

    rep_dict = {tag: [] for tag in tags} # group dict
    tags_up = list(rep_dict.keys()) # current tags list

    tags_subset = tags_up[:530] # tags starting with "a"
    tags_subset = [tag for tag in tags_subset if len(tag)>2] # Skip short tags
    comb = [(tag0,tag1) for tag0,tag1 in combinations(tags_subset, 2)] # All 2 combinations

    for i,(tag0,tag1) in enumerate(comb):
        dist = ed.eval(tag0, tag1)
        if dist==1:
            ask_user = input(f"|{tag0}|{tag1}| Merge: y/N?")
            if ask_user=="y":
                rep_dict[tag0].append(tag1)

    with open("lol.json","w") as outfile:
        json.dump(rep_dict,outfile,indent=4)