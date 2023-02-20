"""FSD50K tag cleaner algorithm. Author: R. Oğuz Araz"""

import os
import json
from shutil import copy
import argparse
from itertools import combinations

import editdistance as ed

EXPORT_DIR = "/home/roguz/freesound-perceptual_similarity/clean_tags"

# TODO: Clean the number tags (bpm needs -)
if __name__=="__main__":

    parser=argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, required=True, 
                        help='JSON file containing the queries.')
    parser.add_argument('-l', '--letter', type=str, required=True, 
                        help='Which letter to fix.')
    parser.add_argument('-o', '--output', type=str, default=EXPORT_DIR, 
                        help='Output directory.')
    args=parser.parse_args()

    # Read the metadata dict
    with open(args.path ,"r") as infile:
        metadata_dict = json.load(infile)

    # Create the output dir
    os.makedirs(args.output, exist_ok=True)

    # Make a copy of the input
    input_name = os.path.splitext(os.path.basename(args.path))[0]
    copy_path = os.path.join(args.output, f"{input_name}_copy.json")
    copy(args.path, copy_path) 
    print(f"Made a copy of the input to: {copy_path}")
    print(f"There are {len(metadata_dict)} clip metadata.")

    # Retrieve unique tags
    tags = [tag for metadata in metadata_dict.values() for tag in metadata["tags"]]
    tags = sorted(list(set(tags)))
    print(f"{len(tags)} unique tags found.")

    # Find start positions of each letter, we will clean the tags letter by letter
    first_letters = [tag[0] for tag in tags]
    tags = tags[first_letters.index("a"):] # Remove numbers for now
    print(f"{len(tags)} tags left after removing only number tags.")
    first_letters = [tag[0] for tag in tags]
    alphabet = sorted(list(set(first_letters)))
    alph_indices = [first_letters.index(a) for a in alphabet]
    alph_indices.append(len(tags)) # Add end of z

    # Select only tags starting with "args.letter"
    start_idx = alph_indices[alphabet.index(args.letter)]
    end_idx = alph_indices[alphabet.index(args.letter)+1]
    tags = tags[start_idx:end_idx]
    print(f"{len(tags)} tags start with '{args.letter}'")
    tags = [tag for tag in tags if len(tag)>3] # Skip short tags
    print(f"{len(tags)} tags left after removing short ones.")
    comb = [(tag0,tag1) for tag0,tag1 in combinations(tags, 2)] # All 2 combinations

    # Find which combinations differ by 1 distance
    comb_dist = []
    for tag0,tag1 in comb:
        dist = ed.eval(tag0, tag1) # Calculate levehnsthein distance
        if dist == 1:
            comb_dist.append((tag0,tag1))
    N = len(comb_dist)
    print(f"Your validation is required for {N} pairs.\n")

    # Typo finder algorithm. Finds 1 character typos
    groups,decisions = [],[]
    for i,(tag0,tag1) in enumerate(comb_dist):
        # Search each group for the tags
        tag0_in,tag1_in = False,False
        n0,n1 = -1,-1
        for j,group in enumerate(groups):
            for tag in group: # Compare tag by tag
                if tag0 == tag:
                    tag0_in = True
                    n0 = j # Record which group tag0 is in
                elif tag1 == tag:
                    tag1_in = True
                    n1 = j # Record which group tag1 is in
        # Ask for user decision if the tags are not merged yet
        if tag0_in and tag1_in:
            if n0 == n1:
                print(f"[{i+1:>3}/{N}]|{tag0}|{tag1}| Already merged to the same group.")
            else:
                print(f"[{i+1:>3}/{N}]|{tag0}|{tag1}| Already merged to different groups.")
            decisions.append([tag0,tag1,"Skipped"]) # Keep a track of the decisions
        else:
            decision = input(f"[{i+1:>{len(str(N))}}/{N}]|{tag0}|{tag1}| Merge? [y/N]: ")=="y"
            decisions.append([tag0,tag1,decision]) # Keep a track of the decisions
            if decision:
                # Put the new tags in corresponding group
                if (not tag0_in) and (not tag1_in): # Neither tag exist in a group, create one
                    groups.append([tag0,tag1])
                elif tag0_in and (not tag1_in): # Add tag1 to the group
                    groups[n0].append(tag1)
                elif (not tag0_in) and tag1_in: # Add tag0 to the group
                    groups[n1].append(tag0)
    print(f"You created {len(groups)} groups.")

    # Export the decisions
    output_path = os.path.join(args.output, f"{args.letter}_decisions.txt")
    print(f"Exported your decisions to: {output_path}")
    with open(output_path,"w") as outfile:
        for x,y,z in decisions:
            outfile.write(f"{x}|{y}|{z}\n")

    # Export the groups
    output_path = os.path.join(args.output, f"{args.letter}.txt")
    print(f"Exported the groups to: {output_path}")
    with open(output_path,"w") as outfile:
        for group in groups:
            outfile.write("|".join(group)+"\n")

    # Ask for which name to keep
    print("\nSelect the representative/correct spelling...")
    replacement_dict = {}
    for k,group in enumerate(groups):
        print(f"\n[{k+1}/{len(groups)}] Which word?")
        for i,name in enumerate(group):
            print(f"{i}: {name}")
        try:
            j = input(f"Choose between [0, {len(group)-1}]: ")
            for i,name in enumerate(group):
                replacement_dict[name] = group[int(j)]
        except: # If the user makes a mistake while typing.
            j = input(f"Choose between [0, {len(group)-1}]: ")
            for i,name in enumerate(group):
                replacement_dict[name] = group[int(j)]

    # Export the replacement dict
    output_path = os.path.join(args.output, f"{args.letter}_replacement.json")
    print(f"Exported the replacement dictionary to: {output_path}")
    with open(output_path,"w") as outfile:
        json.dump(replacement_dict, outfile, indent=4)

    # Unify the grouped tags
    for clip_id,metadata in metadata_dict.items():
        clean_tags = []
        for tag in metadata["tags"]:
            if tag in replacement_dict:
                tag = replacement_dict[tag]
            clean_tags.append(tag)
        metadata_dict[clip_id]["tags"] = clean_tags

    # Export the new metadata
    output_path = os.path.join(args.output, f"{input_name}_{args.letter}.json")
    print(f"\nExporting the new metadata to: {output_path}")
    with open(output_path,"w") as outfile:
        json.dump(metadata_dict,outfile,indent=4)

    # Count remaining tags
    tags = set([tag for metadata in metadata_dict.values() for tag in metadata["tags"]])
    print(f"{len(tags)} unique tags remaining.")

    print("\n############")
    print("Done!")