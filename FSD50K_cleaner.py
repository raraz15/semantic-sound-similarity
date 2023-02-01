import os
import json
from shutil import copy
import argparse
from itertools import combinations

import editdistance as ed

EXPORT_DIR = "/home/roguz/freesound-perceptual_similarity/clean_tags"

# TODO: fix bpm
if __name__=="__main__":

    parser=argparse.ArgumentParser(description='FSD50K tag cleaner.')
    parser.add_argument('-p', '--path', type=str, required=True, help='JSON file containing the queries.')
    parser.add_argument('-l', '--letter', type=str, required=True, help='Which letter to fix.')
    parser.add_argument('-o', '--output', type=str, default=EXPORT_DIR, help='Output directory.')
    args=parser.parse_args()

    with open(args.path ,"r") as infile:
        metadata_dict = json.load(infile)
    input_name = os.path.splitext(os.path.basename(args.path))[0]
    copy_path = os.path.join(args.output, f"{input_name}_copy.json")
    copy(args.path, copy_path) # Make a copy of the file
    print(f"Made a copy of the input to: {copy_path}")
    print(f"There are {len(metadata_dict)} clip metadata.")

    # Retrieve unique tags
    tags = [tag for metadata in metadata_dict.values() for tag in metadata["tags"]]
    tags = sorted(list(set(tags)))
    print(f"{len(tags)} unique tags found.")

    # Find start positions of each letter
    first_letters = [tag[0] for tag in tags]
    tags = tags[first_letters.index("a"):]# Remove numbers for now
    print(f"{len(tags)} tags left after removing only number tags.")
    first_letters = [tag[0] for tag in tags]
    alphabet = sorted(list(set(first_letters)))
    alph_indices = [first_letters.index(a) for a in alphabet]
    alph_indices.append(len(tags)) # Add z
    start_idx = alph_indices[alphabet.index(args.letter)]
    end_idx = alph_indices[alphabet.index(args.letter)+1]

    # Select which tags to work with
    tags = tags[start_idx:end_idx] # Only tags starting with "args.letter"
    print(f"{len(tags)} tags start with '{args.letter}'")
    tags = [tag for tag in tags if len(tag)>3] # Skip short tags
    print(f"{len(tags)} tags left after removing short ones.")
    comb = [(tag0,tag1) for tag0,tag1 in combinations(tags, 2)] # All 2 combinations

    # You can remove these lines if there are too many tags
    N = len([1 for tag0,tag1 in comb if ed.eval(tag0,tag1)==1])
    print(f"Your validation is required for {N} pairs.\n")

    # Typo finder algorithm. Finds 1 character typos
    groups,remove_indices = [],[]
    for i,(tag0,tag1) in enumerate(comb):
        computed = False
        for j,group in enumerate(groups): # Skip already compared tags
            if tag0 in group and tag1 in group:
                computed = True
                break
        if computed:
            continue
        dist = ed.eval(tag0, tag1) # Calculate levehnsthein distance
        if dist==1:
            if input(f"|{tag0}|{tag1}| Merge? [y/N]: ")=="y":
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

    # Export the groups
    with open(os.path.join(args.output, f"{args.letter}.txt"),"w") as outfile:
        for group in groups:
            outfile.write(group+"\n")

    # Ask for which name to keep
    print("\nSelect the representative/correct spelling...")
    replacement_dict = {}
    for group in groups:
        names = group.split("|")
        print("\nWhich word?")
        for i,name in enumerate(names):
            print(f"{i}: {name}")
        try:
            j = input("Number: ")
            for i,name in enumerate(names):
                replacement_dict[name] = names[int(j)]
        except: # If the user makes a mistake while typing.
            j = input("Number: ")
            for i,name in enumerate(names):
                replacement_dict[name] = names[int(j)]

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

    print("\n#############")
    print("Done!")