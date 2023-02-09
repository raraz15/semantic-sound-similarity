import os
import sys
import argparse
#from shutil import copy
import random

import pandas as pd

GT_PATH = "/data/FSD50K/FSD50K.ground_truth/eval.csv"

# TODO: print labels
if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Listen to target and query sounds from an analysis.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to a results.txt file.')
    parser.add_argument('-N', type=int, default=3, help="Number of queries to return.")
    args=parser.parse_args()

    # Parse the analysis file
    with open(args.path ,"r") as infile:
        x = [line for line in infile.read().split("\n") if line]

    # Read the emtadata
    df = pd.read_csv(GT_PATH)

    # Randomly sample a target sound
    idx = random.randint(0,(len(x)//(args.N+1))-1)

    # Print the analysis results
    for i in range((args.N+1)*idx,(args.N+1)*(idx+1)):
        print(x[i])
        fname = x[i].split("/")[-1].split(".wav")[0]
        print(df[df["fname"]==int(fname)]["labels"].values[0])