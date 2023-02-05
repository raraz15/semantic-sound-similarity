import os
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

SEED = 27
N = 20 # Number of samples to include
np.random.seed(SEED)

AUDIO_DIR = "/data/FSD50K/FSD50K.eval_audio"
GT_PATH = "/data/FSD50K/FSD50K.ground_truth/eval.csv"

if __name__=="__main__":

    df = pd.read_csv(GT_PATH)
    print(df.shape)

    # Count all the tags, find unique ones
    all_tags = []
    for tags in df["labels"].to_list():
        for tag in tags.split(","):
            all_tags.append(tag)
    counter = Counter(all_tags)
    counter = {k: v for k,v in sorted(counter.items())}
    print(len(list(counter.keys())))

    high_count_keys = [k for k,v in counter.items() if v>=20]
    print(len(high_count_keys))

    # Find each fname where a label is included
    fnames_dict = defaultdict(list)
    for key in high_count_keys:
        for _,fname,file_labels,_ in df.itertuples():
            if key in file_labels:
                fnames_dict[key].append(fname)

    # For each key sample N independent sampples
    df_sub = []
    for key,fnames in fnames_dict.items():
        samples = np.random.choice(fnames, size=N, replace=False) # Sample N fnames
        for fname in samples:
            audio_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
            labels = df[df["fname"]==fname]["labels"].values[0]
            df_sub.append([audio_path,key,labels])
    df_sub = pd.DataFrame(df_sub, columns=["path","key","labels"])
    print(df_sub.shape)
    
    # Export csv
    df_sub.to_csv(f"FSD50K.ground_truth_eval_{N}.csv",index=False)