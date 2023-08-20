"""Takes a FSD50K csv file specifying audio file names and computes embeddings 
using an AudioCLIP model  """

import os
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd
import numpy as np
import torch

torch.set_grad_enabled(False)

from essentia.standard import EasyLoader

SAMPLE_RATE = 44100
TRIM_DUR = 30 # seconds

from lib.audio_clip.model import AudioCLIP
from lib.audio_clip.utils.transforms import ToTensor1D

from lib.directories import MODELS_DIR, EMBEDDINGS_DIR, AUDIO_DIR, GT_PATH

def process_audio(model, audio_path, output_dir, sample_rate=SAMPLE_RATE):

    # Load the audio file
    loader = EasyLoader()
    loader.configure(filename=audio_path, 
                     sampleRate=sample_rate, 
                     endTime=TRIM_DUR, # FSD50K are already below 30 seconds
                     replayGain=0 # Do not normalize the audio
                     )
    audio = loader()
    # Zero pad short clips (IN FSD50K 7% of the clips are shorter than 1 second)
    if audio.shape[0] < sample_rate:
        audio = np.concatenate((audio, np.zeros((sample_rate-audio.shape[0]))))

    # Bring to the right format
    audio = audio.astype(np.float32)
    audio_transforms = ToTensor1D()
    audio = torch.stack([audio_transforms(audio.reshape(1,-1))])

    # Process
    ((embeddings, _, _), _), _ = model(audio=audio)
    embeddings = embeddings.tolist()
    
    # Save results
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{fname}.json")
    with open(output_path, 'w') as outfile:
        json.dump({'audio_path': audio_path, 
                   'embeddings': embeddings}, outfile, indent=4)

if __name__=="__main__":

    parser=ArgumentParser(description=__doc__, 
                        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path',
                        default=os.path.join(MODELS_DIR, 'AudioCLIP-Full-Training.pt'),
                        type=str, 
                        help="Path to model.pt chekpoint. By default, "
                        "it points to models/AudioCLIP-Full-Training.pt")
    parser.add_argument('-o', 
                        '--output_dir', 
                        type=str, 
                        default="",
                        help="Path to output directory.")
    args=parser.parse_args()

    # Get the model_name
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Load the model
    model = AudioCLIP(pretrained=args.model_path).eval()

    # Read the file names
    fnames = pd.read_csv(GT_PATH)["fname"].to_list()
    audio_paths = [os.path.join(AUDIO_DIR, f"{fname}.wav") for fname in fnames]
    print(f"There are {len(audio_paths)} audio files to process.")

    # Determine the output directory
    if args.output_dir=="":
        # If the default output directory is used add the AUDIO_DIR to the path
        output_dir = os.path.join(EMBEDDINGS_DIR, os.path.basename(AUDIO_DIR), model_name)
    else:
        output_dir = os.path.join(args.output_dir, model_name)
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting the embeddings to: {output_dir}")

    # Process each audio
    start_time = time.time()
    for i,audio_path in enumerate(audio_paths):
        if i%1000==0:
            print(f"[{i:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")
        process_audio(model, audio_path, output_dir)
    total_time = time.time()-start_time
    print(f"\nTotal time: {time.strftime('%M:%S', time.gmtime(total_time))}")
    print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

    #############
    print("Done!\n")   

    

