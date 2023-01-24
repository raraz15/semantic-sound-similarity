# Code adapted from UPF MTG ASP Lab homework of Frederic Font.

import os
import json
import datetime as dt
import argparse

import pandas as pd

import freesound

FREESOUND_API_KEY = "freesound_api.json"
FREESOUND_STORE_METADATA_FIELDS = ['id', 'name', 'username', 'previews', 'license', 'tags']  # Freesound metadata properties to store

DOWNLOADS_DIR=os.path.join("downloads",dt.datetime.strftime(dt.datetime.now(),"%d_%m_%Y-%H_%M"))

# TODO: what does pager do?
def query_freesound(client, query, filt, num_results):
    """Queries freesound with the given query and filter values.
    If no filter is given, a default filter is added to only get sounds shorter than 30 seconds.
    """
    if filt is None:
        filt = 'duration:[0 TO 30]'  # Set default filter
    pager = client.text_search(
        query = query,
        filter = filt,
        fields = ','.join(FREESOUND_STORE_METADATA_FIELDS),
        group_by_pack = 1,
        page_size = num_results
    )
    #pager.next_page()
    return [sound for sound in pager]

def retrieve_sound_preview(client, sound, directory):
    """Download the high-quality OGG sound preview of a given Freesound sound object to the given directory.
    """
    return freesound.FSRequest.retrieve(
        sound.previews.preview_hq_ogg,
        client,
        os.path.join(directory, sound.previews.preview_hq_ogg.split('/')[-1])
    )

def make_pandas_record(sound, directory):
    """Create a dictionary with the metadata that we want to store for each sound.
    """
    record = {key: sound.as_dict()[key] for key in FREESOUND_STORE_METADATA_FIELDS}
    del record['previews']  # Don't store previews dict in record
    record['freesound_id'] = record['id']  # Rename 'id' to 'freesound_id'
    del record['id']
    record['path'] = os.path.join(directory, sound.previews.preview_hq_ogg.split("/")[-1])  # Store path of downloaded file
    return record

if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Freesound Sound Retriever.')
    parser.add_argument('-p', '--path', type=str, required=True, help='JSON file containing the queries.')
    parser.add_argument('-o', '--output-dir', type=str, default=DOWNLOADS_DIR, help='Directory to download the audio files and the dataframe.')
    parser.add_argument('-N', type=int, default=10, help='Number of queries to download.')
    args=parser.parse_args()

    # Load the queries
    with open(args.path, 'r') as infile:
        queries=json.load(infile)

    # Load the freesound API Key
    print("Loading user Freesound API Key...")
    with open(FREESOUND_API_KEY, "r") as infile:
        FREESOUND_API_KEY=json.load(infile)["FREESOUND_API_KEY"]

    # Initialize the freesound client
    print("Initializing the Freesound Client...")
    client = freesound.FreesoundClient()
    client.set_token(FREESOUND_API_KEY)

    # Make the queries to freesound
    print("Making the queries...")
    query_dict = {query['query']: query_freesound(client, query['query'], query['filter'], args.N) for query in queries.values()}

    # Download the queries and collect metadata
    print(f"The files will be downloaded to: {args.output_dir}")
    metadata = []
    for query, sounds in query_dict.items():
        print(f"\nDownloading queries for: {query}")
        query_path=os.path.join(args.output_dir, query)
        os.makedirs(query_path, exist_ok=True) # Create a subdirectory for each query
        for count, sound in enumerate(sounds):
            print(f'ID: {sound.id} [{count + 1}/{len(sounds)}]')
            retrieve_sound_preview(client, sound, query_path)
            metadata.append(make_pandas_record(sound, query_path))

    # Make a Pandas DataFrame with the metadata of our sound collection and save it
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(args.output_dir, 'metadata.csv'))
    print(f'\nSaved DataFrame with {len(df)} entries to: {args.output_dir}/metadata.csv')

    # Copy the queries for future reference
    with open(os.path.join(args.output_dir, 'queries.json'), 'w') as json_file:
        json.dump(queries, json_file, indent=4)

    #############
    print("Done!")