# semantic-sound-similarity

This repository contains the code to run the experiments described in our [CBMI2024](https://cbmi2024.org/) paper, "Evaluation of Deep Audio Representations for Semantic Sound Similarity".

We provide a flexible framework to:
- Extract deep audio representations from neural networks, 
- Process the embeddings
    - time aggregation (mean, max, median, none),
    - dimensionality reduction with PCA,
    - L2 normalization
- Perform similarity search
    - Maximum Inner Product Search (MIPS),
    - Maximum Cosine Similarity Search (MCSS),
    - Nearest Neighbour Search (NNS)
- Evaluate retrieval results
    - Metrics:
        - Mean Average Precision@N (MAP@N)
        - Mean Rank of the First Relevant Item (MR1)
    - Granularity:
        - Class-wise
        - Sound Family-wise
        - Macro averaged (average of the class-wise metrics)

We use the [FSD50K](https://zenodo.org/records/4060432) evaluation set for our experiments, but you can use other datasets with minimal changes.

A copy of the paper is included in the `paper-poster-presentation/` directory. Poster TODO

## Table of Contents

- [Installation](#installation)
- [How To](#how-to)
  - [Embedding Extraction](#embedding-extraction)
  - [Embedding Processing](#embedding-processing)
  - [Similarity Search](#similarity-search)
  - [Performance Evaluation](#performance-evaluation)
    - [MAP@N](#mapn)
    - [MR1](#mr1)
- [Pipelines](#pipelines)
- [Web Interface](#web-interface)


## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/raraz15/semantic-sound-similarity
    cd semantic-sound-similarity
    ```

2. Create a virtual environment (Python 3.10.12 is required):

    ```bash
    python3.10 -m venv sss
    ```

3. Activate the virtual environment:

     ```bash
     source sss/bin/activate
     ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

You're now ready to run the project!

## How To

Here are the main parts of our pipeline.

### Embedding Extraction

To extract embeddings from audio files using the TensorFlowPredict model, you can run the following command:

```bash
usage: tensorflow_predict-embedding_extractor.py [-h] [-o OUTPUT_DIR] config_path audio_dir

Takes a FSD50K csv file specifying audio file names and computes embeddings using a TensorFlowPrecict model. All frame embeddings are exported without aggregation. Currently works with
FSD-Sinet, VGGish, YamNet and OpenL3 models.

positional arguments:
  config_path           Path to config.json file of the model. Assumes the model.pb is next to it.
  audio_dir             Path to directory with audio files.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output directory. (default: )
```

If a model is not included in the Essentia's TensorFlowPredict you should run the following command:

```bash
usage: embedding_extractor.py [-h] [-o OUTPUT_DIR] model_path audio_dir

This script loads a model with a pre-trained checkpoint and extracts clip level embeddings for all the audio files in the FSD50K evaluation dataset.

positional arguments:
  model_path            Path to model.pt chekpoint.
  audio_dir             Path to an audio file or a directory with audio files.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output directory. Default: data/embeddings/<dataset_name>/<model_name> (default: )
```

### Embedding Processing

For processing the embeddings you can use:

```bash
usage: create_clip_level_embedding.py [-h] [-a {mean,median,max,none}] [-N N] [--no-normalization] [--normalization] [--output-dir OUTPUT_DIR] embed_dir

Takes frame level model embeddings and processes them for similarity search. First aggregates frame level embeddings into clip level embeddings then applies PCA to reduce the dimensions
and finally normalizes by the length.

positional arguments:
  embed_dir             Path to an embedding or a directory containing embedding.json files.

options:
  -h, --help            show this help message and exit
  -a {mean,median,max,none}, -aggregation {mean,median,max,none}
                        Type of embedding aggregation. (default: mean)
  -N N                  Number of PCA components to keep. -1 to do not apply. (default: 100)
  --no-normalization    Do not normalize the final clip embedding. (default: False)
  --normalization       Normalize the final clip embedding. (default: False)
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to output directory. If not provided, a directory will be created in the same directory as the embed_dir. (default: )

```

### Similarity Search

After the embeddings are extracted and processed, you can run the similarity search.

```bash
usage: similarity_search.py [-h] [-s {dot,nn}] [-N N] [--ground-truth GROUND_TRUTH] [--output-dir OUTPUT_DIR] embed_dir

In a corpus of sound embeddings, takes each sound as a query and searches for similar sounds using user defined strategies.

positional arguments:
  embed_dir             Directory containing embedding.json files. Embeddings should be prepared with create_clip_level_embedding.py.

options:
  -h, --help            show this help message and exit
  -s {dot,nn}, --search {dot,nn}
                        Type of similarity search algorithm. (default: nn)
  -N N                  Number of search results per query to save. (default: 15)
  --ground-truth GROUND_TRUTH
                        Path to the ground truth CSV file. You can provide a subset of the ground truth by filtering the CSV file before passing it to this script. (default: None)
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Root output directory. Default: data/similarity_rankings/<dataset_name>/<model_name>/<search_type>/ (default: data/similarity_rankings)
```

### Performance Evaluation

Now that the query results are obtained you can evaluate the retrieval results.

#### MAP@N

```bash
usage: evaluate_map_at_n.py [-h] [--metrics METRICS [METRICS ...]] [-N N] [--families-json FAMILIES_JSON] [--output-dir OUTPUT_DIR] results_path ground_truth

Compute evaluation metrics for the similarity search result of an embedding over FSD50K.eval_audio.

positional arguments:
  results_path          Path to similarity_results.json file.
  ground_truth          Path to the ground truth CSV file. You can provide a subset of the ground truth by filtering the CSV file before passing it to this script.

options:
  -h, --help            show this help message and exit
  --metrics METRICS [METRICS ...]
                        Metrics to calculate. (default: ['micro_map@n', 'macro_map@n'])
  -N N                  Cutoff rank. (default: 15)
  --families-json FAMILIES_JSON
                        Path to the JSON file containing the family information of the FSD50K Taxonomy. You can also provide the family information from an ontology (default: None)
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Root output directory. Default: data/evaluation_results/<dataset_name>/<model_name>/<search_type>/ (default: data/evaluation_results)
```

#### MR1

```bash
usage: evaluate_mr1.py [-h] [-s {dot,nn}] [--metrics METRICS [METRICS ...]] [--families-json FAMILIES_JSON] [--output-dir OUTPUT_DIR] embed_dir ground_truth

Since calculation of R1 requires the complete ranking for each query, instead of first running similarity_search.py with N=-1 and storing the results, here we first do a similarity_search
with N=-1 without saving the results and then calculate the R1 for each query.

positional arguments:
  embed_dir             Directory containing embedding.json files. Embeddings should be prepared with create_clip_level_embedding.py.
  ground_truth          Path to the ground truth CSV file. You can provide a subset of the ground truth by filtering the CSV file before passing it to this script.

options:
  -h, --help            show this help message and exit
  -s {dot,nn}, --search {dot,nn}
                        Type of similarity search algorithm. (default: nn)
  --metrics METRICS [METRICS ...]
                        Metrics to calculate. (default: ['micro_mr1', 'macro_mr1'])
  --families-json FAMILIES_JSON
                        Path to the JSON file containing the family information of the FSD50K Taxonomy. You can also provide the family information from an ontology (default:
                        data/ontology/taxonomy/FSD50K_taxonomy-families.json)
  --output-dir OUTPUT_DIR
                        Root output directory. Default: data/evaluation_results/<dataset_name>/<model_name>/<search_type>/ (default: data/evaluation_results)
```

## Pipelines

You can also directly run the bash scripts to run the full pipeline.

First you need to extract embeddings with a model following [Embedding Extraction](#embedding-extraction) once. Then you can define embedding processing and similarity search parameters inside the following bash script. It will create a grid of parameters and perform the full pipeline.

```bash
scripts/experiments/clap_2023-prepare_search_evaluate_experiments.sh
```

If you had processed the embeddings before and want to experiment with just the search function, you can use the following script.

```bash
scripts/experiments/clap_2023-search_evaluate_experiments.sh
```

## Web-interface

```bash
usage: analysis_listener.py [-h] [-p0 PATH0] [-p1 PATH1] [-p2 PATH2] [-p3 PATH3] [-p4 PATH4] [-N N] [--gt-path GT_PATH]

Listen to randomly selected target and query sounds from an analysis file.

options:
  -h, --help            show this help message and exit
  -p0 PATH0, --path0 PATH0
                        Similarity Result Path 0. (default: None)
  -p1 PATH1, --path1 PATH1
                        Similarity Result Path 1. (default: None)
  -p2 PATH2, --path2 PATH2
                        Similarity Result Path 2. (default: None)
  -p3 PATH3, --path3 PATH3
                        Similarity Result Path 3. (default: None)
  -p4 PATH4, --path4 PATH4
                        Similarity Result Path 4. (default: None)
  -N N                  Number of top entries to display. (default: 15)
  --gt-path GT_PATH     Path to the ground truth file. Leave empty for auto. (default: None)

```