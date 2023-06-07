# freesound-sound_similarity

This repository contains the Sound and Music Computing Master's Thesis of Recep Oğuz Araz at University of Pompeu Fabra.

Our goal is to use deep embeddings for improving the similar sounds functionality of Freesound in terms of perceptual similarity.

Advisors: Dmitry Bogdanov, Pablo Alonso
Collaboration with: Frederic Font, Alastair Porter

contact: oguza97@gmail.com

## TODO:
- Understand audioset taxonomy
    - How many labels
    - How many example for each label
        - Is any of the labels contain less than 15 examples?
- PCA
    - Lower PCA components
    - Change PCA naming convention
- Metrics
    - Label-based AP:
        - Consider match only if query label is present
    - Leaf vs node mAP
    - ncdg metric
    - macro vs micro mAP
    - Once u decide on final comparisons, MR1 with no k
- Models
    - L3
    - PANs
    - PAST
- Normalize audio