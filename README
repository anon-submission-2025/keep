# KEEP: Knowledge-preserving and Empirically-refined Embedding Process #

This repository contains the implementation code for "KEEP: Integrating Medical Ontologies with Clinical Data for Robust Code Embeddings", a framework that combines knowledge graphs with clinical data to create enhanced medical code embeddings.

## Embedding Generation ##

The repository supports generating several types of medical code embeddings:

### Trained Embeddings ###
Located in `trained_embeddings/our_embeddings/`:
- `train_glove.py`: Implements GloVe training with configurable hyperparameters including regularization
- `train_node2vec.py`: Implements Node2Vec training on medical knowledge graphs
- `train_cui2vec.py`: Generates Cui2Vec embeddings (in `trained_embeddings/cui2vec/`)

### Pre-trained Language Model Embeddings ###
Located in `pretrained_embeddings/`:
- `get_embeddings.py`: Generates embeddings from BioClinical BERT, Clinical BERT, and BioGPT
- Supports both basic embeddings and hierarchy-aware variants that incorporate taxonomic information

## Evaluation Framework ##

### Intrinsic Evaluation ###
Run `intrinsic_evaluation/intrinsic_evaluation.py` to assess how well embeddings capture known medical relationships and hierarchical structures.

### Extrinsic Evaluation ###
The framework includes two key evaluation scripts:
- `extrinsic_evaluation/lr_tuning_jobs.sh`: Performs hyperparameter optimization
- `extrinsic_evaluation/final_training_jobs.sh`: Executes the final model evaluation using optimized parameters

The extrinsic evaluation assesses embedding performance on downstream clinical prediction tasks.

For detailed methodology and results, please refer to our paper. Note that this is an anonymous repository for peer review purposes.