# SynBioBERT

Tools for refining language models for the Synthetic Biology (SynBio) domain. 
Currently, the language models are implimented using transformer-based algorithms.

InferNER contains code to perform inference on Synthetic Biology articles. It currently
extracts text that mentions:

1. Genes
2. Proteins 
3. Species
3. Cell lines      

More types of entities will be added to the models as they become available. 

To install: pip install -r requirements.txt 

To run: python infer.py
