Annotate ACS using BioBERT.
--------------------------- 

**Objectives**
- Annotate 100 ACS articles using BioBERT.
- Inter-annotator agreement, BioBERT, HUNER, Kevin (expert)

## TODO

1. ACS Inference
 
    1.  Create alignment
    
    2. Map BERT output via alignment
    
    3. Write to BRAT 
        
2. Compare BioBERT, Kevin, HUNER. 
    1. Pull Kevin's
 
3. Doc, Zip, Upload to Willow. 

## ACS Inference

### Create Alignment

Create alignment between tokens in document text and processed subtoken encodings. 

#### INPUT  

1. Raw data directory (see Issue 1) 

    /home/nick/projects/synbiobert/raw-data/ACS-100/
    
2. Head Directory
    
    /home/nick/projects/synbiobert/models/SubwordClassificationHead_iepa_gene_checkpoint_20
       
    
#### OUTPUT. 

1. Processed data directory

    /home/nick/projects/synbiobert/data/ACS-100

Preprocessors created in `InferNER.__init__()`. 
Used in `run_dcoument()` near line 78.      

```python
# run_document line 85-95
with open(path_to_document, encoding='utf8') as f:
    document_as_string = f.read()  # does this work for large documents?

sentencized_document = self.sentencizer(document_as_string)
```

  
## Files and Directories

ACS output:
    
    /home/nick/projects/synbiobert/Experiments/annotate_ASC100_20200619/subtoken_label_probabilities

Subword join program

Brat Conversion program

## Issues

1. The annotated text doesn't match. BRAT complains.  

1. **Document text files used for inter-annotator agreement**. Gaurav did the XML conversion without Jacob's script. Ideally, we should use Jacob's
but doing so may affect the Comparison of BioBERT, Kevin, HUNER.  

