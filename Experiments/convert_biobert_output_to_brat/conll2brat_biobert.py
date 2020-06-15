'''
Convert BioBERT output to BRAT format for viewing and for evaluation against
the gold standard.

Objectives:
1) Use alignments.pt to join tokens that were originially
    on the same line in the GS test file.
2) Parse GS test set, mark sentence boundaries.
3) Get spans for BRAT format.
4) Write to file.
5) View on BRAT.
'''

import os
import torch
import numpy as np


def flatten_alignments(alignments):
    flats = []
    last_seq_len = 0
    for i, seq in enumerate(alignments):
        if i > 0:
            last_seq_len = flats[-1] + 1
        for word_pos in seq:
            if word_pos > -1:
                flats.append(word_pos + last_seq_len)
    return flats


write_conll = False
dataset_name = 'bc2gm'
type = 'gene'
partition = 'test'

preprocessed_dir = os.path.join('/home/nick/projects/multi_tasking_transformers/experiments/data/biomedical/huner',
                                f'{dataset_name}_{type}',
                                f'{partition}')
## Alignments
alignments_path = os.path.join(preprocessed_dir, 'subword_to_spacy_alignment.pt')
alignments = torch.load(alignments_path).numpy()
alignments = flatten_alignments(alignments)

with open('/home/nick/projects/synbiobert/Experiments/convert_biobert_output_to_brat/SubwordClassificationHead_bc2gm_gene.conll', 'r') as subword_predictions_file:
    subword_predictions = []
    for line in subword_predictions_file:
        line = line.strip().split('\t')
        subword_predictions.append(line)
assert len(alignments) == len(subword_predictions)

# Dictionary approach
word_dict = {pos: [] for pos in set(alignments)}
for i, x_i in enumerate(subword_predictions):
    word_dict[alignments[i]].append(x_i)

results = []
for word_pos in word_dict:
    subword_info = word_dict[word_pos]
    token = ''.join([subword[0].replace('##', '') for subword in subword_info])
    true_label = subword_info[0][1].replace('NP', type.title())
    predicted_label = subword_info[0][2].replace('NP',
                                                 type.title())  # This assumes the label of the first subword token as the label for the word.
    probability = subword_info[0][3]  # Same here.
    results.append(f'{token} {true_label} {predicted_label} {probability}')

if write_conll:
    with open('aligned_bc2gm_gene.conll', 'w') as outfile:
        for line in results:
            outfile.write(f'{line}\n')

###
from biomedical_datasets.huner import load_huner_dataset
data = load_huner_dataset('bc2gm', 'gene', 'test')
