'''parse output of infer.py.'''
import pandas as pd

example_output_file = 'example_output.tsv'
with open(example_output_file, encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')

