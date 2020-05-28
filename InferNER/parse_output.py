'''parse output of infer.py.'''
import pandas as pd
from pprint import pprint

TO_BRAT = False

def get_entity_type(entity_string):
    # if entity_string.startswith('B') or entity_string.startswith('I'):
    #     label = entity_string.split('_')[-3]
    #     final_label = f'{entity_string[0:2]}{label}'
    #     return final_label
    if entity_string.startswith('B') or entity_string.startswith('I'):
        label = entity_string[2:]
        return label
    else:
        return entity_string


example_output_file = 'bert_inference_on_huner_data/species_variome_biobert_annotated.tsv'
with open(example_output_file, encoding='utf-8') as f:
    next(f)
    output_lines = []
    for i, line in enumerate(f):
        line = line.strip().split('\t')
        line[5] = get_entity_type(line[5])
        if not line[1].startswith('##'):
            output_lines.append(line)
            token_start_line = line
        else:
            token_start_line[1] = token_start_line[1] + line[1].replace('##', '')

            sentence_start_span, sentence_end_span = [int(x) for x in token_start_line[2].split(' ')]
            current_sentence_start_span, current_sentence_end_span = [int(x) for x in line[2].split(' ')]
            # sentence_start_span += current_sentence_start_span
            sentence_end_span = current_sentence_end_span - 1
            token_start_line[2] = ' '.join([str(sentence_start_span), str(sentence_end_span)])

            doc_start_span, doc_end_span = [int(x) for x in token_start_line[3].split(' ')]
            current_doc_start_span, current_doc_end_span = [int(x) for x in line[3].split(' ')]
            # doc_start_span += current_doc_start_span
            doc_end_span = current_doc_end_span - 1
            token_start_line[3] = ' '.join([str(doc_start_span), str(doc_end_span)])

    # pprint(output_lines[0:100])
    output = pd.DataFrame(output_lines)
    output.to_csv('species_variome_subword_joined.tsv', sep='\t', index=False)

if TO_BRAT:
    brat_output_lines = []
    for i, line in enumerate(output_lines):
        brat_index = f"T{i+1}"
        label = line[5]
        doc_span = line[3]

