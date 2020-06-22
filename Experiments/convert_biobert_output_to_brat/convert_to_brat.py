import yaml
import os
from NER import Annotation
from data import get_huner_datasets

config_path = '/home/nick/projects/multi_tasking_transformers/experiments/bml_20200604_nr_0/huner_datasets.yaml'
huner_datasets = get_huner_datasets(config_path)
annotation_dir = 'predictions_brat_formatted'  # output
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

for type in huner_datasets:
    for dataset_name in huner_datasets[type]:
        annotation = Annotation(os.path.join('aligned_merged_with_gold_standard', f'{dataset_name}_{type}.conll')).conll2brat()

        # Write out annotation and text files in BRAT format
        output_text_file_path = os.path.join(annotation_dir, f'{dataset_name}_{type}.txt')
        output_annotation_file_path = os.path.join(annotation_dir, f'{dataset_name}_{type}.ann')

        with open(output_text_file_path, 'w') as output_text_file, \
                open(output_annotation_file_path, 'w') as output_annotation_file:
            for line in annotation.annotations:
                output_line = f'{line[0]}\t{line[1]} {line[2]} {line[3]}\t{line[4]}'
                print(output_line, file=output_annotation_file)
            print(annotation.document_string, file=output_text_file)
