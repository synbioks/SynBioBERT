import yaml
import os
from NER import Conll2Brat
from data import get_huner_datasets
from biomedical_datasets.huner import batch_gold_standard_annotation

config_path = '/home/nick/projects/multi_tasking_transformers/experiments/bml_20200604_nr_0/huner_datasets.yaml'
huner_datasets = get_huner_datasets(config_path)
output_annotation_dir = 'bert_predictions_brat'  # output
gold_standard_dir = "/home/nick/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data"
output_gold_standard_dir = 'gold_standard_brat'


def conll2brat_bert_predictions(config_path, huner_datasets, annotation_dir):
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)

    for type in huner_datasets:
        for dataset_name in huner_datasets[type]:
            annotation = Conll2Brat(
                os.path.join('aligned_merged_with_gold_standard', f'{dataset_name}_{type}.conll')).conll2brat()

            # Write out annotation and text files in BRAT format
            output_text_file_path = os.path.join(annotation_dir, f'{dataset_name}_{type}.txt')
            output_annotation_file_path = os.path.join(annotation_dir, f'{dataset_name}_{type}.ann')

            with open(output_text_file_path, 'w') as output_text_file, \
                    open(output_annotation_file_path, 'w') as output_annotation_file:
                for line in annotation.annotations:
                    output_line = f'{line[0]}\t{line[1]} {line[2]} {line[3]}\t{line[4]}'
                    print(output_line, file=output_annotation_file)
                print(annotation.document_string, file=output_text_file)


def conll2brat_gold_standard(huner_datasets, gold_standard_dir, output_gold_standard_dir, partition='test'):
    if not os.path.exists(output_gold_standard_dir):
        os.makedirs(output_gold_standard_dir)

    gold_batch = batch_gold_standard_annotation(huner_datasets, gold_standard_dir, partition=partition)
    output_text_file_basenames = [f"{dataset_name}_{type}.txt" for type in huner_datasets for dataset_name in huner_datasets[type]]
    for i, text_filename in enumerate(output_text_file_basenames):
        with open(os.path.join(output_gold_standard_dir, text_filename), 'w') as fout:
            fout.write(gold_batch[i].document_string)

    output_ann_file_basenames = [f"{dataset_name}_{type}.ann" for type in huner_datasets for dataset_name in huner_datasets[type]]
    for i, ann_filename in enumerate(output_ann_file_basenames):
        annotations = gold_batch[i]
        with open(os.path.join(output_gold_standard_dir, ann_filename), 'w') as annout:
            for annotation in annotations:
                type = output_ann_file_basenames[i].split("_")[1].replace('.ann', '')
                if annotation[1] == 'NP':
                    annotation[1] = type.title()
                output_string = f'{annotation[0]}\t{annotation[1]} {annotation[2]} {annotation[3]}\t{annotation[4]}\n'
                annout.write(output_string)
    return gold_batch


if __name__ == '__main__':
    # conll2brat_bert_predictions(config_path, huner_datasets, output_annotation_dir)
    conll2brat_gold_standard(huner_datasets, gold_standard_dir, output_gold_standard_dir, partition='test')
