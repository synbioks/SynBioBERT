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
import yaml
from warnings import warn
import torch
from NER import Conll2Brat

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


def align_to_tokens(input_directory,
                    preprocessed_dir,
                    dataset_name,
                    type,
                    partition='test',
                    write_conll=False,
                    output_directory='aligned',
                    merge_gold_standard=True,
                    gold_standard_dir=''
                    ):
    # # Testing params
    # write_conll = False
    # dataset_name = 'bc2gm'
    # type = 'gene'
    # partition = 'test'

    # preprocessed_dir = os.path.join('/home/nick/projects/multi_tasking_transformers/experiments/data/biomedical/huner',
    #                                 f'{dataset_name}_{type}',
    preprocessed_dir = os.path.join(preprocessed_dir,
                                    f'{dataset_name}_{type}',
                                    f'{partition}')

    ## Alignments
    alignments_path = os.path.join(preprocessed_dir, 'subword_to_spacy_alignment.pt')
    alignments = torch.load(alignments_path).numpy()
    alignments = flatten_alignments(alignments)

    # with open('/home/nick/projects/synbiobert/Experiments/convert_biobert_output_to_brat/SubwordClassificationHead_bc2gm_gene.conll', 'r') as subword_predictions_file:
    with open(os.path.join(input_directory, f'SubwordClassificationHead_{dataset_name}_{type}.conll'),
              'r') as subword_predictions_file:
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
        if not os.path.exists('aligned'):
            os.mkdir('aligned')
        with open(os.path.join('aligned', f'aligned_{dataset_name}_{type}.conll'), 'w') as outfile:
            for line in results:
                outfile.write(f'{line}\n')

    if merge_gold_standard:
        gold_standard_file = os.path.join(gold_standard_dir, type, f'{dataset_name}.conll.{partition}')
        merge_with_gold_standard(results,
                                 dataset_name,
                                 type,
                                 gold_standard_file=gold_standard_file
                                 )


def merge_with_gold_standard(results, dataset_name, type, gold_standard_file, output_dir='aligned_merged_with_gold_standard'):
    are_gold_and_predicted_same_len, gold_len, results_len = test_gold_and_predicted(results, gold_standard_file)
    if not are_gold_and_predicted_same_len:
        warn(
            f'Tokens in gold standard ({gold_len}) and predictions ({results_len}) do not match for {dataset_name} {type}.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ---Start the merge---
    gold_token_index = 0
    with open(gold_standard_file, 'r') as gold_file, \
            open(os.path.join(output_dir, f'{dataset_name}_{type}.conll'), 'w') as outfile:
        for i, line in enumerate(gold_file):
            line = line.strip()
            if line == '' or line.startswith("-DOCSTART- X X O"):
                print(line, file=outfile)
            else:
                gold_token, pos, gold_label = line.strip().split(' ')
                pred_token, results_gold_label, pred_label, prob = results[gold_token_index].split(' ')
                if gold_token == '\u2062':
                    '''Sometimes the gold standard token is an uncommon whitespace character and is 
                    ignored during preprocessing for prediction'''
                    print(line, file=outfile)
                    continue

                if "[UNK]" not in pred_token:
                    if len(gold_token) != len(pred_token):
                        warn(
                            f'Warning: gold token "{gold_token.lower()}" does not match pred token "{pred_token.lower()}" in {dataset_name} {type} on gold line {i}, {gold_token_index}, pred line {gold_token_index}')
                        # Gold standard tokens can contain special unicode characters which are converted during preprocessing. Instead check the length of the token.
                        # Some tokens processed by the BERT tokenizer are unknown. This will return [unk]

                assert gold_label == results_gold_label.replace(type.title(),
                                                                'NP'), f'labels do not match. {gold_label} {results_gold_label.replace(type.title(), "NP")}'

                print(f'{gold_token} {results_gold_label} {pred_label} {prob}', file=outfile)
                gold_token_index += 1


def test_gold_and_predicted(results, gold_standard_file):
    '''
    Check that gold stardard and predictions have the same number of tokens
    '''
    with open(gold_standard_file, 'r') as gold_file:
        print(gold_file.name)
        num_gold_tokens = 0
        for line in gold_file:
            line = line.strip()
            if line == '' or line.startswith("-DOCSTART- X X O"):
                continue
            else:
                num_gold_tokens += 1
    return num_gold_tokens == len(results), num_gold_tokens, len(results)


if __name__ == '__main__':

    with open(
            '/home/nick/projects/multi_tasking_transformers/experiments/bml_20200604_nr_0/huner_datasets.yaml') as config_file:
        huner_datasets = yaml.safe_load(config_file)

    for type in huner_datasets:
        for dataset_name in huner_datasets[type]:
            align_to_tokens(
                input_directory='/home/nick/projects/multi_tasking_transformers/experiments/bml_20200604_nr_0/results',
                preprocessed_dir='/home/nick/projects/multi_tasking_transformers/experiments/data/biomedical/huner',
                dataset_name=dataset_name,
                type=type,
                partition='test',
                write_conll=False,
                output_directory='aligned',
                merge_gold_standard=True,
                gold_standard_dir='/home/nick/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data'
            )



# ### Some useful function
# from biomedical_datasets.huner import load_huner_dataset
# data = load_huner_dataset('bc2gm', 'gene', 'test')
