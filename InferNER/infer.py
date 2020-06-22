"""predict entities using pre-trained/fine-tuned bert model"""

import os, time, json, yaml, re, argparse
import torch
from torch.nn.functional import softmax
from multi_tasking_transformers.heads import SubwordClassificationHead
# from multi_tasking_transformers.multitaskers import MultiTaskingBert
from transformers import BertConfig, BertForTokenClassification, BertModel
# Transformers https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
from tokenizers import BertWordPieceTokenizer
# Tokenizers https://github.com/huggingface/tokenizers/tree/master/bindings/python
from spacy_transformers import TransformersLanguage
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action='store_true', help='dry run')  # by default it stores false
parser.add_argument("--force-run", action='store_true',
                    help='Causes documents with existing output files to be overwritten.')
args = parser.parse_args()


class InferNER(object):

    def __init__(self, head_directories, head_configs, device=None,
                 from_huner=False):
        """

        :param head_directories:
        :param head_configs:
        :param device:
        """
        # SET DEVICE
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        assert len(head_directories) == len(head_configs)
        self.models = []
        for i, head in enumerate(head_directories):
            # LOAD BASE MODEL
            print(f'Loading BERT pre-trained model {head}')
            self.bert = BertModel.from_pretrained(head, from_tf=False)
            # LOAD HEAD
            print(f'Loading {head}')
            path_to_head_config = os.path.join(head, head_configs[i])
            self.path_to_vocab = os.path.join(head, 'vocab.txt')
            self.head_directory = head
            self.head_config = BertConfig.from_pretrained(path_to_head_config)
            head_config_dict = json.load(open(os.path.join(self.head_directory, head_configs[i]), 'rb'))
            self.head = SubwordClassificationHead(head_config_dict['head_task'], labels=head_config_dict['labels'])
            print(self.head.from_pretrained(self.head_directory))
            self.models.append({'head': self.head, 'base': self.bert, 'entity_type': head.split('_')[-3],
                                'dataset': head.split('_')[-4]})

        # LOAD TOKENIZER AND SET OPTIONS
        print('Loading Tokenizer and setting options')
        self.tokenizer = BertWordPieceTokenizer(self.path_to_vocab,
                                                lowercase=False)  # TODO make this configurablle
        self.tokenizer.enable_padding(direction='right',
                                      pad_id=0,
                                      max_length=self.head_config.max_position_embeddings)

        # Construct Sentencizer
        head_name = os.path.basename(self.head_directory)
        self.sentencizer = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})  # TODO rename
        self.sentencizer.add_pipe(self.sentencizer.create_pipe("sentencizer"))

        print('Loaded BERT head, config, tokenizer, and sentencizer')
        self.labels = sorted(self.head.config.labels)  # Fine-tuning may have been done on sorted labels.

        self.from_huner = from_huner
        # TODO align output to document tokens.
        # TODO visualize
        # TODO Stats
        # TODO predict each sentence in batches

    def run_document(self, path_to_document, output_filename=None, output_directory="."):

        output_filename_fullpath = os.path.join(output_directory, output_filename)
        if os.path.exists(output_filename_fullpath):
            print(f"{output_filename_fullpath} exists, skipping")
            return

        with open(path_to_document, encoding='utf8') as f:
            document_as_string = f.read()  # does this work for large documents?

        self.output_dict = {'tokens': [],
                            'sentence_spans': [],
                            'document_spans': [],
                            'probability': [],
                            'labels': []
                            }

        sentencized_document = self.sentencizer(document_as_string)
        number_of_sentences = len(list(sentencized_document.sents))
        test_stop = 10000000
        # number_of_sentences = test_stop
        for model in self.models:
            self.head = model['head']
            self.bert = model['base']

            if self.from_huner:
                model_entity_type = model['entity_type']
                model_dataset = model['dataset']
                document_entity_type = os.path.basename(path_to_document).split("_")[0]
                document_dataset = os.path.basename(path_to_document).split("_")[1].replace('.txt', '')
                if model_entity_type != document_entity_type:
                    print(model_entity_type, document_entity_type)
                    continue
                if model_dataset != document_dataset:
                    print(model_dataset, document_dataset)
                    continue

            for sentence_idx, sentence in enumerate(sentencized_document.sents):
                annotation_start = time.time()
                if sentence_idx > test_stop:
                    break

                print(f'\nAnnotating sentence {sentence_idx} of {number_of_sentences}')

                self.sentence = sentence
                self.sentence_idx = sentence_idx
                # self.sentence = str(list(sentencized_document.sents)[0])
                # self.sentence = "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
                # self.sentence = r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."
                self.sentence_encoding = self.tokenizer.encode(self.sentence.string)
                if len(self.sentence_encoding) > 512:
                    print(self.sentence)

                # PREPARE MODEL INPUT
                input_ids = torch.tensor([self.sentence_encoding.ids], dtype=torch.long)
                attention_mask = torch.tensor([self.sentence_encoding.attention_mask], dtype=torch.long)
                token_type_ids = torch.tensor([self.sentence_encoding.type_ids], dtype=torch.long)
                self.document = sentencized_document
                self.tokens = self.sentence_encoding.tokens
                self.spans = self.sentence_encoding.offsets
                self.input_ids = input_ids

                # RUN EXAMPLE THROUGH BERT
                self.bert.eval()
                if not next(self.bert.parameters()).is_cuda:
                    self.bert.to(device=self.device)
                self.head.eval()
                if not next(self.head.parameters()).is_cuda:
                    self.head.to(device=self.device)
                with torch.no_grad():
                    print(f"BERT Head: {self.head}")
                    print(f"On {self.device} device")
                    input_ids = input_ids.to(device=self.device)
                    attention_mask = attention_mask.to(device=self.device)
                    self.bert_outputs = self.bert(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  # token_type_ids=token_type_ids,
                                                  token_type_ids=None,
                                                  position_ids=None)[0]
                    self.subword_scores = self.head(self.bert_outputs)[0]
                    self.subword_scores_softmax = softmax(self.subword_scores,
                                                          dim=2)  # Get probabilities for each label

                    self.predicted_label_keys = self.subword_scores_softmax.max(2)[1][0]
                    self.predicted_label_probabilities = self.subword_scores_softmax.max(2)[0][0].cpu().numpy()

                    self.labels = sorted(self.head.config.labels)
                    self.predicted_labels = [self.labels[label_key] for label_key in
                                             self.predicted_label_keys.cpu().numpy()]

                    # sentence_subword_length = self.sentence_encoding.special_tokens_mask.count(0)
                    token_mask = self.sentence_encoding.special_tokens_mask

                    # List of indices containing subwords
                    subwords_idx = [index_of_subword for index_of_subword, mask in enumerate(token_mask) if mask == 0]

                    self.predicted_label_probabilities = [self.predicted_label_probabilities[i] for i in subwords_idx]
                    self.output_tokens = [self.sentence_encoding.tokens[i] for i in subwords_idx]
                    # Print subword spans
                    self.output_spans_within_sentence = [
                        " ".join([str(span_idx) for span_idx in self.sentence_encoding.offsets[i]])
                        for i in subwords_idx]
                    self.output_spans_within_document = [" ".join(
                        [str(span_idx + self.sentence.start_char) for span_idx in self.sentence_encoding.offsets[i]])
                                                         for i in subwords_idx]
                    # Print labels
                    self.output_labels = [self.predicted_labels[i].replace("NP", model['entity_type']) for i in
                                          subwords_idx]  # Generalize to task type

                    # Update document output
                    self.output_dict['tokens'] = self.output_dict['tokens'] + self.output_tokens
                    self.output_dict['sentence_spans'] = self.output_dict[
                                                             'sentence_spans'] + self.output_spans_within_sentence
                    self.output_dict['document_spans'] = self.output_dict[
                                                             'document_spans'] + self.output_spans_within_document
                    self.output_dict['probability'] = self.output_dict[
                                                          'probability'] + self.predicted_label_probabilities
                    self.output_dict['labels'] = self.output_dict['labels'] + self.output_labels
                    annotation_end = time.time()
                    print(
                        f'finished sentence {sentence_idx} of {number_of_sentences} in {annotation_end - annotation_start:0.2f} seconds')

        if self.output_dict:
            self.output_table = pd.DataFrame.from_dict(self.output_dict)
            if output_filename:
                self.output_table.to_csv(output_filename_fullpath, sep='\t', header=True, index=True, index_label="#")
            else:
                self.output_table.to_csv(os.path.join(output_directory, 'example_output.tsv'), sep='\t', header=True,
                                         index=True, index_label="#")

    def run_all_documents(self, path_to_document_dir, output_directory=".", recursive=False):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_list = []
        if recursive:
            for root, directories, filenames in os.walk(path_to_document_dir):
                for filename in filenames:
                    file_list.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(path_to_document_dir):
                file_list.append(os.path.join(path_to_document_dir, filename))
        for input_document in file_list:
            if not input_document.endswith(".txt"):
                continue
            output_basename = os.path.basename(input_document).replace('.txt', '') + "_biobert_annotated"
            output_filename = output_basename + ".tsv"
            print(f'Running document {input_document}. \nSaving Results to {output_directory}/{output_filename}')
            self.run_document(input_document, output_filename, output_directory)

    def __str__(self):
        return self.document.sents


if __name__ == '__main__':
    start = time.time()
    # PATH_TO_VOCAB = 'models/vocab.txt'
    # data_dir = 'raw-data'
    # PATH_TO_MODEL = "../models"
    # PATH_TO_BASE_MODEL = r'C:/Users/User/nlp/projects/synbiobert/models/biobert_v1.1_pubmed'
    # PATH_TO_FILE = r'raw-data/ACS-100/sb6/sb6b00371.txt'
    # with open(PATH_TO_FILE, encoding='utf8') as f:
    #     document_as_string = f.read()  # does this work for large documents?

    # foo = InferNER(r"/home/rodriguezne2/results/multitasking_transformers/bert/run_2020_03_22_01_52_40_pine.cs.vcu.edu/SubwordClassificationHead_variome_species_checkpoint_10",
    # "SubwordClassificationHead_variome_species.json", device='cpu')
    config = yaml.safe_load(open('config.yml'))
    # config = yaml.safe_load(open('../Experiments/annotate_ACS100_20200410_0726/config.yml'))
    all_head_paths = sum(list(config['paths_to_heads'].values()), [])
    head_configs = [re.search("SubwordClassification.+json", filename) for path_to_head in all_head_paths for filename
                    in os.listdir(path_to_head)]
    head_configs = [x.group() for x in head_configs if x]

    foo = InferNER(all_head_paths, head_configs, device=config['device'])
    foo.run_all_documents(path_to_document_dir=config['path_to_documents'], output_directory=config['experiment_name'])

    ### RUN SINGLE SENTENCE ###
    # foo.run_single_example(document_as_string)

    ### RUN SINGLE DOCUMENT ###
    # foo.run_document(PATH_TO_FILE)

    ### RUN DOCUMENTS IN DIRECTORY
    # for i in range(3, 10):
    #     print(f'working on ACS-100/sb{i}')
    #     foo.run_all_documents(path_to_document_dir=f'../raw-data/ACS-100/sb{i}',
    #                           output_directory='huner_biobert_annotated')

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')
