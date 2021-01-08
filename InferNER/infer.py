"""predict entities using pre-trained/fine-tuned bert model
Changes:
    Jun 26, 2020: added 'device' param to the model loader to allow for cpu loading.
"""

import os, time, json, yaml, re, argparse
import torch
from torch.nn.functional import softmax
from multitasking_transformers.heads import SubwordClassificationHead
from transformers import BertConfig, BertForTokenClassification, BertModel
# Transformers https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
from tokenizers import BertWordPieceTokenizer
# Tokenizers https://github.com/huggingface/tokenizers/tree/master/bindings/python
from spacy_transformers import TransformersLanguage
import pandas as pd


# TODO predict each sentence in batches

class InferNER(object):

    def __init__(self, head_directories, head_configs, device=None,
                 from_huner=False, lowercase=False):
        """

        :param head_directories: list containing the directory paths to the head models.
        :param head_configs: a list containing the paths to the head config files.
        :param device: One of 'cpu' or 'cuda'. Defaults to 'cpu'.
        :param lowercase: preprocessing option. If predicting an entity type,
            like Gene, where the case matters, set to False (default).
        """
        # SET DEVICE
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        assert len(head_directories) == len(head_configs)

        # LOAD TOKENIZER AND MODELS
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

            # Collect models
            self.models.append({'head': self.head,
                                'base': self.bert,
                                'entity_type': head.split('_')[-3],
                                'dataset': head.split('_')[-4]})

        # LOAD TOKENIZER AND SET OPTIONS
        print('Loading Tokenizer and setting options')
        self.tokenizer = BertWordPieceTokenizer(self.path_to_vocab,  # uses last head loaded for vocab
                                                lowercase=lowercase)
        self.tokenizer.enable_padding(direction='right',
                                      pad_id=0,
                                      max_length=self.head_config.max_position_embeddings)

        # CONSTRUCT PROCESSORS
        head_name = os.path.basename(self.head_directory)
        self.sentencizer = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})
        self.sentencizer.add_pipe(self.sentencizer.create_pipe("sentencizer"))

        print('Loaded BERT head, config, tokenizer, and sentencizer')
        self.labels = sorted(self.head.config.labels)  # Fine-tuning may have been done on sorted labels.

        self.from_huner = from_huner

    def run_document(self, path_to_document, output_filename=None, output_directory="."):

        output_filename_fullpath = os.path.join(output_directory, output_filename)
        if os.path.exists(output_filename_fullpath):
            print(f"{output_filename_fullpath} exists, skipping")
            return

        with open(path_to_document, encoding='utf-8') as f:
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
                self.sentence_encoding = self.tokenizer.encode(self.sentence.string)
                if len(self.sentence_encoding) > 512:
                    print(f"In document {os.path.basename(output_filename)}, this sentence exeeds the maximum token sequence size\n{self.sentence}")
                    print("Skipping document")
                    raise Exception

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
        print('started running all documents')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_list = []
        if recursive:
            print(f'Looking for files to add in {path_to_document_dir}. Searching Recursively')
            for root, directories, filenames in os.walk(path_to_document_dir):
                for filename in filenames:
                    file_list.append(os.path.join(root, filename))
        else:
            print(f'Looking for files to add. Searching {path_to_document_dir}')
            for filename in os.listdir(path_to_document_dir):
                file_list.append(os.path.join(path_to_document_dir, filename))

        log = open('infer.log', 'a')
        failed_list_log = open('infer_failed_list.log', 'a')
        for input_document in file_list:
            if not input_document.endswith(".txt"):
                continue
            output_basename = os.path.basename(input_document).replace('.txt', '') + "_biobert_annotated"
            output_filename = output_basename + ".tsv"

            # Check if the out file exists already, if so skip it.
            if os.path.exists(os.path.join(output_directory, output_filename)):
                print(f'Skipping document {input_document}. \nResults already in {output_directory}/{output_filename}')
                continue
            print(f'Running document {input_document}. \nSaving Results to {output_directory}/{output_filename}')
            try:
                self.run_document(input_document, output_filename, output_directory)
            except Exception as e:
                print(f"Failed to process {output_filename}. See log for error.")
                print(f"Failed to process {output_filename}: {e}", file=log)
                print(f"{output_filename}", file=failed_list_log)
            finally:
                pass

    def __str__(self):
        return self.document.sents


if __name__ == '__main__':

    start = time.time()
    config = yaml.safe_load(open('config.yml'))
    print(config)
    all_head_paths = sum(list(config['paths_to_heads'].values()), [])
    head_configs = [re.search("SubwordClassification.+json", filename) for path_to_head in all_head_paths for filename
                    in os.listdir(path_to_head)]
    head_configs = [x.group() for x in head_configs if x]

    foo = InferNER(all_head_paths, head_configs)
    foo.run_all_documents(path_to_document_dir=config['path_to_documents'], output_directory=config['experiment_name'])

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')
