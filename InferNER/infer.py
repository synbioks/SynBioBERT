"""predict entities using pre-trained/fine-tuned bert model"""

import os, time, json
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


class InferNER(object):
    def __init__(self, head_directory, head_config, device=None):
        """
        Create InferNER object
        :param head_directory: path to head
        :param head_config: name of head config file (not the base model config!)
        """
        # SET DEVICE
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # LOAD BASE MODEL
        print('Loading BERT pre-trained model')
        self.bert = BertModel.from_pretrained(head_directory, from_tf=False)

        # LOAD HEAD
        print('Loading Head')
        path_to_head_config = os.path.join(head_directory, head_config)
        path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        self.head_directory = head_directory
        self.head_config = BertConfig.from_pretrained(path_to_head_config)
        config = json.load(open(os.path.join(self.head_directory, head_config), 'rb'))
        self.head = SubwordClassificationHead(config['head_task'], labels=config['labels'])
        print(self.head.from_pretrained(self.head_directory))

        # LOAD TOKENIZER AND SET OPTIONS
        print('Loading Tokenizer and setting options')
        self.tokenizer = BertWordPieceTokenizer(path_to_vocab,
                                                lowercase=False)
        self.tokenizer.enable_padding(direction='right',
                                      pad_id=0,
                                      max_length=self.head_config.max_position_embeddings)

        # Construct Sentencizer
        head_name = os.path.basename(self.head_directory)
        self.sentencizer = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})  # TODO rename
        self.sentencizer.add_pipe(self.sentencizer.create_pipe("sentencizer"))

        print('Loaded BERT head, config, tokenizer, and sentencizer')
        # TODO predict each sentence in batches
        self.labels = sorted(self.head.config.labels)  # Fine-tuning may have been done on sorted labels.

    def run_document(self, path_to_document):
        with open(path_to_document, encoding='utf8') as f, \
            open('conll_test.txt', 'w') as outfile:
            document_as_string = f.read()  # does this work for large documents?

        self.output_dict = {'tokens': [],
                            'sentence_spans': [],
                            'document_spans': [],
                            'probability': [],
                            'labels': []
                            }

        sentencized_document = self.sentencizer(document_as_string)
        number_of_sentences = len(list(sentencized_document.sents))
        test_stop = number_of_sentences
        # number_of_sentences = test_stop
        for sentence_idx, sentence in enumerate(sentencized_document.sents):
            annotation_start = time.time()
            print(f'\nAnnotating sentence {sentence_idx} of {number_of_sentences}')
            if sentence_idx > test_stop:
                break

            self.sentence = sentence
            self.sentence_idx = sentence_idx
            # self.sentence = str(list(sentencized_document.sents)[0]) # TODO break off into a separate method to use one sentence
            # self.sentence = "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
            # self.sentence = r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."
            self.sentence_encoding = self.tokenizer.encode(self.sentence.string)

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
            self.bert.to(device=self.device)
            self.head.eval()
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
                self.subword_scores_softmax = softmax(self.subword_scores, dim=2)  # Get probabilities for each label

                self.predicted_label_keys = self.subword_scores_softmax.max(2)[1][0]
                self.predicted_label_probabilities = self.subword_scores_softmax.max(2)[0][0].cpu().numpy()

                self.labels = sorted(self.head.config.labels)
                self.predicted_labels = [self.labels[label_key] for label_key in self.predicted_label_keys.cpu().numpy()]

                # sentence_subword_length = self.sentence_encoding.special_tokens_mask.count(0)
                token_mask = self.sentence_encoding.special_tokens_mask

                # List of indices containing subwords
                subwords_idx = [index_of_subword for index_of_subword, mask in enumerate(token_mask) if mask == 0]

                self.predicted_label_probabilities = [self.predicted_label_probabilities[i] for i in subwords_idx]
                self.output_tokens = [self.sentence_encoding.tokens[i] for i in subwords_idx]
                # Print subword spans
                self.output_spans_within_sentence = [" ".join([str(span_idx) for span_idx in self.sentence_encoding.offsets[i]])
                                                     for i in subwords_idx]
                self.output_spans_within_document = [" ".join([str(span_idx + self.sentence.start_char) for span_idx in self.sentence_encoding.offsets[i]])
                                                     for i in subwords_idx]
                # Print labels
                self.output_labels = [self.predicted_labels[i].replace("NP", "Gene_BERT") for i in subwords_idx]  # Generalize to task type

                # Update document output
                self.output_dict['tokens'] = self.output_dict['tokens'] + self.output_tokens
                self.output_dict['sentence_spans'] = self.output_dict['sentence_spans'] + self.output_spans_within_sentence
                self.output_dict['document_spans'] = self.output_dict['document_spans'] + self.output_spans_within_document
                self.output_dict['probability'] = self.output_dict['probability'] + self.predicted_label_probabilities
                self.output_dict['labels'] = self.output_dict['labels'] + self.output_labels
                annotation_end = time.time()
                print(f'finished sentence {sentence_idx} of {number_of_sentences} in {annotation_end-annotation_start:0.2f} seconds')

        self.output_table = pd.DataFrame.from_dict(self.output_dict)
        foo.output_table.to_csv('example_output.tsv', sep='\t', header=True, index=True, index_label="#")

    def run_single_example(self, text):
        # head_name = os.path.basename(self.head_directory)
        # nlp = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})
        # nlp.add_pipe(nlp.create_pipe("sentencizer"))
        sentencized_document = self.sentencizer(text)
        # self.sentence = str(list(sentencized_document.sents)[0]) # TODO break off into a separate method to use one sentence
        # self.sentence = "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
        # self.sentence = r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."
        self.sentence_encoding = self.tokenizer.encode(self.sentence)

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
        self.bert.to(device=self.device)
        with torch.no_grad():
            print(f"Predicting {self.head}")
            self.bert_outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                # token_type_ids=token_type_ids,
                                token_type_ids=None,
                                position_ids=None)[0]
            self.subword_scores = self.head(self.bert_outputs)[0]
            # self.predicted_label_keys = self.subword_scores.max(dim=2).indices[0]
            self.predicted_label_keys = self.subword_scores.max(2)[1][0]  # to run in batch mode, [0] to i

            self.labels = sorted(self.head.config.labels)
            self.predicted_labels = [self.labels[label_key] for label_key in self.predicted_label_keys]

            sentence_subword_length = self.sentence_encoding.special_tokens_mask.count(0)
            token_mask = self.sentence_encoding.special_tokens_mask

            subwords_idx = [index_of_subword for index_of_subword, mask in enumerate(token_mask) if mask==0]

            # Print tokenized sentence
            self.output_tokens = [self.sentence_encoding.tokens[i] for i in subwords_idx]
            # Print subword spans
            self.output_spans = [str(self.sentence_encoding.offsets[i]) for i in subwords_idx]
            # Print labels
            self.output_labels = [self.predicted_labels[i] for i in subwords_idx]
            self.output_table = pd.DataFrame.from_dict(
                {'tokens': self.output_tokens, 'labels': self.output_labels, 'spans': self.output_spans})

    def run_all_documents(self, path_to_document_dir):
        # os.path.
        pass

    def __str__(self):
        return self.document.sents

start = time.time()
PATH_TO_VOCAB = 'models/vocab.txt'
# data_dir = 'raw-data'
# PATH_TO_MODEL = "../models"
PATH_TO_FILE = r'raw-data/ACS-100/sb6/sb6b00371.txt'
# PATH_TO_BASE_MODEL = r'C:/Users/User/nlp/projects/synbiobert/models/biobert_v1.1_pubmed'
with open(PATH_TO_FILE, encoding='utf8') as f:
    document_as_string = f.read()  # does this work for large documents?
# foo = InferNER(r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20")
foo = InferNER(r"/home/rodriguezne2/results/multitasking_transformers/bert/run_2020_03_22_01_52_40_pine.cs.vcu.edu/SubwordClassificationHead_variome_species_checkpoint_10",
               "SubwordClassificationHead_variome_species.json", device='cpu')
# foo.run_single_example(document_as_string)
foo.run_document(PATH_TO_FILE)
# foo.infer_document(document_as_string)
end = time.time()
print(f'Finished in {end - start:0.2f} seconds')

# Note: document prep is lossy
# TODO output conll
# TODO create output summary method
# TODO download other model
