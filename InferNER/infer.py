'''predict entities using pre-trained/fine-tuned bert model'''

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


# log = logging.getLogger('root')

class Sentence(object):
    def __init__(self):
        self.input_ids = None
        self.attention_mask = None  # Mask padding. Needed.
        self.token_type_ids = None  # Segment ids in [0,1] Not needed, but couldn't hurt.
        self.position_ids = None
        self.inputs_embeds = None  # Not needed
        self.labels = None  # Not needed for inference. torch.tensor([1] * input_ids.size(1)).unqueeze(0) # Batch size 1


class InferNER(object):
    def __init__(self, head_directory):
        # SET DEVICE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # LOAD BASE MODEL
        print('Loading BERT pre-trained model')
        self.bert = BertModel.from_pretrained(head_directory, from_tf=False)

        # LOAD HEAD
        print('Loading Head')
        path_to_head_config = os.path.join(head_directory, 'SubwordClassificationHead_iepa_gene.json')
        path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        self.head_directory = head_directory
        self.head_config = BertConfig.from_pretrained(path_to_head_config)
        config = json.load(open(os.path.join(self.head_directory, "SubwordClassificationHead_iepa_gene.json"), 'rb'))
        self.head = SubwordClassificationHead('iepa_gene', labels=config['labels'])
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
        # TODO predict each sentence (in batches?)
        # TODO write to conll file
        # self.document = None  # List of Spacy spans containing sentences
        # self.tokens = None  #
        # self.sentence_encodings = None
        # self.token_type_ids = None
        # self.spans = None
        # self.input_ids = None
        # self.attention_mask =  None
        # self.bert_outputs = None
        # self.document_sentencized_generator = None  # Spacy Doc object
        # self.subword_scores = None
        # self.predicted_label_keys = None
        # self.labels = sorted(self.head.config.labels)  # Fine-tuning may have been done on sorted labels.
        # self.predicted_labels = None
        # self.output_tokens = None

    def infer_document(self, text):
        """
        :param text: A string containing the text of a document.
        :return: A string containing annotations for all entities in Conll format.
        """
        print('Preparing document...')
        prepare_doc_start = time.time()
        self.prepare_document(text)
        prepare_doc_end = time.time()
        print(f'Finished in {prepare_doc_end - prepare_doc_start:0.2f} seconds')

        print('Preparing model inputs...')
        prepare_model_input_start = time.time()
        self.prepare_model_inputs()
        prepare_model_input_end = time.time()
        print(f'Finished in {prepare_model_input_end - prepare_model_input_end:0.2f} seconds')

        print('Predicting labels')
        self.predict()

    def prepare_document(self, text):
        """ Uses spacy sentencizer.
        """
        self.document_sentencized_generator = nlp(text)
        self.document = list(self.document_sentencized_generator.sents)
        self.sentence_encodings = [self.tokenizer.encode(str(sent)) for sent in self.document]  # The Spacy to HF-Tokenizers handoff.

    def prepare_model_inputs(self):
        self.input_ids = torch.tensor([sent.ids for sent in self.sentence_encodings], dtype=torch.long, device=self.device)
        self.attention_mask = torch.tensor([sent.attention_mask for sent in self.sentence_encodings], dtype=torch.long, device=self.device)
        self.token_type_ids = torch.tensor([sent.type_ids for sent in self.sentence_encodings], dtype=torch.long, device=self.device)
        self.tokens = [sent.tokens for sent in self.sentence_encodings]
        self.spans = [sent.offsets for sent in self.sentence_encodings]

    def predict(self):
        """
        Predict labels for a document
        """
        assert len(self.input_ids) == len(self.attention_mask)
        self.bert.eval()
        # self.head.eval()
        with torch.no_grad():
            for i in range(0, len(self.input_ids)):
                if i > 0:
                    break
                input_ids, attention_mask = self.input_ids[i].unsqueeze(0), self.attention_mask[i].unsqueeze(0)
                # print(input_ids)
                start = time.time()
                print(f"Predicting {self.head}")
                input_ids = input_ids.to(device=self.device)
                attention_mask = attention_mask.to(device=self.device)
                self.bert_outputs = self.bert(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              # token_type_ids=token_type_ids,
                                              token_type_ids=None,
                                              position_ids=None)
                self.subword_scores = self.head(self.bert_outputs[0])[0]  # Pass transformer output to head. [0] are the last hidden states.

                self.predicted_label_keys = self.subword_scores.max(2)[1][0]  # The max function returns (values, indices), so [1] is the indices at the argmax. Positional indices=key in key: label. to run in batch mode, [0] to i.
                self.predicted_labels = [self.labels[label_key] for label_key in
                                         self.predicted_label_keys]  # subword token labels.

                token_mask = self.sentence_encodings[i].special_tokens_mask

                subwords_idx = [index_of_subword for index_of_subword, mask in enumerate(token_mask) if mask == 0]

                # Print tokenized sentence
                self.output_tokens = [self.sentence_encodings[i].tokens[j] for j in subwords_idx]
                # Print subword spans
                self.output_spans = [str(self.sentence_encodings[i].offsets[j]) for j in subwords_idx]
                # Print labels
                self.output_labels = [self.predicted_labels[j] for j in subwords_idx]
                self.output_table = pd.DataFrame.from_dict(
                    {'tokens': self.output_tokens, 'labels': self.output_labels, 'spans': self.output_spans})
                print(self.output_table)
                end = time.time()
                print(f'Finished {i+1} of {len(self.input_ids)} predictions in {end - start:0.2f} seconds')

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
        input_ids = torch.tensor([self.sentence_encoding.ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([self.sentence_encoding.attention_mask], dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor([self.sentence_encoding.type_ids], dtype=torch.long, device=self.device)
        self.document = sentencized_document
        self.tokens = self.sentence_encoding.tokens
        self.spans = self.sentence_encoding.offsets
        self.input_ids = input_ids

        # RUN EXAMPLE THROUGH BERT
        self.bert.eval()
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
        test_stop = 20
        number_of_sentences = test_stop
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
            input_ids = torch.tensor([self.sentence_encoding.ids], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([self.sentence_encoding.attention_mask], dtype=torch.long, device=self.device)
            token_type_ids = torch.tensor([self.sentence_encoding.type_ids], dtype=torch.long, device=self.device)
            self.document = sentencized_document
            self.tokens = self.sentence_encoding.tokens
            self.spans = self.sentence_encoding.offsets
            self.input_ids = input_ids

            # RUN EXAMPLE THROUGH BERT
            self.bert.eval()
            self.head.eval()
            with torch.no_grad():
                print(f"BERT Head: {self.head}")
                self.bert_outputs = self.bert(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              # token_type_ids=token_type_ids,
                                              token_type_ids=None,
                                              position_ids=None)[0]
                self.subword_scores = self.head(self.bert_outputs)[0]
                # self.predicted_label_keys = self.subword_scores.max(dim=2).indices[0]
                # self.predicted_label_keys = self.subword_scores.max(2)[1][0]  # to run in batch mode, [0] to i
                self.subword_scores_softmax = softmax(self.subword_scores, dim=2)  # Get probabilities for each label

                self.predicted_label_keys = self.subword_scores_softmax.max(2)[1][0]
                self.predicted_label_probabilities = self.subword_scores_softmax.max(2)[0][0].numpy()

                self.labels = sorted(self.head.config.labels)
                self.predicted_labels = [self.labels[label_key] for label_key in self.predicted_label_keys.numpy()]

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
                # self.output_table = pd.DataFrame.from_dict(
                #     {'tokens': self.output_tokens,
                #      'sentence_spans': self.output_spans_within_sentence,
                #      'document_spans': self.output_spans_within_document,
                #      'labels': self.output_labels
                #      })

                annotation_end = time.time()
                print(f'finished sentence {sentence_idx} of {number_of_sentences} in {annotation_end-annotation_start:0.2f} seconds')
        self.output_table = pd.DataFrame.from_dict(self.output_dict)
        foo.output_table.to_csv('example_output.tsv', sep='\t', header=True, index=True)


    def run_all_documents(self, path_to_document_dir):
        # os.path.
        pass

    def __str__(self):
        return self.document.sents

start = time.time()
PATH_TO_VOCAB = 'models/vocab.txt'
# data_dir = 'raw-data'
# PATH_TO_MODEL = "../models"
PATH_TO_FILE = r'C:/Users/User/nlp/projects/synbiobert/raw-data/ACS-100/sb6/sb6b00371.txt'
# PATH_TO_BASE_MODEL = r'C:/Users/User/nlp/projects/synbiobert/models/biobert_v1.1_pubmed'
with open(PATH_TO_FILE, encoding='utf8') as f:
    document_as_string = f.read()  # does this work for large documents?
foo = InferNER(r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20")
# foo.run_single_example(document_as_string)
foo.run_document(PATH_TO_FILE)
# foo.infer_document(document_as_string)
end = time.time()
print(f'Finished in {end - start:0.2f} seconds')

# Note: document prep is lossy
# TODO output conll
# TODO create output summary method
# TODO download other model
