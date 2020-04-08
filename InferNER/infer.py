'''predict entities using pre-trained/fine-tuned bert model'''

import os, time, json
import torch
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
        self.attention_mask = None # Mask padding. Needed.
        self.token_type_ids = None # Segment ids in [0,1] Not needed, but couldn't hurt.
        self.position_ids = None
        self.inputs_embeds = None # Not needed
        self.labels = None # Not needed for inference. torch.tensor([1] * input_ids.size(1)).unqueeze(0) # Batch size 1

class InferNER(object):
    def __init__(self, head_directory, bert_directory):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ## Load Base model
        self.bert = BertModel.from_pretrained(head_directory, from_tf=False)

        ## Load Head
        path_to_head_config = os.path.join(head_directory, 'SubwordClassificationHead_iepa_gene.json')
        path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        # path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        self.head_directory = head_directory
        self.head_config = BertConfig.from_pretrained(path_to_head_config)

        # ## Load Tokenizer
        self.tokenizer = BertWordPieceTokenizer(path_to_vocab,
                                                lowercase=False)
        self.tokenizer.enable_padding(direction='right',
                                      pad_id=0,
                                      max_length=self.head_config.max_position_embeddings)
        config = json.load(open(os.path.join(self.head_directory, "SubwordClassificationHead_iepa_gene.json"), 'rb'))
        self.head = SubwordClassificationHead('iepa_gene', labels=config['labels'])
        print(self.head.from_pretrained(self.head_directory))

        # # print('Loaded BERT head, config, and tokenizer')
        # #
        # # self.document = None
        # # self.input_ids = None
        # # self.attention_mask = None # Mask padding. Needed.
        # # self.labels = None # Not needed. torch.tensor([1] * input_ids.size(1)).unqueeze(0) # Batch size 1
        # # self.outputs = None # model(input_ids, labels=labels)
        # # # TODO replace with tokenizers version. run on each sentence.
        # # # TODO predict each sentence (in batches?)
        # TODO write to .ann file

    def run_single_example(self, text):
        head_name = os.path.basename(self.head_directory)
        nlp = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})  # TODO rename
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        sentencized_document = nlp(text)
        # self.sentence = str(list(sentencized_document.sents)[0]) # TODO break off into a separate method to use one sentence
        # self.sentence = "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
        self.sentence = r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."
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



            # label_list = ["LABEL_0", "LABEL_1"]
            # print([(token, label_list[prediction]) \
            #        for token, prediction \
            #        in zip(self.sentence_encoding.tokens, self.predictions[0].tolist())])
        #
        #
        # # TODO position ids


    def infer(self, text):
        # Write argmax label to .ann
        # Spacy?
        # self.document = self.preprocess(text)
        # self.input_ids = torch
        # self.input_ids(torch.tensor(self.tokenizer.encode(test_sent)))
        pass

    def predict(self, text):
        pass

    def preprocess(self, text):
        """Input: text (str)
        Updates InferNER.sents with sentences (generator)
        """
        return sentencize(text)

    def __str__(self):
        return(str(list(self.document.sents)))


start = time.time()
PATH_TO_VOCAB = 'models/vocab.txt'
# data_dir = 'raw-data'
# PATH_TO_MODEL = "../models"
PATH_TO_FILE = r'raw-data/ACS-100/sb3/sb3000723.txt'
PATH_TO_BASE_MODEL = r'C:/Users/User/nlp/projects/synbiobert/models/biobert_v1.1_pubmed'
with open(PATH_TO_FILE, encoding='utf8') as f:
    document_as_string = f.read()  # does this work for large documents?
foo = InferNER(r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20",
               PATH_TO_BASE_MODEL)
foo.run_single_example(document_as_string)
end = time.time()
print(f'Finished in {end - start:0.2f} seconds')

# TODO Check for 512 token max + padding
# TODO predict and infer

# "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."