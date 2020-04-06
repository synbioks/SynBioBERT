'''predict entities using pre-trained/fine-tuned bert model'''

import os, time, json
import torch
from multi_tasking_transformers.heads import SubwordClassificationHead
# from multi_tasking_transformers.multitaskers import MultiTaskingBert
from transformers import BertConfig, BertForTokenClassification
# Transformers https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
from tokenizers import BertWordPieceTokenizer
# Tokenizers https://github.com/huggingface/tokenizers/tree/master/bindings/python
from spacy_transformers import TransformersLanguage

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
    def __init__(self, head_directory):
        self.head_directory = head_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        path_to_head_config = os.path.join(head_directory, 'config.json')

        ## LOAD MODEL
        self.head_config = BertConfig.from_pretrained(path_to_head_config)
        self.tokenizer = BertWordPieceTokenizer(path_to_vocab,
                                                lowercase=False)
        self.tokenizer.enable_padding(direction='right',
                                      pad_id=0,
                                      max_length=self.head_config.max_position_embeddings)
        config = json.load(open(os.path.join(self.head_directory, "SubwordClassificationHead_iepa_gene.json"), 'rb'))
        head = SubwordClassificationHead('iepa_gene', labels=config['labels'])
        head.from_pretrained(self.head_directory)
        print(head.config)
        # #
        # self.bert = BertForTokenClassification.from_pretrained(head_directory,
                                                               # config=self.head_config)
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

    def run_single_example(self, text, ):
        # head = r"SubwordClassificationHead_iepa_gene_checkpoint_20"
        head = os.path.basename(self.head_directory)
        nlp = TransformersLanguage(trf_name=head, meta={"lang": "en"})  # TODO rename
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        sentencized_document = nlp(text)
        sentence = str(list(sentencized_document.sents)[0])
        sentence_encoding = self.tokenizer.encode(sentence)
        # PREPARE MODEL INPUT
        input_ids = torch.tensor([sentence_encoding.ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([sentence_encoding.attention_mask], dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor([sentence_encoding.type_ids], dtype=torch.long, device=self.device)
        self.document = sentencized_document
        self.input_ids = input_ids
        self.sentence_encoding = sentence_encoding
        # self.bert.eval()
        # with torch.no_grad():
        #     self.outputs = self.bert(input_ids=input_ids,
        #                         attention_mask=attention_mask,
        #                         token_type_ids=token_type_ids,
        #                         position_ids=None)[0]
        #     self.predictions = torch.argmax(self.outputs, dim=2)
        #     label_list = ["LABEL_0", "LABEL_1"]
        #     print([(token, label_list[prediction]) \
        #            for token, prediction \
        #            in zip(self.sentence_encoding.tokens, self.predictions[0].tolist())])
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
with open(PATH_TO_FILE, encoding='utf8') as f:
    document_as_string = f.read()  # does this work for large documents?
foo = InferNER(r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20")
foo.run_single_example(document_as_string)
end = time.time()
print(f'Finished in {end - start:0.2f} seconds')

# TODO Check for 512 token max + padding
# TODO predict and infer

