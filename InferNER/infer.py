'''predict entities using pre-trained/fine-tuned bert model'''

import os
import logging
import torch
from transformers import BertConfig, BertModel, BertForTokenClassification
from tokenizers import BertWordPieceTokenizer

log = logging.getLogger('root')

class InferNER(object):
    """ """
    def __init__(self, base_model_directory, head_directory, use_tf_model=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path_to_vocab = os.path.join(head_directory, 'vocab.txt')
        path_to_head_config = os.path.join(head_directory, 'config.json')

        ## LOAD MODEL
        self.head_config = BertConfig.from_json_file(path_to_head_config)

        # use_tf_model in the case that you're using a pre-trained model from a tensorflow checkpoint.
        if not use_tf_model:
            use_tf_model = 'biobert_v1' in base_model_directory \
                       or 'biobert_large' in base_model_directory

        self.bert = BertForTokenClassification.from_pretrained(head_directory, config=self.head_config)
        self.tokenizer = BertWordPieceTokenizer(path_to_vocab,
                                                lowercase=False)
        print('Loaded BERT head, config, and tokenizer')




    def infer(self):
        # Write argmax label to .ann
        # Spacy?
        pass

    def predict(self, text):
        input_ids, input_mask = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
        return output


    def _preprocess(self, text):
        """ """
        tokenized_document = self.tokenizer.encode(text)
        input_ids = []
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        return input_ids,input_mask


PATH_TO_VOCAB = 'models/vocab.txt'
# data_dir = 'raw-data'
PATH_TO_MODEL = "../models"
PATH_TO_FILE = '../raw-data/ACS-100/sb3/sb3000723.txt'
# with open(path_to_file, encoding='utf8') as f:
#     document_as_string = f.read()  # does this work for large documents?
# foo = InferNER(base_model_directory="models/biobert_v1.1_pubmed",
#          head_directory="models/SubwordClassificationHead_iepa_gene_checkpoint_20",
#          use_tf_model=True)




# TODO Check for 512 token max + padding
# TODO predict and infer

