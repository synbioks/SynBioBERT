from spacy_transformers import TransformersLanguage, TransformersWordPiecer, \
    TransformersTok2Vec
from spacy_transformers.pipeline.ner import TransformersEntityRecognizer
from spacy_transformers.util import get_tokenizer
import time

head = r"SubwordClassificationHead_iepa_gene_checkpoint_20"
path = r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20"
from_tf = False

nlp = TransformersLanguage(trf_name=head, meta={"lang": "en"})  # TODO rename
nlp.add_pipe(nlp.create_pipe("sentencizer"))
bert_tokenizer = get_tokenizer('bert').from_pretrained(path, do_lower_case=False)
nlp.add_pipe(TransformersWordPiecer(nlp.vocab, model=bert_tokenizer, trf_name=path))
nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, path))


PATH_TO_FILE = r'../raw-data/ACS-100/sb3/sb3000723.txt'
# TODO alignment
with open(PATH_TO_FILE, encoding='utf-8') as f:
    start = time.time()
    doc = nlp(f.read())
    end = time.time()
    print(f'Finished sentencizing and subword tokenizing in {end-start:0.2f} seconds')

ner = TransformersEntityRecognizer(nlp.vocab)



