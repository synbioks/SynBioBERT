import time
from spacy_transformers import TransformersLanguage, TransformersWordPiecer, \
    TransformersTok2Vec
from spacy_transformers.pipeline.ner import TransformersEntityRecognizer
from spacy_transformers.util import get_tokenizer
# from InferNER.ner import TransformersEntityRecognizer

def sentencize(text,
               head = r"SubwordClassificationHead_iepa_gene_checkpoint_20",
               path = r"C:\Users\User\nlp\projects\synbiobert\models\SubwordClassificationHead_iepa_gene_checkpoint_20",
               from_tf = False
               ):
    nlp = TransformersLanguage(trf_name=head, meta={"lang": "en"})  # TODO rename
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    bert_tokenizer = get_tokenizer('bert').from_pretrained(path, do_lower_case=False)
    nlp.add_pipe(TransformersWordPiecer(nlp.vocab, model=bert_tokenizer, trf_name=path))
    nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, path))
    doc = nlp(text)
    return doc

PATH_TO_FILE = r'../raw-data/ACS-100/sb3/sb3000723.txt'
with open(PATH_TO_FILE, encoding='utf-8') as f:
    start = time.time()
    x = sentencize(f.read())
    end = time.time()
    print(f'Finished sentencizing and subword tokenizing with Spacy in {end-start:0.2f} seconds')
    # start = time.time()
    # doc = pt_preprocess(f.read())
    # end = time.time()
    print(f'Finished sentencizing and subword tokenizing with PyTorch in {end-start:0.2f} seconds')

# ner = TransformersEntityRecognizer(nlp.vocab)



