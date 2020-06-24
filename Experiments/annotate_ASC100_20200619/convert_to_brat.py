from InferNER import InferNER
import os


class InferNERUtils(InferNER):
    def __init__(self, head_directories, head_configs, model_loading_device=None, document=""):
        super(InferNERUtils, self).__init__(head_directories, head_configs, model_loading_device)
        self.document = document

    def load_document(self, path_to_document):
        assert path_to_document.endswith('.txt')

        # LOAD RAW DATA AND SENTENCIZE
        with open(path_to_document, encoding='utf8') as f:
            self.document = f.read() # does this work for large documents?
        return self.document

    def sentencize_document(self):
        sentencized_document = self.sentencizer(self.document)
        return sentencized_document

    def join_subword_predictions(self, subword_predictions):
        annotations = []
        for subword in subword_predictions:
            mapped_subword = self.document[subword.start: subword.end]
            test_subword_mapping(mapped_subword, subword.token)
            subword.original_token = mapped_subword
            if not subword.token.startswith('##'):
                annotations.append(BertAnnotation(subword.original_token,
                                                  subword.start,
                                                  subword.end,
                                                  subword.label.title(),
                                                  subword.score
                                                  )
                                   )
            else:
                anno = annotations[-1]
                anno.end = subword.end
                anno.token = self.document[anno.start: subword.end]  # TODO instead, remap new span
        return annotations

    def bert_to_brat(self, annotations):
        # annotations = [BratAnnotation(token, start, end, label)
        #                for token, start, end, label, score in predictions]
        brat_annotations = []
        T_index = 1
        previous = Annotation('', 0, 0, 'O')

        for i, current in enumerate(annotations):
            # current.label = current.label if current.label.startswith(('B-', 'I-')) else 'O'  # changes BERT_TOKEN to 'O'

            if current.label.startswith('B-') or (
                    current.label.startswith('I-') and previous.label in ['O', 'Bert_Token']):
                current.label = current.label.replace('B-', '').replace('I-', '')
                current.index = T_index
                brat_annotations.append(current)
                T_index += 1
                previous = current

            elif current.label.startswith('I-'):
                anno = brat_annotations[-1]
                anno.token = self.document[anno.start: current.end]
                anno.end = current.end
                previous = current

            else:
                previous = current
        return brat_annotations

    @staticmethod
    def parse_predictions(path):
        with open(path, 'r') as f:
            next(f)  # skip header
            predictions = []
            for line in f:
                _, token, _, offsets, score, label = line.strip().split("\t")
                predictions.append(
                    BertAnnotation(
                        token=token,
                        start=int(offsets.split(' ')[0]),
                        end=int(offsets.split(' ')[1]),
                        score=float(score),
                        label=label
                    )
                )
        return predictions

    def write_brat(self, brat_annotations, output_path: str):
        outfile_ann_path = f'{output_path}.ann'
        with open(outfile_ann_path, 'w') as outfile_ann:
            for anno in brat_annotations:
                outfile_ann.write(f'T{anno.index}\t{anno.label} {anno.start} {anno.end}\t{anno.token}\n')

        outfile_text_path = f'{output_path}.txt'
        with open(outfile_text_path, 'w') as outfile_text:
            outfile_text.write(self.document)

    def encode_sentences(self, sentencized_document):
        encodings = []
        output_lines = []
        last_sentence_position = 0
        for i, sentence in enumerate(sentencized_document.sents):
            encoding = self.tokenizer.encode(sentence.string)
            attention_mask = encoding.attention_mask
            text_normalized = encoding.normalized_str
            text_original = encoding.original_str
            overflowing = encoding.overflowing
            tokens = encoding.tokens
            offsets = encoding.offsets
            encodings.append(encoding)
        assert len(list(sentencized_document.sents)) == len(encodings)
        return encodings


class Annotation:
    def __init__(self,
                 token: str,
                 start: int,
                 end: int,
                 label: str,
                 ):
        self.token = token
        self.start = start
        self.end = end
        self.label = label

    def __str__(self):
        label = f"-{self.label}" if not self.label == 'O' else ""
        return f"{self.token}{label}"


class BertAnnotation(Annotation):
    def __init__(self,
                 token: str,
                 start: int,
                 end: int,
                 label: str,
                 score: float,
                 original_token=None):
        super(BertAnnotation, self).__init__(token, start, end, label)
        self.score = score
        self.original_token = original_token


class BratAnnotation(Annotation):
    def __init__(self,
                 token: str,
                 start: int,
                 end: int,
                 label: str,
                 index=None):
        """
        index: Text-bound index. See BRAT standoff format.
        """
        super(BratAnnotation, self).__init__(token, start, end, label)
        self.index = index


def test_subword_mapping(mapped_subword, subword_token):
    import warnings
    if not mapped_subword == subword_token.replace('##', ''):
         if not len(mapped_subword) == len(subword_token): \
            warnings.warn(f'"{mapped_subword}": {len(mapped_subword)} "{subword_token}": {len(subword_token)}')


def filter_document_by_id(documents, document_id):
    """Returns the first document who's filename begins with the document_id"""
    return [path for path in documents if os.path.basename(path).startswith(document_id)][0]


if __name__ == '__main__':
    '''EXAMPLE
    head_dir = ["/home/nick/projects/synbiobert/models/SubwordClassificationHead_iepa_gene_checkpoint_20"]
    head_configs = [os.path.join(head_dir[0], "SubwordClassificationHead_iepa_gene.json")]
    document_path = "/home/nick/projects/synbiobert/raw-data/ACS-100/sb6/sb6b00053.txt"
    subword_predictions_path = "/home/nick/projects/synbiobert/Experiments/annotate_ASC100_20200619/subtoken_label_probabilities/sb6b00053_biobert_annotated.tsv"
    config = {'device': 'cpu'}
    output_ann_path = os.path.join("/home/nick/Desktop", "test_infer_brat")
    
    inferner = InferNERUtils(head_dir, head_configs, model_loading_device=config['device'])
    inferner.sentencize_document(path_to_document=document_path)
    subword_predictions = inferner.parse_predictions(subword_predictions_path)
    predictions = inferner.join_subword_predictions(subword_predictions)  # TODO add method to InferNER
    brat_annotations = bert_to_brat(predictions)
    inferner.write_brat(output_ann_path)
    '''
    import time
    import yaml

    start = time.time()

    # LOAD DATA AND MODELS
    with open('convert_to_brat_config.yml') as yml:
        config = yaml.safe_load(yml)
    head_dir = config['paths_to_heads']['chemical']
    head_configs = [os.path.join(head_dir[0], f) for f in os.listdir(head_dir[0]) if
                    f.endswith('.json') and f.startswith("SubwordClassification")]

    document_dir = "/home/nick/projects/synbiobert/raw-data/ACS-100"
    document_paths = []
    for root, directories, filenames in os.walk(document_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                document_paths.append(os.path.join(root, filename))

    subword_predictions_dir = "subtoken_label_probabilities"
    subword_predictions_paths = []
    for root, directories, filenames in os.walk(subword_predictions_dir):
        for filename in filenames:
            if filename.endswith('.tsv'):
                subword_predictions_paths.append(os.path.join(root, filename))
    document_ids = [os.path.basename(x).replace('_biobert_annotated.tsv', '') for x in subword_predictions_paths]
    if not os.path.exists('results/data'):
        os.makedirs('results/data')


    # BRAT CONVERSION
    for document_id, subword_predictions_path in zip(document_ids, subword_predictions_paths):
        document_path = filter_document_by_id(document_paths, document_id)
        output_ann_path = os.path.join("results/data", f"{document_id}")
        inferner = InferNERUtils(head_dir, head_configs, model_loading_device=config['device'])
        inferner.load_document(path_to_document=document_path)
        subword_predictions = inferner.parse_predictions(subword_predictions_path)
        predictions = inferner.join_subword_predictions(subword_predictions)
        brat_annotations = inferner.bert_to_brat(predictions)
        inferner.write_brat(brat_annotations, output_ann_path)

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')
