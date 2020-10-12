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


def gather_documents(document_directory):
    global document_paths, root, directories, filenames, filename

    document_paths = []
    for root, directories, filenames in os.walk(document_directory):
        for filename in filenames:
            if filename.endswith('.txt'):
                document_paths.append(os.path.join(root, filename))
    return document_paths

def load_models(config_yml):
    with open(config_yml) as yml:
        config = yaml.safe_load(yml)
    head_dir = config['paths_to_heads']['chemical']
    head_configs = [os.path.join(head_dir[0], f) for f in os.listdir(head_dir[0]) if
                    f.endswith('.json') and f.startswith("SubwordClassification")]
    return config, head_configs, head_dir


def gather_predictions(subword_predictions_directory):
    subword_predictions_paths = []
    for root, directories, filenames in os.walk(subword_predictions_directory):
        for filename in filenames:
            if filename.endswith('.tsv'):
                subword_predictions_paths.append(os.path.join(root, filename))
    doc_ids = [os.path.basename(x).replace('_biobert_annotated.tsv', '') for x in subword_predictions_paths]
    if not os.path.exists('results/data'):
        os.makedirs('results/data')
    return doc_ids, subword_predictions_paths


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

    document_dir = "/home/nick/projects/synbiobert/raw-data/ACS-100"
    subword_predictions_dir = "subtoken_label_probabilities"
    config_yml = 'convert_to_brat_config.yml'

    # # LOAD DATA AND MODELS
    # config, head_configs, head_dir = load_models(config_yml)
    # documents = gather_documents(document_dir)  # full paths to documents
    # document_ids, subword_predictions_paths = gather_predictions(subword_predictions_dir)
    #
    # # BRAT CONVERSION
    # for document_id, subword_predictions_path in zip(document_ids, subword_predictions_paths):
    #     if not document_id == "sb5b00025":
    #         continue
    #     document_path = filter_document_by_id(documents, document_id)
    #     output_ann_path = os.path.join("results/data", f"{document_id}")
    #     inferner = InferNERUtils(head_dir, head_configs, model_loading_device=config['device'])
    #     inferner.load_document(path_to_document=document_path)
    #     subword_predictions = inferner.parse_predictions(subword_predictions_path)
    #     predictions = inferner.join_subword_predictions(subword_predictions)
    #     brat_annotations = inferner.bert_to_brat(predictions)
    #
        sorted_annos = sorted(brat_annotations, key=lambda x: x.start)
        filtered_annos = []
        for i, anno in enumerate(sorted_annos):
            if i == 0 or i == len(sorted_annos)-1:
                continue

            next_anno = sorted_annos[i + 1]
            if anno.end >= next_anno.end:
                # get highest score
                filtered_annos.append(max((anno, next_anno), key=lambda x: x.score))
            else:
                filtered_annos.append(anno)






        # inferner.write_brat(brat_annotations, output_ann_path)
        inferner.write_brat(filtered_annos, "/home/nick/projects/brat-v1.3_Crunchy_Frog/data/acs-100")

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')