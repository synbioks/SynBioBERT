import os


class BertToBrat:
    def __init__(self, document=""):
        self.document = document

    def load_document(self, path_to_document):
        assert path_to_document.endswith('.txt')

        # LOAD RAW DATA AND SENTENCIZE
        with open(path_to_document, encoding='utf8') as f:
            self.document = f.read()  # does this work for large documents?
            self.document = self.document.replace('\n', ' ')
        return self.document

    def join_subword_predictions(self, subword_predictions):
        annotations = []
        for subword in subword_predictions:
            mapped_subword = self.document[subword.start: subword.end]
            # test_subword_mapping(mapped_subword, subword.token)  # special characters break this test
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
                anno.token = self.document[anno.start: subword.end]
        return annotations

    def bert_to_brat(self, annotations):
        brat_annotations = []
        T_index = 1
        previous = Annotation('', 0, 0, 'O')

        for i, current in enumerate(annotations):
            if i == 0 and not current.label == 'O':
                self.create_new_annotation(T_index, brat_annotations, current)
                T_index += 1
                continue

            ## Check for the possible start of an new annotation
            if current.label.startswith('B-') or (
                    current.label.startswith('I-') and previous.label in ['O', 'Bert_Token']):
                ## Join with previous if there's no space in between the previous annotation
                if not T_index:
                    self.join_with_previous(brat_annotations, current)
                else:
                    self.create_new_annotation(T_index, brat_annotations, current)
                    T_index += 1

            ## Continuation of the annotation
            elif current.label.startswith('I-'):
                self.join_with_previous(brat_annotations, current)

            previous = current
        return brat_annotations

    def create_new_annotation(self, T_index, brat_annotations, current):
        current.label = current.label.replace('B-', '').replace('I-', '')
        current.index = T_index
        brat_annotations.append(current)

    def join_with_previous(self, brat_annotations, current):
        anno = brat_annotations[-1]
        anno.token = self.document[anno.start: current.end]
        anno.end = current.end
        previous = current

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
        """
        Write to brat formated file. makes new index if needed.
        :param brat_annotations: a list of annotations containing brat required fields.
        :param output_path:
        """
        outfile_ann_path = f'{output_path}.ann'
        with open(outfile_ann_path, 'w') as outfile_ann:
            for i, anno in enumerate(brat_annotations):
                anno.index = i + 1
                outfile_ann.write(f'T{anno.index}\t{anno.label} {anno.start} {anno.end}\t{anno.token}\n')

        outfile_text_path = f'{output_path}.txt'
        with open(outfile_text_path, 'w') as outfile_text:
            outfile_text.write(self.document)


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


def update_annotation_view(annotation_view, annotation):
    annotation_view['ents'].append({'start': annotation.start,
                                    'end': annotation.end,
                                    'label': annotation.label})


def sort_annotations_by_start(annotations):
    return sorted(annotations, key=lambda x: x.start)


def sort_annotations_by_end(annotations):
    return sorted(annotations, key=lambda x: x.end)


if __name__ == '__main__':
    '''
    '''
    import time
    import yaml

    start = time.time()

    ## INPUTS
    config_yml = 'convert_to_brat_config.yml'

    with open(config_yml) as yml:
        config = yaml.safe_load(yml)
    document_dir = config['path_to_documents']
    subword_predictions_dir = config['subword_predictions_dir']

    ## OUTPUTS
    # output_ann_dir = "/home/nick/projects/brat-v1.3_Crunchy_Frog/data/acs-100"  # DEV
    output_ann_dir = config['output_ann_dir']  # Exp

    # LOAD DATA AND MODELS
    documents = gather_documents(document_dir)  # full paths to documents
    document_ids, subword_predictions_paths = gather_predictions(subword_predictions_dir)

    # BRAT CONVERSION
    for document_id, subword_predictions_path in zip(document_ids, subword_predictions_paths):
        print(document_id)
        print(f'working on {document_id}')
        document_path = filter_document_by_id(documents, document_id)
        output_ann_path = os.path.join(output_ann_dir, f"{document_id}")
        bert_to_brat = BertToBrat()
        bert_to_brat.load_document(path_to_document=document_path)
        subword_predictions = bert_to_brat.parse_predictions(subword_predictions_path)
        predictions = bert_to_brat.join_subword_predictions(subword_predictions)
        brat_annotations = bert_to_brat.bert_to_brat(predictions)
        filtered_annos = {'ents': [], 'text': bert_to_brat.document}  # note: ents has to be sorted

        # Join consecutive annotations of same label
        i = len(brat_annotations) - 1
        while i > -1:
            current = brat_annotations[i]
            if i > 0:
                previous = brat_annotations[i - 1]
                if current.start - previous.end <= 1 and \
                        current.label == previous.label:
                    previous.end = current.end
                    previous.token = bert_to_brat.document[previous.start: previous.end]
                    # # Average their scores
                    # previous.score = sum((current.score, previous.score)) / 2
                    # Keep best score
                    previous.score = max(current.score, previous.score)
                    brat_annotations.pop(i)
                else:
                    update_annotation_view(filtered_annos, current)
            else:
                update_annotation_view(filtered_annos, current)
            i -= 1

        # Resolve overlapping
        sorted_annos = sort_annotations_by_start(brat_annotations)
        sorted_annos = [anno for anno in sorted_annos if anno.token.strip() != '.']
        filtered_annos = {'ents': [], 'text': bert_to_brat.document}  # note: ents has to be sorted
        i = len(sorted_annos) - 1
        while i > -1:
            current = sorted_annos[i]
            if i > 0:
                previous = sorted_annos[i - 1]
                if previous.end > current.start:
                    longest_spaning = max(previous, current, key=lambda x: x.end - x.start)
                    sorted_annos[i - 1] = longest_spaning
                    sorted_annos.pop(i)
                elif (previous.start == current.start) and (previous.end == current.end):
                    best_scoring = max(previous, current, key=lambda x: x.score)
                    sorted_annos[i - 1] = best_scoring
                    sorted_annos.pop(i)

                else:
                    update_annotation_view(filtered_annos, current)
            else:
                update_annotation_view(filtered_annos, current)

            i -= 1

        bert_to_brat.write_brat(sorted_annos, output_path=output_ann_path)

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')
