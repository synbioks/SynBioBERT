"""A program to convert the output of InferNER to BRAT standoff format"""
import numpy as np
from tqdm import tqdm
import yaml
import os
import pandas as pd

# from scipy.stats import trim_mean
# from numpy import median
# from pprint import pprint

REMOVE_WORDS = ['Figure', 'Table']


class BertToBrat:
    """Class for converting BERT output from InferNER to BRAT standoff format
    Note: method bert_to_brat is not used in the current pipeline (SBKS) """

    def __init__(self, document="", remove_words=None):
        self.document = document
        if remove_words is None:
            self.remove_words = remove_words
        else:
            if type(remove_words) == list:
                self.remove_words = remove_words
            elif remove_words:
                self.remove_words = REMOVE_WORDS

    def load_document(self, path_to_document):
        assert path_to_document.endswith('.txt')

        # LOAD RAW DATA AND SENTENCIZE
        with open(path_to_document, encoding='utf8') as f:
            self.document = f.read()  # does this work for large documents?
            self.document = self.document.replace('\n', ' ')
        return self.document

    def join_subword_predictions(self, subword_predictions):
        """ Join subword predictions into word-level predictions."""
        annotations = []
        for subword in subword_predictions:
            mapped_subword = self.document[subword.start: subword.end]
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

        if self.remove_words:
            for ann in annotations:
                if any([x_i in ann.token for x_i in self.remove_words]):
                    ann.label = 'O'
        return annotations

    def __bert_to_brat(self, annotations):
        """Not used"""
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
        return f"{self.token}"

    def __repr__(self):
        return f"{self.token}"


class BertAnnotation(Annotation):
    """A Class to store NER predictions from BERT."""

    def __init__(self,
                 token: str,
                 start: int,
                 end: int,
                 label: str,
                 score: float,
                 original_token=None,
                 bio=True):
        """

        Args:
            token (): A token as represented after WordPiece tokenization
            start (): span start
            end (): span end
            label (): predicted label
            score (): model prediction score. Softmax score is recommended
            original_token (): token as represented in the original document.
                Depending on how the document was tokenized,
                things like accents and special characeters may have normalized.
                This attribute stores the un-normalized form of the token.
            bio (): boolean value representing whether the labels are in IOB format.
        """
        super(BertAnnotation, self).__init__(token, start, end, label)
        self.score = score
        self.original_token = original_token
        if bio and self.label.startswith(('B-', 'I-')):
            self.entity_type = self.label[2:]


class BratAnnotation(Annotation):
    """A class to store annotations with BRAT standoff attributes, including
    span start/stop, and index (the "T" index in standoff format).
    """
    def __init__(self,
                 token: str,
                 start: int,
                 end: int,
                 label: str,
                 index=None):
        """
        index: Text-bound index. See BRAT standoff format.
        In practice, annotations are reindexed just before writing the *.ann file.
        But this class may be useful for keeping track of annotations during
        processing.
        """
        super(BratAnnotation, self).__init__(token, start, end, label)
        self.index = index


def view_subword_predictions(predictions, start=0, end=10):
    """For debugging
    """

    for i in range(start, end):
        print(predictions[i].token, "\t", predictions[i].label)


def join_adjacent_labels(predictions):
    """" Join adject labels at the word-level"""
    previous_annotation = predictions[0]
    # If the first subword is I change to B
    if previous_annotation.label.startswith('I-'):
        previous_annotation.label = 'B-' + previous_annotation.label[2:]
    for current_annotation in predictions[1:]:
        # Make label B if starting new annotation
        if current_annotation.label.startswith('I-'):
            if previous_annotation.label == 'O':
                current_annotation.label = 'B-' + current_annotation.label[2:]

        # Join annotation with previous if both are B or I
        if current_annotation.label.startswith(('B-', 'I-')):
            if previous_annotation.label.startswith(('B-', 'I-')):
                current_annotation.label = 'I-' + current_annotation.label[2:]

        previous_annotation = current_annotation


def collapse_labels(predictions):
    """ Flatten labels. Predictions must be BIO labels sorted by entity type """
    collapsed_labels = []
    current_annotation = {}

    for token in predictions:
        if token.label == 'O':
            if current_annotation:
                collapsed_labels.append(current_annotation)
            current_annotation = {'start': token.start,
                                  'end': token.end,
                                  'tokens': [token],
                                  'label': token.label,
                                  'scores': [token.score]}

        elif token.label.startswith('B-'):  # start new current_annotation
            current_annotation = {'start': token.start,
                                  'end': token.end,
                                  'tokens': [token],
                                  'label': token.label[2:],
                                  'scores': [token.score],
                                  }
            # collapsed_labels.append(current_annotation)

        elif token.label.startswith('I-'):
            current_annotation['end'] = token.end
            current_annotation['tokens'].append(token)
            current_annotation['scores'].append(token.score)

    if current_annotation:
        collapsed_labels.append(current_annotation)

    return collapsed_labels


def mean(numbers):
    return sum(numbers) / len(numbers)


def get_collapased_annotations(predictions_dir, path_to_documents, document_id,
                               remove_words=None):
    document_basename, file_predictions = get_predictions_file(document_id, predictions_dir)

    bert_to_brat = BertToBrat(remove_words=remove_words)
    doc = bert_to_brat.load_document(
        os.path.join(path_to_documents, document_basename)
    )
    subword_predictions = bert_to_brat.parse_predictions(file_predictions)
    predictions = bert_to_brat.join_subword_predictions(subword_predictions)

    join_adjacent_labels(predictions)

    # Debugging
    # print(view_subword_predictions(predictions, 200, 300))

    return collapse_labels(predictions), doc


def get_predictions_file(document_id, predictions_dir):
    # document_id = 'sb5b00016'
    document_basename = f'{document_id}.txt'
    file_predictions = f"{predictions_dir}/{document_id}_biobert_annotated.tsv"
    return document_basename, file_predictions


def write_text_document(doc, document_id, output_dir):
    output_doc_name = f'{document_id}.txt'
    with open(os.path.join(output_dir, output_doc_name), 'w') as f:
        f.write(doc)


def write_annotation_file(brat_ann_df, document_id, output_dir):
    output_ann_name = f'{document_id}.ann'
    brat_ann_df.to_csv(path_or_buf=os.path.join(output_dir, output_ann_name),
                       sep='\t', header=False, index=False)


def to_standoff_df(df, doc):
    anns = df[df['label'] != 'O']
    brat_lines = []
    t_id = 1
    for n, span in anns.iterrows():
        # Text-bound annotation ID
        t_idx = f'T{t_id}'
        t_id += 1

        # Label and Span-index
        label = span['label']
        start = span['start']
        end = span['end']

        # Token string extracted from the document
        token = doc[start: end]
        brat_lines.append([t_idx, f'{label} {start} {end}', token])
    brat_ann_df = pd.DataFrame(brat_lines)
    return brat_ann_df


def keep_max_scoring_spans(df):
    """ If the case of overlapping prediction spans, keep the span with the
    highest score
    Returns: A filtered dataframe containing only the highest scoring spans with
    no overlap.
    """
    df = df[df.groupby(['overlap_group']).avg_score.transform(max) == df.avg_score]
    return df


def assign_overlap_group(df, include_O_labels=True):
    """ Assign spans to groups that overlap. Overlapping groups can contain
    more than two overlapping spans.
    Args:
        df ():
        include_O_labels (): Whether to consider O labels (IOB-format) in resolving
        overlaps. If True, the scores for O labels (which tend to be quite high)
        are compared to all other scores in the overlap group. Turning this on
        will signigicantly reduce the number of remain annotations after overlap
        resolution.

    Returns: A dataframe with no overlapping annotation spans.

    """
    df_copy = df.copy()
    if include_O_labels:
        sorted_spans = df_copy.sort_values(['start', 'end'])
        df_copy = df_copy.assign(overlap_group=(
                sorted_spans.end - sorted_spans.start.shift(-1)).shift().lt(1).cumsum())  # lt = less than.
    else:
        df_copy['overlap_group'] = np.nan
        mentions = df_copy[df_copy.label != 'O']
        sorted_spans = mentions.sort_values(['start', 'end'])
        mentions = mentions.assign(overlap_group=(
                sorted_spans.end - sorted_spans.start.shift(-1)).shift().lt(1).cumsum())  # lt = less than.
        df_copy.loc[mentions.index] = mentions
    return df_copy


def remove_duplicate_O_labels(ann):
    """ During Inference, the documents may be have inference performed more than once,
    like for instance more than one model is used. This that case, the document
    will be repeated multiple times in the infernece program output. In order to
    focus on relevant annotations in downstream processing, duplicate O labels are
    removed.
    """
    for record in ann:
        record['string'] = " ".join([str(x) for x in record['tokens']])
    df = pd.DataFrame.from_records(ann)
    df = df.drop_duplicates(subset=['start', 'end', 'string', 'label'], keep='first')
    return df


def remove_single_char_mentions(ann):
    for record in ann:
        if record['label'] != 'O' and record['end'] - record['start'] == 1:
            record['label'] = 'O'
            record['tokens'][0].label = 'O'
    return ann


if __name__ == "__main__":
    # TODO make cli

    # I/O
    # document_id = 'sb5b00016'
    # test_document_basename = f'{document_id}.txt'
    with open('convert_to_brat_config.yml') as config_file:
        config = yaml.safe_load(config_file)

    path_to_documents = config['path_to_documents']
    predictions_dir = config['subword_predictions_dir']
    output_dir = config['output_ann_dir']

    # PREDICTIONS

    # DOCUMENTS
    doc_list = []
    doc_ids = []
    for filename in os.listdir(path_to_documents):
        doc_list.append(os.path.join(path_to_documents, filename))
        doc_ids.append(filename.replace('.txt', ''))

    # OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    # WRITE ANNOTATIONS

    # Degugging
    # doc_ids = ['sb300129j']

    for document_id in tqdm(doc_ids):
        try:
            ann, doc = get_collapased_annotations(predictions_dir, path_to_documents, document_id,
                                                  remove_words=['Figure', 'Table'])
            ann = sorted(ann, key=lambda x: x['start'])  # sorted annotation by start
            # remove single character entities
            ann = remove_single_char_mentions(ann)

            # Debugging. Test for single character annotations
            # pprint([record for record in ann if record['label'] != 'O' and record['end'] - record['start'] == 1])

            # remove duplicate O
            df = remove_duplicate_O_labels(ann)

            # average score column
            df['avg_score'] = df.scores.apply(mean)
            # df['median_score'] = df.scores.apply(median)
            # df['trim_mean'] = df.scores.apply(trim_mean, proportiontocut=0.4)

            # group overlapping spans
            df = assign_overlap_group(df, include_O_labels=False)

            # keep max scoring annotation within each overlap group
            df = keep_max_scoring_spans(df)

            # Collect annotations. Write to Brat standoff format
            brat_ann_df = to_standoff_df(df, doc)

            # Write Document and Annotation Files
            # write ann
            write_annotation_file(brat_ann_df, document_id, output_dir)
            # write doc
            write_text_document(doc, document_id, output_dir)

        except FileNotFoundError:
            print(f'could not parse predictions for {document_id}')
            continue
