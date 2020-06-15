class Annotation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.document_string = ''
        self.annotations = []  # Standoff format https://brat.nlplab.org/standoff.html
        self._conll_delimiter = None
        self._token_field = None
        self._label_field = None

    def from_conll(self, delimiter=' ', token_field=0, label_field=2):
        self._conll_delimiter = delimiter
        self._token_field = token_field
        self._label_field = label_field
        return self

    def conll2brat(self):
        with open(self.file_path) as merged_file:
            T_index = 1
            position_counter = 0
            document_string = ''
            annotations = []

            for line in merged_file:
                # line = line.strip()
                if line.startswith("-DOCSTART- X X O"):
                    continue
                if line == "\n":
                    if not document_string:
                        continue
                    position_counter += len(line)
                    document_string += line

                else:
                    token = line.split(self._conll_delimiter)[self._token_field]
                    label = line.split(self._conll_delimiter)[self._label_field]
                    token_span = (position_counter + 1, position_counter+len(token))
                    position_counter += len(token) + 1
                    document_string += ' ' + token

                    ## Annotation
                    if label.startswith("B-"):
                        start_span = token_span[0]
                        end_span = token_span[1]
                        label = label.replace('B-', '')
                        annotations.append([f'T{T_index}', label, start_span, end_span, token])
                        T_index += 1
                    elif label.startswith('I-'):
                        end_span = token_span[1]
                        anno = annotations.pop()
                        anno[4] += f' {token}'
                        anno[3] =  end_span
                        annotations.append(anno)

            self.document_string = document_string
            self.annotations = annotations
            return self

    def __getitem__(self, item):
        ''' implicitly asserts alignment is correct by slicing with the
        entity span
        '''
        return self.annotations[item]

