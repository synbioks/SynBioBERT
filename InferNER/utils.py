from .infer import InferNER

class InferSingleExample(InferNER):
    def __init__(self,):
        super(InferSingleExample, self).__init__()

    def run_single_example(self, text):
        # head_name = os.path.basename(self.head_directory)
        # nlp = TransformersLanguage(trf_name=head_name, meta={"lang": "en"})
        # nlp.add_pipe(nlp.create_pipe("sentencizer"))
        sentencized_document = self.sentencizer(text)
        # self.sentence = str(list(sentencized_document.sents)[0]) # TODO break off into a separate method to use one sentence
        # self.sentence = "The Ca2+ ionophore , A23187 or ionomycin , mimicked the effect of AVP , whereas the protein kinase C ( PKC ) activator , TPA , only induced a slight increase in AA release"
        # self.sentence = r"Activating mutations in BRAF have been reported in 5–15 % of colorectal carcinomas ( CRC ) , with by far the most common mutation being a 1796T to A transversion leading to a V600E substitution [1-3] .  The BRAF V600E hotspot mutation is strongly associated with the microsatellite instability ( MSI+ ) phenotype but is mutually exclusive with KRAS mutations [4-7] ."
        self.sentence_encoding = self.tokenizer.encode(self.sentence)

        # PREPARE MODEL INPUT
        input_ids = torch.tensor([self.sentence_encoding.ids], dtype=torch.long)
        attention_mask = torch.tensor([self.sentence_encoding.attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([self.sentence_encoding.type_ids], dtype=torch.long)
        self.document = sentencized_document
        self.tokens = self.sentence_encoding.tokens
        self.spans = self.sentence_encoding.offsets
        self.input_ids = input_ids

        # RUN EXAMPLE THROUGH BERT
        self.bert.eval()
        self.bert.to(device=self.device)
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

            subwords_idx = [index_of_subword for index_of_subword, mask in enumerate(token_mask) if mask == 0]

            # Print tokenized sentence
            self.output_tokens = [self.sentence_encoding.tokens[i] for i in subwords_idx]
            # Print subword spans
            self.output_spans = [str(self.sentence_encoding.offsets[i]) for i in subwords_idx]
            # Print labels
            self.output_labels = [self.predicted_labels[i] for i in subwords_idx]
            self.output_table = pd.DataFrame.from_dict(
                {'tokens': self.output_tokens, 'labels': self.output_labels, 'spans': self.output_spans})