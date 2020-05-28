#!/usr/bin/env bash

### Variome Species
cp ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/species/variome.conll.test \
  huner_data/species_"$(basename ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/species/variome.conll.test)"

perl huner2brat-eval.pl \
  huner_data/species_variome.conll.test \
  species

### Variome Gene
cp ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/gene/variome.conll.test \
  huner_data/gene_"$(basename ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/gene/variome.conll.test)"

perl huner2brat-eval.pl \
  huner_data/gene_variome.conll.test \
  gene

### Chebi Chemical
cp ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/chemical/chebi.conll.test \
  huner_data/chemical_"$(basename ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/chemical/chebi.conll.test)"

perl huner2brat-eval.pl \
  huner_data/chemical_chebi.conll.test \
  chemical

### Gellus Cellline
cp ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/cellline/gellus.conll.test \
  huner_data/cellline_"$(basename ~/nlp/projects/multi_tasking_transformers/experiments/biomedical_datasets/biomedical_datasets/huner/data/cellline/gellus.conll.test)"

perl huner2brat-eval.pl \
  huner_data/cellline_gellus.conll.test \
  cellline
