import os

import yaml
import re
from InferNER import InferNER
import time

if __name__ == '__main__':
    start = time.time()
    config = yaml.safe_load(open('config.yml'))
    # config = yaml.safe_load(open('../Experiments/annotate_ACS100_20200410_0726/config.yml'))
    all_head_paths = sum(list(config['paths_to_heads'].values()), [])
    head_configs = [re.search("SubwordClassification.+json", filename) for path_to_head in all_head_paths for filename in os.listdir(path_to_head)]
    head_configs = [x.group() for x in head_configs if x]

    foo = InferNER(all_head_paths, head_configs,
                   device=config['device'])
    foo.run_all_documents(path_to_document_dir=config['path_to_documents'],
                          output_directory='subtoken_label_probabilities',
                          recursive=True)

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')