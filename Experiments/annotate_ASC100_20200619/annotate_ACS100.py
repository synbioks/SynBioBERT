import os

import yaml
import re
from InferNER import InferNER
import time

if __name__ == '__main__':
    start = time.time()
    # Load config
    config = yaml.safe_load(open('config.yml'))

    # Parse paths. (must be a better way to do this).
    all_head_paths = sum(list(config['paths_to_heads'].values()), [])
    head_configs = [re.search("SubwordClassification.+json", filename) for path_to_head in all_head_paths for filename in os.listdir(path_to_head)]
    head_configs = [x.group() for x in head_configs if x]

    infer = InferNER(all_head_paths, head_configs,
                   device=config['device'])

    # foo.run_all_documents(path_to_document_dir=config['path_to_documents'],
    #                       output_directory='subtoken_label_probabilities',
    #                       recursive=True)

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')