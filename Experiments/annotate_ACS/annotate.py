from InferNER import InferNER

class ACSInferNER(InferNER):
    def __init__(self, head_directories, device):
        super(ACSInferNER).__init__(head_directories,
                                    head_configs,
                                    device,
                                    lowercase=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dry-run", action='store_true', help='dry run')  # by default it stores false
    # parser.add_argument("--force-run", action='store_true',
    #                     help='Causes documents with existing output files to be overwritten.')
    # args = parser.parse_args()

    start = time.time()
    # PATH_TO_VOCAB = 'models/vocab.txt'
    # data_dir = 'raw-data'
    # PATH_TO_MODEL = "../models"
    # PATH_TO_BASE_MODEL = r'C:/Users/User/nlp/projects/synbiobert/models/biobert_v1.1_pubmed'
    # PATH_TO_FILE = r'raw-data/ACS-100/sb6/sb6b00371.txt'
    # with open(PATH_TO_FILE, encoding='utf8') as f:
    #     document_as_string = f.read()  # does this work for large documents?

    # foo = InferNER(r"/home/rodriguezne2/results/multitasking_transformers/bert/run_2020_03_22_01_52_40_pine.cs.vcu.edu/SubwordClassificationHead_variome_species_checkpoint_10",
    # "SubwordClassificationHead_variome_species.json", device='cpu')
    config = yaml.safe_load(open('config.yml'))
    print(config)
    # config = yaml.safe_load(open('../Experiments/annotate_ACS100_20200410_0726/config.yml'))
    all_head_paths = sum(list(config['paths_to_heads'].values()), [])
    head_configs = [re.search("SubwordClassification.+json", filename) for path_to_head in all_head_paths for filename
                    in os.listdir(path_to_head)]
    head_configs = [x.group() for x in head_configs if x]

    foo = InferNER(all_head_paths, head_configs, device=config['device'])
    foo.run_all_documents(path_to_document_dir=config['path_to_documents'], output_directory=config['experiment_name'])

    ### RUN SINGLE SENTENCE ###
    # foo.run_single_example(document_as_string)

    ### RUN SINGLE DOCUMENT ###
    # foo.run_document(PATH_TO_FILE)

    ### RUN DOCUMENTS IN DIRECTORY
    # for i in range(3, 10):
    #     print(f'working on ACS-100/sb{i}')
    #     foo.run_all_documents(path_to_document_dir=f'../raw-data/ACS-100/sb{i}',
    #                           output_directory='huner_biobert_annotated')

    end = time.time()
    print(f'Finished in {end - start:0.2f} seconds')
