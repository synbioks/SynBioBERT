from multi_tasking_transformers.evaluation.mlflow import MLFlowRun
import gin


def get_filenames_best_epoch(performances):
    epoch_of_best_mtl_performance = {}
    filenames_best_epoch = {}
    for dataset, task, metric, my_performance, other_performance in performances:
        epoch_of_best_mtl_performance[dataset] = my_performance.index(max(my_performance))
        filenames_best_epoch[dataset] = f"SubwordClassificationHead_{dataset}_checkpoint_{epoch_of_best_mtl_performance[dataset]}"
    return filenames_best_epoch


@gin.configurable
def get_best_epoch(ml_flow_directory,
                   run_directory,
                   experiment_name):

    huner_bert_finetuning = MLFlowRun(tracking_uri=ml_flow_directory,
                                      experiment_name=experiment_name,
                                      run_id=run_directory
                                      )
    performances = huner_bert_finetuning.compare_against_run(huner_bert_finetuning, 'ner')
    epoch_of_best_mtl_performance = {}
    for dataset, task, metric, my_performance, other_performance in performances:
        epoch_of_best_mtl_performance[dataset] = my_performance.index(max(my_performance))
        print(f"{dataset}_{task}\t{epoch_of_best_mtl_performance[dataset]}")
    print()
    for dataset, task, metric, my_performance, other_performance in performances:
        print(f"{dataset}_{task}\t{max(my_performance)}\t{max(other_performance)}")
    return get_filenames_best_epoch(performances)



if __name__ == '__main__':
    get_best_epoch()