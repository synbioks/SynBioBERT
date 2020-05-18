from multi_tasking_transformers.evaluation.mlflow import MLFlowRun
# huner_bert_finetuning = MLFlowRun(tracking_uri='http://localhost:1730/',
#                                      experiment_name='task_finetuning',
#                                      run_id='a3e092285a34480ca1c1b2c114a7967d'
#                                      )

huner_bert_finetuning = MLFlowRun(tracking_uri='http://localhost:1730/',
                                     experiment_name='task_finetuning',
                                     run_id='38638a57137c42f1b6493c1f4d1addde'
                                     )

performances = huner_bert_finetuning.compare_against_run(huner_bert_finetuning, 'ner')
epoch_of_best_mtl_performance = {}
for dataset, task, metric, my_performance, other_performance in performances:
    epoch_of_best_mtl_performance[dataset] = my_performance.index(max(my_performance))
    print(f"{dataset}_{task}\t{epoch_of_best_mtl_performance[dataset]}")
print()
for dataset, task, metric, my_performance, other_performance in performances:
    print(f"{dataset}_{task}\t{max(my_performance)}\t{max(other_performance)}")
    # print(dataset, task, metric)

def get_filenames_best_epoch(performances):
    epoch_of_best_mtl_performance = {}
    for dataset, task, metric, my_performance, other_performance in performances:
        epoch_of_best_mtl_performance[dataset] = my_performance.index(max(my_performance))
        print(f"SubwordClassificationHead_{dataset}_checkpoint_{epoch_of_best_mtl_performance[dataset]}")
get_filenames_best_epoch(performances)
# print(huner_bert_finetuning)