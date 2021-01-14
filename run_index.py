from pipeline import Pipeline, get_config
from model.word_vector import VectorNames
from model.dataset import dataset_path_dict, sts_tasks

def index(cfg):
    p = Pipeline(cfg)
    p.preprocess()


if __name__ == "__main__":
    for wv in VectorNames:
        print(wv)
        for dataset in dataset_path_dict:
            print('\t', dataset)
            cfg = get_config('config/default.yaml')
            cfg['word_vector']['word_vector'] = wv
            if dataset.split(':')[0] == 'sts':
                year = dataset.split(':')[1]
                for task in sts_tasks[year]:
                    print('\t\t', task)
                    dataset_name = dataset + ":" + task
                    cfg['dataset']['dataset_name'] = dataset_name
                    index(cfg)
            else:
                cfg['dataset']['dataset_name'] = dataset
                index(cfg)
