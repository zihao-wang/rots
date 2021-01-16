from pipeline import Pipeline, get_config
from model.word_vector import VectorNames
from model.dataset import dataset_path_dict, sts_tasks
from utils import Writer
from pprint import pprint


model = {
    # 'SIF': "config/SIF.yaml",
    # 'uSIF': "config/uSIF.yaml",
    # 'SIF+adjustwrd': "config/SIF+adjustwrd.yaml",
    # 'uSIF+adjustwrd': "config/uSIF+adjustwrd.yaml"
    'WRD+SUP': "config/WRD+SUP.yaml",
    'WRD+SWC': "config/WRD+SWC.yaml",
    'WRD+WR': "config/WRD+WR.yaml",
    # 'WRD+SUP+adjustcos': "config/WRD+SUP+adjustcos.yaml",
    # 'WRD+SWC+adjustcos': "config/WRD+SWC+adjustcos.yaml",
    # 'WRD+WR+adjustcos': "config/WRD+WR+adjustcos.yaml",
    # 'ROTS': "config/ROTS.yaml",
}

def run(cfg):
    p = Pipeline(cfg)
    p.preprocess()
    return p.run()


def dump(m, wv, dataset, ans, writer:Writer):
    record = {
        'word vector': wv,
        'dataset': dataset
    }

    print(ans)
    if isinstance(ans, tuple):
        l, s, r = ans
        record['model'] = m
        record['left'] = l
        record['right'] = r
        record['score'] = s
        pprint(record)
        writer.append_trace(trace_name='major_comparison', data=record)

    if isinstance(ans, dict):
        for k, (l, s, r) in ans.items():
            record['model'] = m + k
            record['left'] = l
            record['right'] = r
            record['score'] = s
            pprint(record)
            writer.append_trace(trace_name='major_comparison', data=record)


if __name__ == "__main__":

    for m in model:
        config_file = model[m]
        writer = Writer(case_name="major_compare" + m, meta={})
        for wv in VectorNames:
            print(wv)
            for dataset in dataset_path_dict:
                print('\t', dataset)
                cfg = get_config(config_file)
                cfg['word_vector']['word_vector'] = wv
                if dataset.split(':')[0] == 'sts':
                    year = dataset.split(':')[1]
                    for task in sts_tasks[year]:
                        print('\t\t', task)
                        dataset_name = dataset + ":" + task
                        cfg['dataset']['dataset_name'] = dataset_name
                        ans = run(cfg)
                        dump(m, wv, dataset_name, ans, writer)
                else:
                    cfg['dataset']['dataset_name'] = dataset
                    ans = run(cfg)
                    dump(m, wv, dataset, ans, writer)
