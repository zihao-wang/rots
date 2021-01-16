"""
Function for logging, and other auxilury functions
"""

import hashlib
import json
import os
import pickle
import time
import yaml
from os.path import join

import numpy as np
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

class Writer:

    _log_path = join(dir_path, 'log')

    def __init__(self, case_name, meta, log_path=None):
        if isinstance(meta, dict):
            self.meta = meta
        else:
            self.meta = vars(meta)
        self.time = time.time()
        self.meta['time'] = self.time
        self.idstr = case_name
        self.column_name = {}
        self.idstr += time.strftime("%y%m%d.%H:%M:%S", time.localtime()) + \
                 hashlib.sha1(str(self.meta).encode('UTF-8')).hexdigest()[:8]

        self.log_path = log_path if log_path else self._log_path
        os.makedirs(self.case_dir, exist_ok=False)

        with open(self.metaf, 'wt') as f:
            json.dump(self.meta, f)

    def append_trace(self, trace_name, data):
        if trace_name not in self.column_name:
            self.column_name[trace_name] = list(data.keys())
            assert len(self.column_name[trace_name]) > 0
        if not os.path.exists(self.tracef(trace_name)):
            with open(self.tracef(trace_name), 'at') as f:
                f.write(','.join(self.column_name[trace_name]) + '\n')
        with open(self.tracef(trace_name), 'at') as f:
            f.write(','.join([str(data[c]) for c in self.column_name[trace_name]]) + '\n')

    def save_pickle(self, obj, name):
        with open(join(self.case_dir, name), 'wb') as f:
            pickle.dump(obj, f)

    def save_array(self, arr, name):
        np.save(join(self.case_dir, name), arr)

    def save_json(self, obj, name):
        if not name.endswith('json'):
            name += '.json'
        with open(join(self.case_dir, name), 'wt') as f:
            json.dump(obj, f)

    def save_model(self, model: torch.nn.Module, e):
        print("saving model : ", e)
        device = model.device
        torch.save(model.cpu().state_dict(), self.modelf(e))
        model.to(device)

    def save_plot(self, fig, name):
        fig.savefig(join(self.case_dir, name))

    @property
    def case_dir(self):
        return join(self.log_path, self.idstr)

    @property
    def metaf(self):
        return join(self.case_dir, 'meta.json')

    def tracef(self, name):
        return join(self.case_dir, '{}.csv'.format(name))

    def modelf(self, e):
        return join(self.case_dir, '{}.ckpt'.format(e))


def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd, Loader=yaml.FullLoader)
    return data
