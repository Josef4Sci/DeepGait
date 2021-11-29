import logging
import os
import shutil
import json
from tensorboardX import SummaryWriter

import pickle
import torch
import numpy as np
import socket
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt


class TensorboardLogger(object):

    def __init__(self, log_dir, mess=None):
        self.log_dir = self._prepare_log_dir(log_dir, mess)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        self.epoch = None

    def log_value(self, name, value, step=None):
        if not step:
            step = self.global_step
        self.writer.add_scalar(name, value, step)

        return self

    def add_figure(self, tag, figure, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step=self.global_step, close=close, walltime=walltime)
        return self

    def log_embedding(self, features, labels, labels_header=None, images=None, step=None, name='default'):
        if not step:
            step = self.global_step

        if images is not None:
            images = torch.Tensor(images)

            for k, img in enumerate(images):
                img = (img - img.min()) / (img.max()-img.min())
                images[k] = img

        self.writer.add_embedding(torch.Tensor(features), labels, images, step, tag=name, metadata_header=labels_header)
        return self

    def step(self, step=1):
        self.global_step += step

    def log_options(self, options, changes=None):
        if type(options) != dict:
            options = options.__dict__

        options['hash'] = json_hash(options)

        with open(os.path.join(self.log_dir, 'options.json'), 'w') as fp:
            json.dump(options, fp)

        if changes:
            with open(os.path.join(self.log_dir, 'changes.json'), 'w') as fp:
                json.dump(changes, fp)

    def log_dict(self, dict_to_log, prefix=None, suffix=None, stdout=True, step=None):
        for k, v in dict_to_log.items():
            name = '-'.join(filter(None, [prefix, k, suffix]))
            self.log_value(name, v, step)
            if stdout:
                logging.info('{} {:5f}'.format(name, v))

    def log_txt(self, name_to_log, val_to_log):
        logging.info('{} {}'.format(name_to_log, val_to_log))

    def add_pr_curve_from_dict_list(self, dict_list, step=None, name='ROC'):
        if not step and self.epoch:
            suffix = self.epoch
        elif not step:
            suffix = self.global_step
        else:
            suffix = step

        with open(os.path.join(self.log_dir, 'pr-curve-{}.pkl'.format(suffix)), 'wb') as fp:
            pickle.dump(dict_list, fp)

        true_positive_counts = [d['true_positives'] for d in dict_list]
        false_positive_counts = [d['false_positives'] for d in dict_list]
        true_negative_counts = [d['true_negatives'] for d in dict_list]
        false_negative_counts = [d['false_negatives'] for d in dict_list]
        precision = [d['precision'] for d in dict_list]
        recall = [d['recall'] for d in dict_list]
        thresh = [d['threshold'] for d in dict_list]


        # fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(18.5, 10.5)
        # fig.savefig(os.path.join(self.log_dir, 'pr-curve-{}.png'.format(suffix)), dpi=100)
        # plt.clf()
        # plt.close(fig)


        recall = np.array(recall)
        recall, uniq_idx = np.unique(recall, return_index=True)
        true_positive_counts = np.array(true_positive_counts)[uniq_idx]
        false_positive_counts = np.array(false_positive_counts)[uniq_idx]
        true_negative_counts = np.array(true_negative_counts)[uniq_idx]
        false_negative_counts = np.array(false_negative_counts)[uniq_idx]
        precision = np.array(precision)[uniq_idx]

        idxs = np.argsort(recall)[::-1]
        true_positive_counts = true_positive_counts[idxs].tolist()
        false_positive_counts = false_positive_counts[idxs].tolist()
        true_negative_counts = true_negative_counts[idxs].tolist()
        false_negative_counts = false_negative_counts[idxs].tolist()
        precision = precision[idxs].tolist()
        recall = recall[idxs].tolist()

        self.add_pr_curve_raw(true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                              precision, recall, step, name)

    def add_pr_curve_raw(self, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                         precision, recall, step=None, name='ROC'):
        if not step:
            step = self.global_step

        num_thresholds = len(true_positive_counts)
        self.writer.add_pr_curve_raw(name, true_positive_counts, false_positive_counts, true_negative_counts,
                                     false_negative_counts, precision, recall, step, num_thresholds,
                                     weights=None)

    def add_pr_curve(self, tag, labels, predictions):
        self.writer.add_pr_curve(tag, labels, predictions, global_step=self.global_step)

    @staticmethod
    def _prepare_log_dir(log_path, mess=None):
        import datetime
        now = datetime.datetime.now()
        log_path = log_path + '-%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        if mess:
            log_path = log_path + '-'+mess

        if os.path.isdir(log_path):
            log_path += '-%02d' % now.second

        os.mkdir(os.path.expanduser(log_path))

        logpath = os.path.join(log_path, 'log.txt')
        print(logpath)
        logging.basicConfig(filename=logpath, filemode='w', format='%(name)s - %(levelname)s - %(message)s'
                            , level=logging.INFO)
        # file_handler = logging.FileHandler(os.path.join(log_path, 'log.txt'), mode='w')
        # file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s '))
        # logging.info("writing log file to: %s ", os.path.join(log_path, 'log.txt'))
        # logging.getLogger().addHandler(file_handler)

        import sys
        shutil.copy(os.path.abspath(sys.modules['__main__'].__file__), log_path)
        return log_path


def json_hash(d):
    from hashlib import sha1
    assert d is not None, "Cannot hash None!"

    def hashnumpy(a):
        if type(a) == dict:
            for k, v in a.items():
                a[k] = hashnumpy(v)

        if type(a) == list:
            for i, v in enumerate(a):
                a[i] = hashnumpy(v)

        if type(a) == np.ndarray:
            return sha1(a).hexdigest()

        if hasattr(a, '__dict__'):
            return hashnumpy(a.__dict__)

        return a

    return sha1(json.dumps(hashnumpy(d), sort_keys=True).encode()).hexdigest()


def getMonteCarloNextFolder(logpath2):
    startIndex = 0
    for x in [f.path for f in os.scandir(logpath2) if f.is_dir()]:
        if ('MonteCarlo' in x):
            currentIndex = int(x.split('\'')[1].split('MonteCarlo-')[1].split('-')[0])
            if currentIndex > startIndex:
                startIndex = currentIndex
    return '\'MonteCarlo-' + str(startIndex + 1) + '\''


class GPU:
    device = torch.device('cpu')

    @staticmethod
    def get_free_gpu(memory=1000):
        skinner_map = {0: 2, 1: 0, 2: 1, 3: 3}
        a = os.popen("/usr/bin/nvidia-smi | grep 'MiB /' | awk -e '{print $9}' | sed -e 's/MiB//'")

        free_memory = []
        while 1:
            line = a.readline()
            if not line:
                break
            free_memory.append(int(line))

        gpu = np.argmin(free_memory)
        if free_memory[gpu] < memory:
            if socket.gethostname() == "skinner":
                for k, v in skinner_map.items():
                    if v == gpu:
                        return k
            return gpu

        logging.error('No free GPU available.')
        exit(1)

    @classmethod
    def set(cls, gpuid, memory=1000):
        gpuid = int(gpuid)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logging.info("searching for free GPU")
            if gpuid == -1:
                gpuid = GPU.get_free_gpu(memory)
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)
            if torch.cuda.device_count() == 1:  # sometimes this does not work
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(gpuid))
        else:
            gpuid = os.environ['CUDA_VISIBLE_DEVICES']
            logging.info('taking GPU {} as specified in envorionment variable'.format(gpuid))
            torch.cuda.set_device(0)

        cls.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

        logging.info('Using GPU {}'.format(gpuid))
        return gpuid