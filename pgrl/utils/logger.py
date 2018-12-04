import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os
import csv

class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0

    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path):
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data

    def make_train_plots(self,
                         log_path=None,
                         keys = None,
                         save_loc = None):
        if log_path is None and not bool(self.log):
            print("Need to provide either the log or path to a log file")

        if bool(self.log) is False:
            self.read_log(log_path)

        for key in keys:
            if key in self.log.keys():
                plt.figure(figsize=(10,6))
                plt.plot(self.log[key])
                plt.title(key)
                plt.savefig(save_loc+'/'+key+'.png', dpi=300)
                plt.close()

