# Import necessary libraries
from typing import Optional, Union, Tuple

from cmapPy.pandasGEXpress.parse import parse
import numpy as np
import json
from collections import Counter

from numpy.core.multiarray import ndarray


def get_target_labels(working_data, mydict):
    print("Creating target labels")
    target = []
    cnt = 0
    for i in working_data.columns:
        for pert, collist in mydict.items():
            if i in collist:
                if (cnt % 1000 == 0):
                    print(cnt, end=' ')
                target.append(pert)
                cnt += 1
    return target


# Create 2 dictionaries
# pert2profile: perturbagen: number of profiles for that particular perturbagen
# location_pert: perturbagen: location of 1st profile of perturbagen

def create_pert2profile(data):
    print("Creating pert2profile")
    return Counter(data.target)


def create_location_pert(data):
    print("creating location_pert")
    location_pert = dict()
    cnt = 0
    for i in set(np.unique(data.target.values)):
        loc = np.where(data['target'] == i)[0][0]
        location_pert[i] = loc
        if (cnt % 100 == 0):
            print(cnt, end=' ')
        cnt += 1
    return location_pert


def gctx2pd(gctxfile, jsonfile):
    obj = parse(gctxfile)
    data = obj.data_df
    with open(jsonfile, 'r') as fp:
        metadata = json.load(fp)
    return data, metadata


# Create pairs and targets of size 'batch_size'
def get_training_data(data, pert2profiles, location_pert, batch_size):
    rng = np.random

    list_of_perturbagens = np.unique(data.target)
    print(len(list_of_perturbagens))
    dim = 978

    batch_perturbagens = rng.choice(list_of_perturbagens, size=(len(list_of_perturbagens) / 5,), replace=False)

    pairs = [np.zeros((batch_size, dim)) for i in range(2)]

    targets = np.zeros((batch_size,))
    targets[batch_size // 2:] = 1

    for i in range(batch_size):
        pert1 = batch_perturbagens[i]
        idx_1 = rng.randint(0, pert2profiles[pert1])
        pairs[0][i, :] = data.iloc[location_pert[pert1] + idx_1, 0:978]

        pert2 = pert1
        if i < batch_size // 2:
            pert2 = rng.choice(batch_perturbagens)
        idx_2 = rng.randint(0, pert2profiles[pert2])
        pairs[1][i, :] = data.iloc[location_pert[pert2] + idx_2, 0:978]

    return np.asarray(pairs), np.asarray(targets)


def train_and_test_perturbagens(list_of_perturbagens):
    train = rng.choice(list_of_perturbagens, size=(len(list_of_perturbagens) * 4 // 5,), replace=False)

    def Diff(li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif

    test = Diff(list_of_perturbagens, train)
    return train, test


def generate_data(data, test_pert, batch_size, dim=978):
    pairs, targets = list()
    for pert in test_pert:
        pairs.append()
