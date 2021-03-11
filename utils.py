import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import random, randrange, betavariate
from numpy.random import uniform, choice
from cascade_ucb import CascadeUCB
from tqdm import tqdm, trange
import math


def generate_data(rounds, weights):
    simulated_data = [np.random.binomial(1, weights) for _ in range(rounds)]
    return simulated_data


def run_experiment(list_L, list_delta, list_K, number_of_rounds, n_runs, p):
    n_regret = np.zeros((len(list_L), n_runs))
    for i, (L, K, delta) in enumerate(zip(list_L, list_K, list_delta)):
        pbar = tqdm(range(n_runs), desc='description')
        pbar.set_description("L= {} , K= {},delta ={}".format(L,K,delta))
        for run in pbar:  # trange(n_runs,) :
            weights = [p for i in range(K)] + [np.abs(p-delta)
                                               for i in range(L-K)]
            cascade_model = CascadeUCB(number_of_rounds, L, K)
            dataset = generate_data(number_of_rounds, weights)
            cascade_model.initialize(dataset, weights)
            for t in range(1, number_of_rounds):
                cascade_model.one_round(t, dataset)
            n_regret[i, run] = np.cumsum(cascade_model.regrets)[-1]
    res = pd.DataFrame({'L': list_L,
                        'K': list_K,
                        'delta': list_delta,
                        'mean': n_regret.mean(axis=-1),
                        'std': n_regret.std(axis=-1)})
    return res, n_regret
