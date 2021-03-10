import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import random, randrange, betavariate
from numpy.random import uniform, choice
import math


def generate_data(rounds, weights):
    simulated_data = [np.random.binomial(1, weights) for _ in range(rounds)]
    return simulated_data


class CascadeUCB():
    def __init__(self, number_of_rounds, L, K):
        super().__init__()
        self.number_of_rounds = number_of_rounds
        self.L = L
        self.K = K
        self.T = np.zeros((number_of_rounds, L))
        self.U = np.zeros((number_of_rounds, L))
        self.w = np.zeros((number_of_rounds, L))
        self.A = np.zeros((number_of_rounds, K), dtype=np.int32)
        self.C = np.zeros(number_of_rounds)
        self.regrets = np.zeros(number_of_rounds)
        return

    def initialize(self, dataset, weights):
        self.T[0, :] = 1
        for t in range(self.L):
            d = np.random.permutation(self.L)
            At = np.append([t], d[d != t][:self.K-1])
            reward = dataset[t][At]
            self.w[0, t] = reward[0]
        # Best reward
        r = 1
        for k in range(self.K):
            r = r*(1-weights[k])
        self.best_f = 1-r

    def f(self, t):
        # best permutation reward
        r = 1
        for k in range(self.K):
            r = r*(1-self.w[t-1, self.A[t, k]])
        reward = 1-r
        return reward

    def update_ucb_item(self, e, t):
        c = np.sqrt((1.5*np.log(t))/self.T[t-1, e])
        return self.w[t-1, e] + c

    def update_weights(self, t):
        self.T[t] = self.T[t-1]
        self.w[t] = self.w[t-1]
        for k in range(min(self.K, int(self.C[t])+1)):
            if k < int(self.C[t]):
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]])/self.T[t, self.A[t, k]]

            else:
                self.T[t, self.A[t, k]] += 1
                self.w[t, self.A[t, k]] = (
                    self.T[t-1, self.A[t, k]]*self.w[t-1, self.A[t, k]]+1)/self.T[t, self.A[t, k]]

    def one_round(self, t, dataset):
        self.U[t] = [self.update_ucb_item(e, t)
                     for e in range(self.L)]
        self.A[t] = np.argsort(self.U[t])[-self.K:][::-1]
        # get reward
        reward = dataset[t][self.A[t]]
        # compute regret
        immediate_regret = self.best_f-self.f(t)
        self.regrets[t] = self.regrets[t-1] + np.abs(immediate_regret)
        # get index of  attractive item
        if np.sum(reward) > 0:
            self.C[t] = np.argmax(reward == 1)
        else:
            self.C[t] = 1e6
        self.update_weights(t)
        return True


def run_experiment(list_L, list_delta, list_K, number_of_rounds,n_runs):
    n_regret = np.zeros((len(list_L),n_runs))
    for i, (L, K , delta)  in enumerate(zip(list_L,list_K,list_delta) ):
        for run in range(n_runs) : 
            weights = [p for i in range(K)] + [np.abs(p-delta) for i in range(L-K)]
            cascade_model = CascadeUCB(number_of_rounds,L,K)
            dataset = generate_data(number_of_rounds, weights)
            cascade_model.initialize(dataset,weights)
            for t in range(1,number_of_rounds) :
                cascade_model.one_round(t,dataset)
            n_regret[i,run]=cascade_model.regrets[-1]
    res = pd.DataFrame({'L': list_L ,
              'K':list_K, 
              'delta': list_delta,
              'mean' :n_regret.mean(axis=-1),
              'std': n_regret.std(axis=-1)})
    return res , n_regret
    
