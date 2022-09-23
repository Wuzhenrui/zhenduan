# encoding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix


class ClassifyEnv(gym.Env):

    def __init__(self, mode, imb_rate, trainx, trainy, ):  # mode means training or testing
        self.mode = mode
        self.imb_rate = imb_rate

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])          #shape[0]  输出矩阵的行数  shape[1]输出矩阵有多少列

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)
        print(self.action_space)
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)
        y_true_cur = []
        info = {}
        terminal = False
        if a == self.Answer[self.id[self.step_ind]]:
            if self.Answer[self.id[self.step_ind]] == 0:            #第一个正确的   给1  后面的要乘一个系数
                reward = 1.
            else:
                reward = 1. * self.imb_rate
        else:
            if self.Answer[self.id[self.step_ind]] == 0:          #同理
                reward = -1.
                if self.mode == 'train':
                    terminal = True
            else:
                reward = -1. * self.imb_rate
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:       #步数等于游戏总会和数   ---------------------------------------------------------
            y_true_cur = self.Answer[self.id]
            info['gmean'], info['fmeasure'] = self.My_metrics(np.array(self.y_pred),
                                                              np.array(y_true_cur[:self.step_ind]))
            terminal = True

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info

    def My_metrics(self, y_pre, y_true):
        #confusion_matrix  混淆矩阵  也是误差矩阵  输入真实值和预测值   然后输出一个矩阵
       # array([[1, 1, 0],
       # [1, 1, 0],
       # [1, 0, 2]])

        confusion_mat = confusion_matrix(y_true, y_pre)
        print('\n')
        print(classification_report(y_true, y_pre))
        conM = np.array(confusion_mat, dtype='float')
        TP = conM[1][1]
        TN = conM[0][0]
        FN = conM[1][0]
        FP = conM[0][1]
        TPrate = TP / (TP + FN)  # 真阳性率
        TNrate = TN / (TN + FP)  # 真阴性率
        FPrate = FP / (TN + FP)  # 假阳性率
        FNrate = FN / (TP + FN)  # 假阴性率
        PPvalue = TP / (TP + FP)  # 阳性预测值
        NPvalue = TN / (TN + FN)  # 假性预测值

        G_mean = np.sqrt(TPrate * TNrate)

        Recall = TPrate = TP / (TP + FN)
        Precision = PPvalue = TP / (TP + FP)
        F_measure = 2 * Recall * Precision / (Recall + Precision)
        print(confusion_mat)
        res = 'G-mean:{}, F_measure:{}\n'.format(G_mean, F_measure)
        print(res)
        print()
        return G_mean, F_measure               #计算G_mean 和 F_mean

    # return: (states, observations)
    def reset(self):
        if self.mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]
