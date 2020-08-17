# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:51:17 2018

@author: CIMlab徐孟維
"""
import simpy
# from ShopFloor_noTk import Center
from ShopFloor_1pts_reward import Center
import Learn as Le
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
#import pickle

Task = "None" #TestD, TestQ, TestDQN, TestSVM, CollectV, CollectW, TrainQ, TrainDQN
Task = "TrainDQN"
'''
np.random.seed(124)
env = simpy. Environment()
Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=2, AGV_disRuleW=2, Ledispatch = "None")
env.run(until = 10000)
'''

Testing = False
WANTLEARN = False #Learning per episode

wantLearn_everyStep = True
pretrained = False

if Task == "TrainDQN":

    le = []
#        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.05, R_disc = 0.01, greedy = 0.9\
#                         , greedy_incre = 0.04, replace_iter = 25\
#                         , memory_size = 30000, batch_size = 300, Type = "V"))
    le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.03, R_disc = 0.9, greedy = 0.5\
                     , greedy_incre = 0.004, replace_iter = 25\
                     , memory_size = 30000, batch_size = 300, Type = "V", startLearning = 3000, wantLearn = wantLearn_everyStep))
    le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.03, R_disc = 0.9, greedy = 0.5\
                     , greedy_incre = 0.004, replace_iter = 25\
                     , memory_size = 30000, batch_size = 300, Type = "W", startLearning = 3000, wantLearn = wantLearn_everyStep))

    if pretrained == True:
        for model in le:
            model.Load_net()
            model.epsilon = 0.7
            model.epsilon_max = 0.999
            #model.RL = 0.001

#    for DOE in tqdm(range(30)):#len(exp_case):
    for DOE in range(1):
#        le = []
##        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.05, R_disc = 0.01, greedy = 0.9\
##                         , greedy_incre = 0.04, replace_iter = 25\
##                         , memory_size = 30000, batch_size = 300, Type = "V"))
#        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.03, R_disc = 0.9, greedy = 0.5\
#                         , greedy_incre = 0.004, replace_iter = 25\
#                         , memory_size = 30000, batch_size = 100, Type = "V"))
#        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.03, R_disc = 0.9, greedy = 0.5\
#                         , greedy_incre = 0.004, replace_iter = 25\
#                         , memory_size = 30000, batch_size = 300, Type = "W"))
#
#        for model in le:
#            model.Load_net()
#            model.epsilon = 1#0.5
#            model.epsilon_max = 1#0.999
#            model.RL = 0.001

        expt = []
        expm = []
        expaV = []
        expaW = []
        for i in tqdm(range(1000)):
#        for i in range(1):
#            print('episode: ', i)
            '''
            np.random.seed(123)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=4, Ledispatch = le)
            Controller.Period = 2000
            env.run(until = 10001)

            le[1].Learning()
            print("{}.MeanTardiness: {}".format(i, Controller.MeanTardiness))
            le[1].performance_his.append(Controller.MeanTardiness)
            '''
#            np.random.seed(999)
#            fix_seed = DOE
            env = simpy. Environment()
            Controller = Center(None, env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=4, Ledispatch = le)
            Controller.Period = 2000
            env.run(until = 20001)


            expaV.append(list(le[0].Actions))
            for j in range(6):
                le[0].Actions[j] = 0
            expaW.append(list(le[1].Actions))
            for j in range(3):
                le[1].Actions[j] = 0
            expt.append(Controller.Throughput)
            expm.append(Controller.MeanTardiness)


            if le[0].wantLearn == False and le[1].wantLearn == False and WANTLEARN == True:
                le[0].Learning()
                le[1].Learning()

            #print("Throughput: {}".format(Controller.Throughput))
#            print("{}.MeanTardiness: {}".format(i, Controller.MeanTardiness))
            le[0].performance_his.append(Controller.MeanTardiness)




        t1 = pd.DataFrame(expaV)
        t1.columns = ["STT", "MOQS", "MFCFS", "DS", "EDD", "DS"]
        t2 = pd.DataFrame(expaW)
        t2.columns = ["NV", "LIV", "LU"]
        t3 = pd.DataFrame(expt)
        t3.columns = ["Throughput"]
        t4 = pd.DataFrame(expm)
        t4.columns = ["MeanTardiness"]
        LER = pd.concat([t1, t2, t3, t4], axis = 1)
        LER.to_csv("Result/Performance-{}.csv".format(DOE))


        le[0].plot_performance(times=DOE, case=0)
        le[1].plot_performance(times=DOE, case=1)

        le[0].plot_cost(times=DOE, case=0)
        le[1].plot_cost(times=DOE, case=1)

#        print(model.epsilon for model in le)

        #save the weights
        le[1].Save_net()
        le[0].Save_net()
#
#    elif Task == "TestDQN":
#        le = []
#        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
#                         , greedy_incre = 0.005, replace_iter = 25\
#                         , memory_size = 3000, batch_size = 300, Type = "V"))
#        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
#                         , greedy_incre = 0.005, replace_iter = 25\
#                         , memory_size = 300, batch_size = 30, Type = "W"))
#
#        le[0].Load_net()
#        le[1].Load_net()
#
#        le[0].epsilon = 1
#        le[1].epsilon = 1
#        performance = []
#        for i in range(30):
#            np.random.seed(124+i)
#            env = simpy. Environment()
#            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
#            env.run(until = 10000)
#            performance.append(Controller.MeanTardiness)
#            print("{}. MeanTardiness: {}".format(i, Controller.MeanTardiness))
#
#        P = pd.DataFrame(performance)
#        P.columns = ["Mean Tardiness"]
#        P.to_csv("Result/Performance.csv")


'''
le = []
le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.05, R_disc = 0.01, greedy = 0.9\
                 , greedy_incre = 0.04, replace_iter = 25\
                 , memory_size = 30000, batch_size = 300, Type = "V"))
le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.05, R_disc = 0.01, greedy = 0.9\
                 , greedy_incre = 0.04, replace_iter = 25\
                 , memory_size = 30000, batch_size = 300, Type = "W"))

le[0].Init_Net(100)
le[0].epsilon = 1

for i in range(1):
    np.random.seed(124)
    env = simpy. Environment()
    Controller = Center(env, x=16, y=16, routRule=1, AGV_num=8, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
    Controller.Period = 500
    env.run(until = 10001)

    print("MeanTardiness: {}".format(Controller.MeanTardiness))'''
