# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:39:45 2018

@author: CIMlab徐孟維
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import numpy as np
import Routing
import random as rd
#import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


sp = Routing.ShortestPath

class Status:
    def __init__(self, Controller):
        self.Range = [[ 61. ,  50.4,  39.8,  29.2,  18.6,   8. ],#Job number
                      [458, 410, 363, 315, 268, 221],#Mean Remaining Operation Time
                      [445, 365, 285, 205, 125, 45],#stay mean
                      [215, 174, 133, 92, 51, 10],#stay std
                      [1018, 688, 358, 28, -301, -631], #rdd mean
                      [1074, 901, 728, 554, 381, 208],#rdd std
                      [-105, -152, -199, -246, -293, -340],#slack mean
                      [164, 136, 108, 80, 52, 24],#slack std
                      [135, 110, 85, 60, 35, 10],#nrt
                      [114, 93, 72, 51, 30, 9],#wl mean
                      [0.0165, 0.0132, 0.0099, 0.0066, 0.0033, 0]]#u std

        self.Controller = Controller
        self.Jnum = []
        self.SL_mean = []
        self.SL_std = []
        self.NO_mean = []
        self.NO_std = []
        self.NO_max = []
        self.U_mean = []
        self.U_std = []
        self.Ele_mean = []
        self.Ele_low = []
        self.WL_mean = []
        self.WL_std = []
        self.NIV = []
        self.NRT = []

        self.Stay_mean = []
        self.Stay_std = []
        self.RDD_mean = []
        self.RDD_std = []
        self.ROT_mean = []
        self.ROT_std = []

        #Performance
        self.tardiness = []
        self.throughput = 0
        self.meantardiness = 0

        self.currentstate = [0, 0]
        self.currentact = [0, 0]
        self.currentDegree = [0, 0]

    def Get(self, Type):
        self.Update()

        if Type == "V" or Type == "W":
            return [self.Jnum, self.ROT_mean, self.ROT_std, self.SL_mean, self.SL_std, self.RDD_mean, self.RDD_std, self.NO_mean
                    , self.NO_std, self.NO_max, self.U_mean, self.U_std, self.Ele_mean, self.Ele_low, self.WL_mean, self.WL_std, self.RDD_mean
                    , self.RDD_std, self.NIV, self.NRT, self.Stay_mean, self.Stay_std]
        elif Type == "v":
            #1# Number of Job
            Jnum = 0
            for i in range(6):
                if self.Jnum < self.Range[0][i]:
                    Jnum = i
            #Mean Remaining Operation Time
            ROT_mean = 0
            for i in range(6):
                if self.ROT_mean < self.Range[1][i]:
                    ROT_mean = i
            #4# Mean Remaining Due Date
            RDD_mean = 0
            for i in range(6):
                if self.RDD_mean < self.Range[4][i]:
                    RDD_mean = i

            #5# Std Remaining Due Date
            RDD_std = 0
            for i in range(6):
                if self.RDD_std < self.Range[5][i]:
                    RDD_std = i
            return [Jnum, ROT_mean, RDD_mean, RDD_std]
        elif Type == "w":
            #Number of Job
            Jnum = 0
            for i in range(6):
                if self.Jnum < self.Range[0][i]:
                    Jnum = i
            #Mean Remaining Operation Time
            ROT_mean = 0
            for i in range(6):
                if self.ROT_mean < self.Range[1][i]:
                    ROT_mean = i
            #Mean Remaining Due Date
            RDD_mean = 0
            for i in range(6):
                if self.RDD_mean < self.Range[4][i]:
                    RDD_mean = i
            RDD_std = 0
            for i in range(6):
                if self.RDD_std < self.Range[5][i]:
                    RDD_std = i
            return [Jnum, ROT_mean, RDD_mean, RDD_std]



    def show(self):
        [self.Jnum, self.ROT_mean, self.ROT_std, self.SL_mean, self.SL_std, self.RDD_mean, self.RDD_std, self.NO_mean
                    , self.NO_std, self.NO_max, self.U_mean, self.U_std, self.WL_mean, self.WL_std, self.RDD_mean
                    , self.RDD_std, self.NIV, self.NRT, self.Stay_mean, self.Stay_std]

        self.Update()
        # #1
        # print("Jnum", self.Jnum)
        #
        # #6################################
        # print("ROT_mean", self.ROT_mean)
        # #7
        # print("ROT_std", self.ROT_std)
        #
        # #8################################
        # print("SL_mean", self.SL_mean)
        # #9
        # print("SL_std", self.SL_std)
        #
        # #11
        # print("NO_mean", self.NO_mean)
        # #12
        # print("NO_std", self.NO_std)
        # #13
        # print("NO_max", self.NO_max)
        #
        # #16
        # print("U_mean", self.U_mean)
        # #17
        # print("U_std", self.U_std)
        #
        # #19
        # print("Ele_mean", self.Ele_mean)
        # #20
        # print("Ele_low", self.Ele_low)
        #
        # #14
        # print("WL_mean", self.WL_mean)
        # #15
        # print("WL_std", self.WL_std)
        #
        # #4
        # print("RDD_mean", self.RDD_mean)
        # #5
        # print("RDD_std", self.RDD_std)
        #
        # #16
        # print("NIV", self.NIV)
        # #10
        # print("NRT", self.NRT)
        #
        # #2
        # print("Stay_mean", self.Stay_mean)
        # #3
        # print("Stay_std", self.Stay_std)

    def Update(self):

        #Number of jobs in system
        self.Jnum = len(self.Controller.sysJob)
        #Mean & Standard deviation & Maximum value of output queue number
        NO = []
        for i in range(1, len(self.Controller.stations)-1):
            no = self.Controller.stations[i].out_num
            NO.append(no)
        self.NO_mean = float(round(np.mean(NO),2))
        self.NO_std = float(round(np.std(NO),2))
        self.NO_max = np.max(NO)
        #AGV utilities
        U = []
        Ele = []
        low = 0
        for i in self.Controller.AGVs:
            if self.Controller.Time != 0:
                u = (self.Controller.Time - i.IdleTime)/self.Controller.Time
            else:
                u = 0
            if i.Electricity < 0.2:
                low += 1
            Ele.append(i.Electricity)
            U.append(u)
        self.U_mean = float(round(np.mean(U),4))
        self.U_std = float(round(np.std(U),4))
        self.Ele_mean = float(round(np.mean(Ele),4))
        self.Ele_low = low
        #Number of Idling Vehicle in system
        self.NIV = self.Controller.IV_num
        #Remaining Traveling numbers of system
        self.NRT = self.Controller.Rtravel
        #Mean & Minimum value of workload of each workstations
        #Mean and Standard deviation of Stay time
        #Mean and Standard deviation of Remaining Due Date
        #Mean and Standard deviation of Remaining Operation Time
        WL = np.zeros((self.Controller.WS_num))
        Stay = []
        RDD = []
        ROT = []
        SL = []
        for i in self.Controller.sysJob:
            stay = self.Controller.Time - i.FlowTime
            Stay.append(stay)
            rdd = i.DueDate - self.Controller.Time
            RDD.append(rdd)
            rpt = 0
            rtt = 0
            for j in range(i.current_seq, len(i.seq)):
                rpt += i.PT[j]
            for j in range(i.current_seq, len(i.seq)-1):
                WL[j] += i.PT[j]
                f,t = i.seq[j], i.seq[j+1]
                rtt += self.Controller.FTmatrix[f][t]
            rot = rpt + rtt
            ROT.append(rot)
            sl = rdd - rot
            SL.append(sl)
        if WL != []:
            self.WL_mean = np.mean(WL)
            self.WL_std = np.std(WL)
        else:
            self.WL_mean = 0
            self.WL_std = 0
        if Stay != []:
            self.Stay_mean = np.mean(Stay)
            self.Stay_std = np.std(Stay)
        else:
            self.Stay_mean = 0
            self.Stay_std = 0
        if RDD != []:
            self.RDD_mean = np.mean(RDD)
            self.RDD_std = np.std(RDD)
        else:
            self.RDD_mean = 0
            self.RDD_std = 0
        if ROT !=[]:
            self.ROT_mean = np.mean(ROT)
            self.ROT_std = np.std(ROT)
        else:
            self.ROT_mean = 0
            self.ROT_std = 0
        if SL !=[]:
            self.SL_mean = np.mean(SL)
            self.SL_std = np.std(SL)
        else:
            self.SL_mean = 0
            self.SL_std = 0


class DQN:
    def __init__(self, n_actions, n_features, LR, R_disc, greedy, greedy_incre
               , replace_iter, memory_size, batch_size, Type, startLearning, wantLearn):
        self.n_action = n_actions
        self.n_feature = n_features
        self.LR = LR
        self.gamma = R_disc
        self.epsilon_max = greedy
        self.epsilon_incre = greedy_incre
        self.epsilon = 0.
        self.replace_iter = replace_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.Type = Type

        self.wantLearn = wantLearn
        self.startLearning = startLearning

        self.Lcount = 0
        self.memory = np.zeros((self.memory_size, self.n_feature*2+2))

        self.tempstore = []

        with tf.device('/gpu:0'):

            self.build_net()
            self.replace_para()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.cost_his = []
        self.performance_his = []

        self.Rewards = []
        self.Actions = []

        if not hasattr(self, "memory_count"):
            self.memory_count = 0

        for i in range(n_actions):
            self.Actions.append(0)


    def build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_feature], name = "s")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_feature]\
                                 , name = "s_")
        self.Q_target = tf.placeholder(tf.float32, [None, self.n_action]\
                                       , name = "Q_target")
        self.r = tf.placeholder(tf.float32, [None, ], name = "r")
        self.a = tf.placeholder(tf.int32, [None, ], name = "a")
        self.nl1 = 64
        self.nl2 = 32
        w_init = tf.random_normal_initializer(0., 0.1)
        b_init = tf.constant_initializer(0.1)
        if self.Type == "V":
            # Q_Network
            with tf.variable_scope("Q_netV", reuse=tf.AUTO_REUSE):
                el1 = tf.layers.dense(self.s, self.nl1, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "el1")
                el2 = tf.layers.dense(el1, self.nl2, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "el2")
                self.Q_eval = tf.layers.dense(el2, self.n_action\
                                              , kernel_initializer = w_init\
                                              , bias_initializer = b_init\
                                              , name = "Q")
            # Target_Network
            with tf.variable_scope("target_netV", reuse=tf.AUTO_REUSE):
                tl1 = tf.layers.dense(self.s_, self.nl1, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "tl1")
                tl2 = tf.layers.dense(tl1, self.nl2, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "tl2")
                self.Q_next = tf.layers.dense(tl2, self.n_action\
                                              , kernel_initializer = w_init\
                                              , bias_initializer = b_init\
                                              , name = "Q2")
            # Target Value
            with tf.variable_scope("Q_targetV", reuse=tf.AUTO_REUSE):
                Q_target = self.r + self.gamma*tf.reduce_max(self.Q_next, axis = 1\
                                                             , name = "Q_max")
                self.Q_target = tf.stop_gradient(Q_target)
            # Evaluate
            with tf.variable_scope("Q_evalV", reuse=tf.AUTO_REUSE):
                A_indice = tf.stack([tf.range(tf.shape(self.a)[0]\
                                              , dtype = tf.int32), self.a]\
                                                , axis = 1)
                self.Q_eval_A = tf.gather_nd(params = self.Q_eval\
                                             , indices = A_indice)
        elif self.Type == "W":
            # Q_Network
            with tf.variable_scope("Q_netW", reuse=tf.AUTO_REUSE):
                el1 = tf.layers.dense(self.s, self.nl1, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "el1")
                el2 = tf.layers.dense(el1, self.nl2, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "el2")
                self.Q_eval = tf.layers.dense(el2, self.n_action\
                                              , kernel_initializer = w_init\
                                              , bias_initializer = b_init\
                                              , name = "Q")
            # Target_Network
            with tf.variable_scope("target_netW", reuse=tf.AUTO_REUSE):
                tl1 = tf.layers.dense(self.s_, self.nl1, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "tl1")
                tl2 = tf.layers.dense(tl1, self.nl2, tf.nn.relu\
                                      , kernel_initializer = w_init\
                                      , bias_initializer = b_init, name = "tl2")
                self.Q_next = tf.layers.dense(tl2, self.n_action\
                                              , kernel_initializer = w_init\
                                              , bias_initializer = b_init\
                                              , name = "Q2")
            # Target Value
            with tf.variable_scope("Q_targetW", reuse=tf.AUTO_REUSE):
                Q_target = self.r + self.gamma*tf.reduce_max(self.Q_next, axis = 1\
                                                             , name = "Q_max")
                self.Q_target = tf.stop_gradient(Q_target)
            # Evaluate
            with tf.variable_scope("Q_evalW", reuse=tf.AUTO_REUSE):
                A_indice = tf.stack([tf.range(tf.shape(self.a)[0]\
                                              , dtype = tf.int32), self.a]\
                                                , axis = 1)
                self.Q_eval_A = tf.gather_nd(params = self.Q_eval\
                                             , indices = A_indice)
        # Loss
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Q_target\
                                                        , self.Q_eval_A\
                                                        , name = "TD_error"))
        # Train
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.RMSPropOptimizer(self.LR).minimize(\
                                                     self.loss)

        ### Train Initial Net
        #Load Data
        if self.Type == "V":
            data = pd.read_csv("V5000_8.csv").iloc[: ,1:]
        elif self.Type == "W":
            data = pd.read_csv("W5000_8.csv").iloc[: ,1:]

        self.f = data.iloc[:, :-1].values.tolist()
        self.l = data["class"].tolist()


        # Multi-class Output
        N_label = self.n_action
        self.labels = (np.arange(N_label) == np.array(self.l)[:, None]).astype(np.float32)

        #self.Q_predict = (np.arange(N_label) == np.array(self.Q_eval[:, None])).astype(np.float32)


        #Loss function
        with tf.variable_scope("Init_loss"):
            self.init_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Q_eval, labels=self.labels))
        #optimizer
        with tf.variable_scope("Init_opt", reuse=tf.AUTO_REUSE):
            self.init_opt = tf.train.AdamOptimizer(0.01).minimize(self.init_loss)

        self.train_prediction = tf.nn.softmax(self.Q_eval)



    def store_transition(self, s, a, s_, r):
        if not hasattr(self, "memory_count"):
            self.memory_count = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % self.memory_size
        self.memory[index, :] = transition
        self.memory_count += 1

    def store_temp(self, s, a, s_):
        if not hasattr(self, "memory_count"):
            self.memory_count = 0
        self.tempstore.append([s, a, s_])

    def Choose_act(self, status):
        status = np.array(status)[np.newaxis, :]
        if rd.random() < self.epsilon:
            Q_values = self.sess.run(self.Q_eval, feed_dict = {self.s: status})
            action = np.argmax(Q_values)
        else:
            action = rd.randint(0, self.n_action-1)
        if self.Type == "V":
            self.Actions[action] += 1
        elif self.Type == "W":
            self.Actions[action] += 1
        return action

    def replace_para(self):
        if self.Type == "V":
            t_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES\
                                       , scope = "target_netV")
            Q_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES\
                                       , scope = "Q_netV")
            with tf.variable_scope("soft_replacement"):
                self.para_replace = [tf.assign(t, e) for t, e in zip(t_para\
                                     , Q_para)]
        elif self.Type == "W":
            t_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES\
                                       , scope = "target_netW")
            Q_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES\
                                       , scope = "Q_netW")
            with tf.variable_scope("soft_replacement"):
                self.para_replace = [tf.assign(t, e) for t, e in zip(t_para\
                                     , Q_para)]

    def Learning(self):

        if self.Lcount % self.replace_iter == 0:
            self.replace_para()
        if self.memory_count > self.startLearning: ############################################################
#        if True: #self.memory_count > 300: ############################################################
            if self.memory_count > self.memory_size:
                index = np.random.choice(self.memory_size, self.batch_size, replace = True)
            else:
                index = np.random.choice(self.memory_count, self.batch_size, replace = True)
            batch = self.memory[index, :]
            temp, cost = self.sess.run([self.train_op, self.loss], feed_dict = {\
                          self.s: batch[:, :self.n_feature]\
                          , self.a: batch[:, self.n_feature]\
                          , self.r: batch[:, self.n_feature+1]\
                          , self.s_: batch[:, -self.n_feature:]})

#            print('========={}============{}'.format(temp, cost))
            self.cost_his.append(cost)

            self.Lcount += 1


            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_incre

#            if self.Lcount >= 5000:
#                self.epsilon = self.epsilon_max

    def accuracy(self, predictions, labels):
        temp= []
        for i in range(1500):
            temp.append(predictions[i].argmax())
        print(set(temp))
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def Init_Net(self, times):
        self.sess.run(tf.global_variables_initializer())
        for step in range(times):
            temp, cost, pre = self.sess.run([self.init_opt, self.init_loss, self.train_prediction], feed_dict = {self.s: self.f})
            '''
            if step % 10 == 0:
                print('step:{} loss:{:.6f} accuracy: {:.2f}'.format(step, cost, self.accuracy(pre, self.labels)))'''
        self.replace_para()


    def Save_net(self):
        if self.Type == "V":
            self.saver.save(self.sess, "test/modelV-1.cpkt")
        elif self.Type == "W":
#            self.saver = tf.train.Saver()
            self.saver.save(self.sess, "test/modelW-1.cpkt")

    def Load_net(self):
        if self.Type == "V":
#            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, "test/modelV-1.cpkt")
        elif self.Type == "W":
#            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, "test/modelW-1.cpkt")

    def plot_cost(self, times=0, case=0):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if case == 0:
            plt.savefig("Result/cost_-{}.png".format(times))
        else:
            plt.savefig("Result/cost_-{}-1.png".format(times))
#        plt.show()

    def plot_performance(self, times=0, case=0):
        plt.figure()
        plt.plot(np.arange(len(self.performance_his)), self.performance_his)
        plt.ylabel('MeanTardiness')
        plt.xlabel('episode')
        plt.title("Performance - Episode")
        if case == 0:
            plt.savefig("Result/result_-{}.png".format(times))
        else:
            plt.savefig("Result/result_-{}-1.png".format(times))
#        plt.show()
