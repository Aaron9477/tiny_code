# -*-coding:utf-8-*-

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk    # GUI
import utils as u

UNIT = 40   # 单位长度的像素值
MAZE_H = 20  # 高度/UNIT
MAZE_W = 20  # 长度/UNIT
TOTAL_HIGH = UNIT * MAZE_H  # 高度/像素
TOTAL_LENGTH = UNIT * MAZE_W    # 长度/像素
HELF_HIGH = 0.5 * TOTAL_HIGH    # 半高
MIN_D = UNIT    # 初始最小距离
MAX_D = 10*UNIT # 初始最远距离
STEP_SIZE = 0.5 # 每次移动多大的UNIT

# 针对不同任务类型、风速、降雨量，需要的两机距离上下界
T_down = [10, 20, 40]
T_up = [20, 40, 80]
W_down = [10,11,14,19,25]
R_down = [8,10,12,15,20]
# 执行任务时的任务类型、风速、降雨量
# INPUT = [1,2,3]
# INPUT_RATE = np.array([INPUT[0]/len(T_down), INPUT[1]/len(W_down), INPUT[2]/len(R_down)])

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['l', 'r', 'n'] # left, right, do nothing
        self.n_actions = len(self.action_space)
        self.n_features = 4 # distance target wind rain
        self.min_range = 6  # 飞机间合理距离范围的最小值
        self.input = None   # 环境信息 随机产生
        self.start_point = None # 无人机起始位置
        self.distance_range = None  # 得到无人机之间的距离范围 这个距离不能过小，像[18，20]会导致震荡
        self.input_rate = None  # 将输入归一化 神经网络输入都要归一化！！！！！！
        self.random_choose_get_range()  # 随机生成合理的环境信息 并得到范围和归一化的值
        self.best_distance = 0  # 每次实验中的奖励值最大的位置
        self.stay_time = 0  # 记录无人机持续在规定范围内的时间长度
        self.distance_group = []    # 记录在一次实验中的距离变化
        self.distance = [] #记录两机之间的距离
        self.error_history = [] # 记录整轮实验中的误差变化趋势
        self.reward_group = []   # 记录在一次实验中的奖励变化
        self.reward_history = [] # 记录整轮实验中的奖励变化趋势
        self.period_choose = 5  # 每次用于计算最终稳定距离大小和误差取的时间片长度
        # self.done_time = 5 # 规定无人机持续在规定范围内多久才可以置done标志位 从而进入下一次循环
        self.done_time = 90 # 规定无人机运动多少次之后才可以置done标志位 从而进入下一次循环
        # self.title('maze')
        # self.geometry('{0}x{1}'.format(TOTAL_HIGH, TOTAL_LENGTH))   # 画布大小
        self._build_maze()  # 在init中执行，相当于构造函数

    def _build_maze(self):  # GUI, wait changing!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.canvas = tk.Canvas(self, bg='white',   # 画布大小设定
                           height=TOTAL_HIGH,
                           width=TOTAL_LENGTH)

        # 画一条线
        self.canvas.create_line(0, 0.5*TOTAL_HIGH, TOTAL_LENGTH, 0.5*TOTAL_HIGH)
        # 随机产生初始点和目标点
        origin, oval_center = self.get_random_point() # 随机得到初始点和目标点
        # 固定目标点为中心，初始点为距离26
        #origin, oval_center = self.get_dixtance_fixed_26()
        # 红正方形目标
        self.rect = self.canvas.create_rectangle(
            origin - 10, HELF_HIGH - 10,
            origin + 10, HELF_HIGH + 10,
            fill='red')

        # 黄色目标点
        self.oval = self.canvas.create_oval(
            oval_center - 10, HELF_HIGH - 10,
            oval_center + 10, HELF_HIGH + 10,
            fill='yellow'
        )

        # pack all
        # self.canvas.pack()


    def reset(self):    # 每次测试初始化
        self.update()
        time.sleep(0.1) # 进行下次探索等待时间    # 为什么只有第一次会等待？ 是因为根本就到不了终点，如果一直到不了终点会一直跑
        self.canvas.delete(self.rect)   # 删除上次的控制点
        self.canvas.delete(self.oval)

        # 随机产生初始点和目标点
        origin, oval_center = self.get_random_point() # 随机得到初始点和目标点
        # 固定目标点为中心，初始点为距离26
        #origin, oval_center = self.get_dixtance_fixed_26()
        self.rect = self.canvas.create_rectangle(   # 绘图
            origin - 10, HELF_HIGH - 10,
            origin + 10, HELF_HIGH + 10,
            fill='red')

        self.oval = self.canvas.create_oval(
            oval_center - 10, HELF_HIGH - 10,
            oval_center + 10, HELF_HIGH + 10,
            fill='yellow'
        )
        self.random_choose_get_range()    # 每次reset都需要重新随机生成合理的环境信息 并得到范围和归一化的值

        # 得到距离信息 并归一化 此处可以把除数换成总长的一半，因为目标点一直在中心
        distance_rate = (self.canvas.coords(self.rect)[0] - np.array(self.canvas.coords(self.oval)[0])) / TOTAL_LENGTH
        print(np.append(self.input_rate, distance_rate))
        # 返回未归一化的环境信息
        env = [T_down[self.input[0]], W_down[self.input[1]], R_down[self.input[2]], distance_rate]
        return (np.array(env))  # 这个函数在run_this中运行的,每次初始化后返回当前的状态,作为一个状态
        # 返回归一化的环境信息
        # return (np.append(self.input_rate, distance_rate))

        # print((np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / TOTAL_LENGTH)
        # return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / TOTAL_LENGTH

    def step(self, action):
        s = self.canvas.coords(self.rect)   # 原位置 返回图形的当前坐标（左上角坐标，右下角坐标）
        base_action = np.array([0, 0])  # 基准点
        done = False    # 定义done
        # 防止走出画布
        if action == 0:   # left
            if s[0] > self.canvas.coords(self.oval)[0] - self.range_up * STEP_SIZE * UNIT:  #与邻机相距20m
                base_action[0] -= 0.1*STEP_SIZE*UNIT
            else:
                pass
        elif action == 1:   # right
            if s[0] < self.canvas.coords(self.oval)[0] - self.range_down * STEP_SIZE * UNIT:  #与邻机相距11m
                base_action[0] += 0.1*STEP_SIZE*UNIT
            else:
                pass
        elif action == 2:   # do nothing
            pass

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动

        next_coords = self.canvas.coords(self.rect)  # 下一个状态

        # 两飞机之间的距离，必须保留正负号进行return，否则不知道目标在左边还是右边
        self.distance = abs(next_coords[0] - self.canvas.coords(self.oval)[0]) / (STEP_SIZE * UNIT)
       
        print("飞机间的距离是： ", self.distance)
   
        # reward function
        # if next_coords == self.canvas.coords(self.oval):    # 两机相撞，结束
        #     reward = -50
        #     done = True
        # elif distance<self.distance_range[0] or distance>self.distance_range[1]:    # 在范围外就负奖励
        #     reward = -20
        #     done = False
        #     self.stay_time = 0  # 出了规定区域，重置时间
        # else:
        #     reward = int(self.get_reward(distance))  # 根据两机距离计算奖励值，神经网络只能输入int型！！！！！！！！！！！！！！！！！
        #     self.stay_time += 1 # 保持在规定区域
        #     done = False

        if self.distance < self.distance_range[0]:   # 在范围外就负奖励
            #  这里加两个端点的奖励值，是为了让奖励值平滑，发现奖励值平滑容易收敛
            reward = self.distance - self.distance_range[0] + self.get_reward(self.distance_range[0])
        elif self.distance > self.distance_range[1]:
            reward = self.distance_range[1] - self.distance + self.get_reward(self.distance_range[1])
        #elif distance == self.best_distance:
        #    reward = 100
        else:
            reward = int(self.get_reward(self.distance))  # 根据两机距离计算奖励值，神经网络只能输入int型！！！！！！
        self.stay_time += 1 # 运行时间+1
        # 记录每次运行的结果,当done之后,来取最后n次的值,之后清空
        self.distance_group.append(self.distance)
        self.reward_group.append(reward)

        if self.stay_time >= self.done_time:
            done = True
            self.stay_time = 0  # 本次结束，清零
            result_distance = sum(self.distance_group[-self.period_choose:])/self.period_choose # 取后5次做平均
            #error = u.error_square(self.distance_group[-self.period_choose:], self.best_distance)   # 取后5次的误差
            error = result_distance - self.best_distance
            last_reward = sum(self.reward_group[-self.period_choose:])  # 求最后五次的奖励和
            self.error_history.append(error)
            self.reward_history.append(last_reward)
            print('最后一段时间的平均距离是：', result_distance)
            print('最后一段时间的误差是：', error)
            print('最后一段时间的奖励和是：', last_reward)
            self.distance_group.clear() # 清空本次仿真记录的值
            self.reward_group.clear()

        print('当前奖励值为：', reward)

        distance_rate = (self.canvas.coords(self.rect)[0] - np.array(self.canvas.coords(self.oval)[0])) / TOTAL_LENGTH
        s_ = (np.append(self.input_rate, distance_rate))    # 下一步的状态，input_rate包括环境信息，加上距离信息

        print(self.return_reward_next_step())

        # s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / TOTAL_LENGTH   # s_是相对于终点的坐标差
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()
        pass

    # 随机得到初始点和重点的位置 暂时不用
    def get_random_point(self):
        target_fixed = 0.5*UNIT*MAZE_W
        start_ran = random.randrange(self.range_down, self.range_up, 1)
        start_random = target_fixed - start_ran * STEP_SIZE * UNIT
        return start_random, target_fixed

    # 初始化 目标点在中心 起始点距离为固定16个单位
    def get_dixtance_fixed_26(self):
        target_fixed = 0.5*UNIT*MAZE_W
        #judge = random.randint(0,1) # 随机选择初始点在左边还是右边    # python中0和负数代表false，正数代表true
	#if judge:
        start_random = target_fixed - 16 * STEP_SIZE * UNIT
        #else:
        #    start_random = target_fixed + 26 * STEP_SIZE * UNIT
        self.start_point = start_random
        self.neighbor_point = target_fixed
        return start_random, target_fixed

    # 得到距离范围 是random_choose_get_range函数中的一部分 暂时不用
    def get_distance_range(self):
        # print(T_down[INPUT[0]])
        # print(W_down[INPUT[1]])
        # print(R_down[INPUT[2]])
        self.range_down = max(T_down[self.input[0]], W_down[self.input[1]], R_down[self.input[2]])
        self.range_up = T_up[self.input[0]]
        if self.range_down < self.range_up:
            return [self.range_down, self.range_up]
        else:
            print("出错，距离下界大于距离上界")
            exit()

    # 随机产生环境信息 判断是否合理 输出两机
    def random_choose_get_range(self):
        # 基本random包中的randint必须输入两个参数（上界和下界） numpy中的random.randint可以只输入一个参数n，范围[0,n-1]
        print("正在随机产生数据...")
        while True:
            # 随机产生数据
            self.input = [np.random.randint(len(T_down)), np.random.randint(len(W_down)), np.random.randint(len(R_down))]
            # 固定环境信息为 任务1 风速2 降雨量2
            #self.input = [0,1,1]
            self.range_down = max(T_down[self.input[0]], W_down[self.input[1]], R_down[self.input[2]])   # 得到距离最小值
            self.range_up = T_up[self.input[0]]  # 距离最大值
            if self.range_up-self.range_down >= self.min_range:   # 判断是否在合理距离范围内
                print("随机产生的数据为：任务类型：",self.input[0]+1,",风速：",self.input[1]+1,"，降雨量：",self.input[2]+1)
                # 归一化
                self.input_rate = np.array([self.input[0] / len(T_down), self.input[1] / len(W_down), self.input[2] / len(R_down)])
                self.distance_range = [self.range_down, self.range_up]
                # 简洁方式求得最大reward
                # self.distance_max_reward = max(list(map(lambda x: self.get_reward(x), self.distance_range)))
                self.best_distance, max_reward = self.get_distance_max_reward(self.distance_range)
                print("范围：", str(self.distance_range))
                print("最大奖励值的位置：", str(self.best_distance), " 最大奖励值为：", str(max_reward))
                break   # 数据合理 跳出
            else:
                print("随机得到的数据不符合要求，重新随机产生数据")

    # 如果在合理范围内计算奖励值
    def get_reward(self, d):
        #R = (100*d + 10*u.square(d) - 0.5*u.cube(d) -2100)*10
        R = (3*d + 10*u.square(d) - 0.25*u.cube(d))
        #R = (0.3*d + 0.2*0.05*u.square(d) + 0.5*10000/u.square(d)-40)
        return R

    def get_distance_max_reward(self, input):
        max_reward = -100
        this_distance = 0
        self.range_down=input[0]
        self.range_up=input[1]
        step=(input[1]-input[0])*10+1
        for i in u.floatrange(self.range_down, self.range_up, step):
            if self.get_reward(i) > max_reward:
                this_distance = i
                max_reward = self.get_reward(i)
        return [this_distance, max_reward]

    def plot_error_change(self):
        plt.plot(np.arange(len(self.error_history)), self.error_history)
        plt.ylabel('error')
        plt.xlabel('training steps')
        plt.show()

    #def plot_error_change(self):
        #plt.plot(np.arange(len(self.error1_history)), self.error1_history)
        #plt.ylabel('error1')
        #plt.xlabel('training steps')
        #plt.show()

    def plot_reward_change(self):
        plt.plot(np.arange(len(self.reward_history)), self.reward_history)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.show()

    def return_reward_next_step(self):
        # next_reward = np.array()
        red_rect = self.canvas.coords(self.rect)    # 控制的无人机横坐标
        yellow_oval = self.canvas.coords(self.oval) # 中心无人机的横坐标
        if red_rect[0] < yellow_oval[0]:    # 根据左右无人机位置的关系,来进行动作判断
            next_reward = np.array([self.get_reward(self.distance+0.1),self.get_reward(self.distance-0.1),self.get_reward(self.distance)])
        else:
            next_reward = np.array([self.get_reward(self.distance-0.1),self.get_reward(self.distance+0.1),self.get_reward(self.distance)])
        return next_reward


