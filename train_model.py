# coding=utf-8
import argparse, os
import tensorflow as tf
from PIL import Image
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent                  #这个是keras 的rl库  pip install keras-rl
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from ICMDP_Env import ClassifyEnv
from get_model import get_text_model, get_image_model
from data_pre import load_data, get_imb_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from read_my_data import hanshu

#下面这个 add_argument()函数主要用于读入命令行参数    设置默认值
parser = argparse.ArgumentParser()
#parser.add_argument('--data',choices=['mnist', 'cifar10','famnist','imdb'], default='famnist')
parser.add_argument('--data',choices=['mnist', 'cifar10','famnist','imdb','mydata'], default='famnist')
parser.add_argument('--model', choices=['image', 'text'], default='image')
parser.add_argument('--imb-rate',type=float, default=0.05)                  #----------------------------样本不平衡率  要修改的
# parser.add_argument('--min-class', type=str, default='456')                #         选了 456
# parser.add_argument('--maj-class', type=str, default='789')                #          选了 789
parser.add_argument('--min-class', type=str, default='0')                #          选了 0
parser.add_argument('--maj-class', type=str, default='1234')                #          选了 1234
parser.add_argument('--training-steps', type=int, default=120000)           #训练次数
args = parser.parse_args()                                                #
data_name = args.data                                                    #



# x_train, y_train, x_test, y_test = load_data(data_name)
x_train, y_train, x_test, y_test = load_data('mydata')

# imb_rate = args.imb_rate        #imb_rate 默认0.05------------------------------------I
imb_rate = 0.25
maj_class = list(map(int, list(args.maj_class)))     #map(int)  将后面的列表中的元素变成int型
min_class = list(map(int, list(args.min_class)))

x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)
print(x_train.shape, y_train.shape)
in_shape = x_train.shape[1:]
num_classes = len(set(y_test))           #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
                                        # 这里主要是给下面的nb_actions赋值   表明又多少种动作
mode = 'train'
env = ClassifyEnv(mode, imb_rate, x_train, y_train)
nb_actions = num_classes
training_steps = args.training_steps          #1.2w步
if args.model == 'image':
    model = get_image_model(in_shape, num_classes)
else:
    in_shape = [5000, 500]
    model = get_text_model(in_shape, num_classes)

INPUT_SHAPE = in_shape
print(model.summary())


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        if args.model == 'text':
            return observation
        img = observation.reshape(INPUT_SHAPE)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        if args.model == 'text':
            return batch.reshape((-1, INPUT_SHAPE[1]))
        batch = batch.reshape((-1,) + INPUT_SHAPE)
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


memory = SequentialMemory(limit=100000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
#值从1 开始并且不小于0.1 同时测试我们的随机数是否小于 0.05。我们将步数设置为 1 到 10，000 之间的步数，Keras-RL 为我们处理衰减数学运算。

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=0.5, target_model_update=10000,
               train_interval=4, delta_clip=1.)
#nb_steps_warmup 梯度变化的步数  太少了不行     target_model_update  是目标网络更新频率
dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, nb_steps=training_steps, log_interval=60000)


env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)
env = ClassifyEnv(mode, imb_rate, x_test, y_test)
env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)

