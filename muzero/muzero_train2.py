import copy
import math
import os
import pickle
import random
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import collections
import typing
from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Input, Activation, Flatten, Conv1D, Add, BatchNormalization, AveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import Progbar

import multiprocessing
from multiprocessing import Process


mode = 'train'
name = 'muzero'
level = 1
if level == 2:
    name += name + 'lv2'

nov_path = './ReinfoceLearningForTrading/data/' + f'sp500_{mode}.csv'
game_dir = './model'
os.makedirs(game_dir, exist_ok=True)

df = pd.read_csv(nov_path)
df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

####### Helpers ##########
MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

# ツリーの最小値を保持するクラス。
class MinMaxStats:
    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
        # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MuZeroConfig:
    def __init__(self,
                action_space_size: int,
                max_moves: int,
                discount: float,
                dirichlet_alpha: float,
                num_simulations: int,
                batch_size: int,
                td_steps: int,
                num_actors: int,
                lr_init: float,
                lr_decay_steps: float,
                visit_softmax_temperature_fn,
                known_bounds: Optional[KnownBounds] = None):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # ルート事前探査ノイズ。
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB式
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # 環境で発生する値に関する情報がすでにある場合は、
        # それらを使用して再スケーリングを初期化できます。
        # これは厳密には必要ありませんが、ボードゲームでAlphaZeroと同じ動作を確立します。
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000)
        self.checkpoint_interval = int(2)
        self.window_size = int(100)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # 指数学習率のスケジュール
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.env = Environment(df, initial_money=1000000, mode='train')
        self.scaler, self.scaler2 = self._standard_scaler(self.env)

    def _standard_scaler(self, env):
        states = []
        rewards = []
        for _ in range(env.df_total_steps):
            action = np.random.choice(env.action_space)
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append([reward])
            if done:
                break

        scaler = StandardScaler()
        scaler2 = StandardScaler()
        scaler.fit(states)
        scaler2.fit(rewards)
        return scaler, scaler2

    def new_game(self):
        return Game(self.action_space_size, self.discount,
                    self.env, self.scaler, self.scaler2)

def make_trade_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 600e3:
            return 0.75
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=3,
        max_moves=10000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=4,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature)

class Action:
    '''
    a = Action(0)
    b = Action(2)

    mydict = {a: "value for 0", b: "value for 2"}
    print(mydict[a], mydict[b]) # value for 0 value for 2
    a.index = 2                     # →ハッシュ値が変わる
    print(mydict[a], mydict[b]) # value for 2 value for 2
    c = Action(3)
    print(a in mydict)
    print(c > a)
    print(c == a)
    '''
    def __init__(self, index: int):
        '''コンストラクタ'''
        self.index = index

    def __hash__(self):
        '''hash呼び出し時に呼び出される'''
        return self.index

    def __eq__(self, other):
        '''同値の時に呼び出される'''
        return self.index == other.index

    def __gt__(self, other):
        '''大小比較された時に呼び出される'''
        return self.index > other.index

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class ActionHistory:
    '''検索内で使用されるシンプルな履歴コンテナ。
    実行されたアクションを追跡するためにのみ使用されます。'''
    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

class Environment:
    def __init__(self, df, initial_money=100000, mode = 'test', commission = 0):

        self.df = df.dropna().reset_index()

        self.df_total_steps  = len(self.df)-1
        self.initial_money   = initial_money
        self.mode            = mode
        self.commission      = commission
        self.trade_time      = None
        self.trade_win       = None
        self.brfore_buy_cash = None
        self.action_space    = np.array([0, 1, 2]) # buy,hold,sell
        self.hold_a_position = None
        self.now_price       = None
        self.cash_in_hand    = None
        self.sell_price      = None
        self.buy_price       = None

        self.reset()

    def reset(self):

        self.trade_time      = 0
        self.trade_win       = 0
        self.brfore_buy_cash = 0
        self.end_step        = self.df_total_steps
        self.now_step        = 0
        self.hold_a_position = 0.0
        self.now_price       = self.df.loc[self.now_step, 'SP500']
        self.cash_in_hand    = self.initial_money
        self.sell_price      = 0
        self.buy_price       = 0

        return self._get_now_state()

    def step(self, action):

        self.now_step += 1
        self.now_price = self.df.loc[self.now_step, 'SP500']

        done = (self.end_step == self.now_step)

        self.sell_price = 0
        self._trade(action,done)
        reward = 0
        if (self.sell_price > 0) and (self.buy_price > 0) and ((self.sell_price - self.buy_price) != 0):
            reward = (self.sell_price - self.buy_price) / self.buy_price
            self.buy_price = 0
        cur_revenue = self._get_revenue()

        info = { 'cur_revenue' : cur_revenue , 'trade_time' : self.trade_time, 'trade_win' : self.trade_win }

        return self._get_now_state(), reward, done, info

    def _get_now_state(self):
        state = np.empty(3)
        state[0] = self.hold_a_position
        state[1] = self.now_price
        state[2] = self.cash_in_hand
        return state

    def _get_revenue(self):
        return self.hold_a_position * self.now_price + self.cash_in_hand

    def _trade(self, action,lastorder = False):
        if lastorder:
            if self.hold_a_position != 0:
                self.cash_in_hand += self.now_price * self.hold_a_position
                self.hold_a_position = 0
                self.trade_time += 1
                if self.cash_in_hand > self.brfore_buy_cash:
                    self.trade_win += 1
        else:
            if self.action_space[0] == action: # buy
                if self.hold_a_position == 0:
                    buy_flag = True
                    self.brfore_buy_cash = copy.copy(self.cash_in_hand)
                    while buy_flag:
                        if self.cash_in_hand > self.now_price:
                            self.hold_a_position += 1
                            self.buy_price += self.now_price
                            self.cash_in_hand -= self.now_price + self.commission * self.now_price
                        else:
                            buy_flag = False
            if self.action_space[2] == action: # sell
                if self.hold_a_position != 0:
                    self.sell_price += self.now_price * self.hold_a_position
                    self.cash_in_hand += self.now_price * self.hold_a_position - self.commission * self.now_price * self.hold_a_position
                    self.hold_a_position = 0
                    self.trade_time += 1
                    if self.cash_in_hand > self.brfore_buy_cash:
                        self.trade_win += 1

# 環境との相互作用の単一のエピソード。
class Game:
    def __init__(self, action_space_size: int,
                 discount: float, env: Environment,
                 scaler: StandardScaler, scaler2: StandardScaler):
        self.env = env  # Game specific environment.
        self.history = []
        self.rewards = []
        self.image = [] # (長さ, 32, 6) 出力用
        self.image_histroy = [] # (長さ, 6) 一時保存保存用
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.terminal_flag = False
        self.info = { 'cur_revenue' : 1.0 , 'trade_time' : 1.0, 'trade_win' : 1.0 }

        self.scaler = scaler
        self.scaler2 = scaler2

        self.act_onehot = [[ 1, 0, 0],  # [0]: buy
                           [ 0, 1, 0],  # [1]: hold
                           [ 0, 0, 1]]  # [2]: sell
        _ = self.env.reset()

    # ゲーム固有の終了ルール。
    def terminal(self) -> bool:
        return self.terminal_flag

    # 法的措置のゲーム固有の計算。
    def legal_actions(self) -> List[Action]:
        act_list = [Action(0), Action(1)]
        if self.env.hold_a_position != 0:
            act_list = [Action(1), Action(2)]
        return act_list

    # 環境を進める。
    def apply(self, action: Action):
        act = action.index
        state, reward, done, info = self.env.step(act)
        if done:
            self.terminal_flag = done
            self.info = info

        if len(self.image_histroy) >= 31:
            reward = self.scaler2.transform([[reward]])
            reward = reward[0][0]
            self.rewards.append(reward)
            self.history.append(action)
        self._make_image_histroy(state, act)

    # image_histroyとimageの生成
    def _make_image_histroy(self, state: list, act: int):
        state = self.scaler.transform([state])
        image = state[0].tolist() + self.act_onehot[act]
        self.image_histroy.append(image)
        if len(self.image_histroy) >= 32:
            self.image.append(self.image_histroy[-32:])

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    # ゲーム固有の特徴平面(obs)
    def make_image(self, state_index: int):
        return [self.image[state_index]]

    # 値ターゲットは、検索ツリーNステップの割引ルート値と、それまでのすべての報酬の割引合計です。
    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        '''
        input: state_index: int, num_unroll_steps: 5, td_steps: 10)
        output: target_value: TD目標価値(z), target_reward: 即時報酬(u), target_policy: MCTSポリシー(pai)
        '''
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, [0.333, 0.334, 0.333]))
        return targets

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g, num_unroll_steps, td_steps))
                    for g in games]
        return [(g.make_image(i), self._action_to_num(g.history[i:i + num_unroll_steps]),
                g.make_target(i, num_unroll_steps, td_steps))
                for (g, i) in game_pos]

    # バッファーから均一または優先度に応じてサンプルゲーム。
    def sample_game(self) -> Game:
        n = random.randrange(len(self.buffer))
        return self.buffer[n]

    # ゲームから均一に、または優先度に応じて位置をサンプリングします。
    def sample_position(self, game, num_unroll_steps: int, td_steps: int) -> int:
        d = num_unroll_steps - td_steps
        n = len(game.image) - (d if d > 0 else -d)
        i = random.randrange(n)
        return i

    def _action_to_num(sellf, action_list: List[Action]) -> List:
        return [a.index for a in action_list]

class RepresentationNetwork:
    def __init__(self):
        '''in:(None, 32, 6), out:(None, 8, 96)'''
        self.obs_shape = (32, 6)
        self.nn_actions = 3
        self.filters = 48
        self.arr = [[8, 5, 3, 1],[8, 5, 3, 1],[8, 5, 3]]
        self.filter = [self.filters, self.filters * 2, self.filters * 2]

        self.kr = l2(0.0005)
        #self.opt = SGD(learning_rate = 0.001, momentum = 0.9)
        self.opt = Adam(learning_rate=0.0001, epsilon=0.001)
        self.units = 64

        self._main_network_layer()

    def _main_network_layer(self):
        x = input = Input(shape = self.obs_shape) # (None, 32, 6)
        for a, f in zip(self.arr, self.filter):
            x = self._residual_layer(a, f)(x)

        # x: (None, 32, 96)
        x = AveragePooling1D(4, padding='same')(x) # (None, 8, 96)

        a, b, c = tf.shape(x) # (None, 8, 96)
        a = 1 if a is None else a

        x_min = tf.fill([a, b, c], tf.reduce_min(x))
        x_max = tf.fill([a, b, c], tf.reduce_max(x))

        hidden_states = (x - x_min) / (x_max - x_min)

        model = Model(inputs = input, outputs= hidden_states)
        model.compile(loss = 'categorical_crossentropy', optimizer = self.opt, metrics=['accuracy'])
        self.model = model


    def _residual_layer(self, arr, filter):
        def f(input_block):
            x = input_block
            for a in arr:
                if a >=  5:
                    x = self._conv_layer(filter, a, True)(x)
                elif a == 3:
                    x = self._conv_layer(filter, a, False)(x)
                else:
                    if len(arr) == 3:
                        input_block = BatchNormalization()(input_block)
                    else:
                        input_block = self._conv_layer(filter,
                                                       a, False)(input_block)

            x = Add()([x, input_block])
            x = Activation('relu')(x)
            return x
        return f

    def _conv_layer(self, filters, kernel_size  = 1, join_act = True):
        def f(input_block):
            x = Conv1D(filters=filters, kernel_size=kernel_size,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(0.0005),
                       kernel_initializer="he_normal")(input_block)
            x = BatchNormalization()(x)
            if join_act:
                x = Activation('relu')(x)
            return x
        return f

class PredictionNetwork:
    '''in:(None, 8, 96), out:(None, 3), (None, 1)'''
    def __init__(self):
        self.hidden_state_shape = (8, 96)
        self.nn_actions = 3
        self.n_supports = 61

        self.kr = l2(0.0005)
        #self.opt = SGD(learning_rate = 0.001, momentum = 0.9)
        self.opt = Adam(learning_rate=0.0001, epsilon=0.001)

        self._main_network_layer()

    def _main_network_layer(self):
        x = input = Input(shape = self.hidden_state_shape)

        p = self._conv_layer(2)(x) # (None, 8, 2)
        p = Flatten()(x) # (None, 16)
        p = Dense(self.nn_actions, kernel_regularizer=self.kr,
                  kernel_initializer="he_normal")(p) # (None, 3)
        p = tf.nn.softmax(p)

        v = self._conv_layer(1)(x) # (None, 8, 1)
        v = Flatten()(x) # (None, 8)

        v = Dense(self.n_supports, kernel_regularizer=self.kr,
                  kernel_initializer="he_normal")(v) # (None, 61)
        v = tf.nn.softmax(v)

        model = Model(inputs = input, outputs= [p, v])
        model.compile(loss = 'categorical_crossentropy', optimizer = self.opt, metrics=['accuracy'])
        self.model = model

    def _conv_layer(self, filters, kernel_size  = 1, join_act = True):
        def f(input_block):
            x = Conv1D(filters=filters, kernel_size=kernel_size,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(0.0005),
                       kernel_initializer="he_normal")(input_block)
            x = BatchNormalization()(x)
            if join_act:
                x = Activation('relu')(x)
            return x
        return f

class DynamicsNetwork:
    '''in:(None, 8, 96), (None, 1), out:(None, 8, 96), (None, 1)'''
    def __init__(self):
        self.hidden_state_shape = (8, 96)
        self.action_shape = (1,)
        self.nn_actions = 3
        self.filters = 96
        self.arr = [[8, 5, 3, 1],[8, 5, 3, 1],[8, 5, 3]]
        self.filter = [self.filters, self.filters, self.filters]

        self.kr = l2(0.0005)
        #self.opt = SGD(learning_rate = 0.001, momentum = 0.9)
        self.opt = Adam(learning_rate=0.0001, epsilon=0.001)

        self._main_network_layer()

    def _main_network_layer(self):
        hs = input_hidden_state = Input(shape = self.hidden_state_shape)
        ia = input_action = Input(shape = self.action_shape)

        a, b, c = tf.shape(hs) # (None, 8, 96)
        a = 1 if a is None else a

        actions_onehot = tf.transpose(tf.reshape(tf.repeat(
            tf.one_hot(tf.cast(ia, dtype='int32'), self.nn_actions),
            repeats = b, axis=1), (a, self.nn_actions, b)), perm=[0, 2, 1])
        x = tf.concat([hs, actions_onehot], axis=2) #: (1, 8, 96 + 3)

        for a, f in zip(self.arr, self.filter):
            x = self._residual_layer(a, f)(x)

        a, b, c = tf.shape(hs) # (None, 8, 99)
        a = 1 if a is None else a

        x_min = tf.fill([a, b, c], tf.reduce_min(x))
        x_max = tf.fill([a, b, c], tf.reduce_max(x))

        hidden_states = (x - x_min) / (x_max - x_min)

        x = Flatten()(x)
        x = Dense(61, kernel_regularizer=l2(0.0005),
                  kernel_initializer="he_normal")(x)
        categorical_rewards = tf.nn.softmax(x)

        model = Model(inputs = [input_hidden_state, input_action],
                      outputs= [hidden_states, categorical_rewards])
        model.compile(loss = 'categorical_crossentropy', optimizer = self.opt,
                      metrics=['accuracy'])
        self.model = model

    def _residual_layer(self, arr, filter):
        def f(input_block):
            x = input_block
            for a in arr:
                if a >=  5:
                    x = self._conv_layer(filter, a, True)(x)
                elif a == 3:
                    x = self._conv_layer(filter, a, False)(x)
                else:
                    if len(arr) == 3:
                        input_block = BatchNormalization()(input_block)
                    else:
                        input_block = self._conv_layer(filter,
                                                       a, False)(input_block)

            x = Add()([x, input_block])
            x = Activation('relu')(x)
            return x
        return f

    def _conv_layer(self, filters, kernel_size  = 1, join_act = True):
        def f(input_block):
            x = Conv1D(filters=filters, kernel_size=kernel_size,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(0.0005),
                       kernel_initializer="he_normal")(input_block)
            x = BatchNormalization()(x)
            if join_act:
                x = Activation('relu')(x)
            return x
        return f

class NetworkOutput(typing.NamedTuple):
    '''
    データ送信用クラス
    typing.NamedTuple: 型がついているcollection.namedtupleを定義できる
    a = Action(2)
    net = NetworkOutput(4, 1.2, {a: 3.3}, [123.0, 555.6, 76.0])
    print(net.value, net.reward, net.policy_logits[a], net.hidden_state)
    (4, 1.2, 3.3, [123.0, 555.6, 76.0])
    '''
    value: float
    reward: float
    policy_logits: Dict[Action, float] # {Action(0): 0.1, Action(1): 0.5, Action(2): 0.4}
    hidden_state: List[float]

class Network:
    def __init__(self, actor_num = 99):
        self.rnet = RepresentationNetwork()
        self.pnet = PredictionNetwork()
        self.dnet = DynamicsNetwork()
        self.rnet_model = self.rnet.model
        self.pnet_model = self.pnet.model
        self.dnet_model = self.dnet.model
        self.rnet_name = 'muzero_RepresentationNetwork'
        self.pnet_name = 'muzero_PredictionNetwork'
        self.dnet_name = 'muzero_DynamicsNetwork'
        self.training_steps_num = int(1)

        self._load_network()
        if actor_num == 0:
            self.training_steps_num = int(10e5) # 0.25
        elif actor_num == 1:
            self.training_steps_num = int(700e3) # 0.5
        elif actor_num == 2:
            self.training_steps_num = int(550e3) # 0.75

    # representation + prediction function
    def initial_inference(self, image) -> NetworkOutput:
        '''何もしてないので、報酬報酬0に設定。
        image: (32, 6), hidden_state: (1, 8, 96)
        policy_logits: (1, 3), value: (1, 1)'''
        hidden_state = self.rnet_model(tf.convert_to_tensor(image))
        policy_logits, value = self.pnet_model(hidden_state)
        dic = {}
        for i, l in enumerate(policy_logits[0]):
            dic[Action(i)] = l
        return NetworkOutput(self._rescaling_inverse(value[0][0].numpy()), 0.0, dic, hidden_state)

    # dynamics + prediction function
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        '''hidden_state: (1, 8, 96), action:(1, )
        categorical_rewards: (1, 61)
        hidden_state:  (1, 8, 96), reward:  (1, 61)'''
        act = tf.convert_to_tensor([action.index])
        hidden_state, categorical_rewards = self.dnet_model([hidden_state, act])
        policy_logits, categorical_values = self.pnet_model(hidden_state)

        supports = tf.range(-30, 31, dtype=tf.float32)
        reward = tf.reduce_sum(supports * categorical_rewards, axis=1, keepdims=True)
        value = tf.reduce_sum(supports * categorical_values, axis=1, keepdims=True)
        dic = {}
        for i, l in enumerate(policy_logits[0]):
            dic[Action(i)] = l
        return NetworkOutput(self._rescaling_inverse(value[0][0].numpy()), reward[0][0].numpy(), dic, hidden_state)

    # このネットワークの重みを返します。
    def get_weights(self):
        return [self.rnet_model.get_weights(),
                self.pnet_model.get_weights(),
                self.dnet_model.get_weights()]

    # ネットワークが訓練されたステップ/バッチの数。
    def training_steps(self) -> int:
        '''
        training_steps < 500e3 -> 1.0
        training_steps < 600e3 -> 0.75
        training_steps < 750e3 -> 0.5
        over                   -> 0.25
        '''
        return self.training_steps_num

    def _rescaling_inverse(self, x):
        eps = 0.001
        if x > 0:
            return ((2*eps*x+2*eps+1-
                        (4*eps*(eps+1+x)+1)**0.5)/(2*eps**2))
        else:
            return ((-2*eps*x+2*eps+1-
                        (4*eps*(eps+1-x)+1)**0.5)/(2*eps**2)*(-1))

    def save_network(self):
        self.rnet_model.save_weights(f'{game_dir}/{self.rnet_name}.h5')
        self.pnet_model.save_weights(f'{game_dir}/{self.pnet_name}.h5')
        self.dnet_model.save_weights(f'{game_dir}/{self.dnet_name}.h5')

    def _load_network(self):
        if os.path.isfile(f'{game_dir}/{self.rnet_name}.h5'):
            self.rnet_model.load_weights(f'{game_dir}/{self.rnet_name}.h5')
        if os.path.isfile(f'{game_dir}/{self.pnet_name}.h5'):
            self.pnet_model.load_weights(f'{game_dir}/{self.pnet_name}.h5')
        if os.path.isfile(f'{game_dir}/{self.dnet_name}.h5'):
            self.dnet_model.load_weights(f'{game_dir}/{self.dnet_name}.h5')

class SharedStorage:
    def __init__(self):
        self._networks = {}

    def latest_network(self, actor_num = 99) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self._make_uniform_network(actor_num)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network

    def _make_uniform_network(self, actor_num = 99):
        return Network(actor_num)

####### Part 1: Self-Play ########
# 各セルフプレイジョブは他のすべてのジョブとは独立しています。
# 最新のネットワークスナップショットを取得し、ゲームを作成し、共有リプレイバッファに書き込むことでトレーニングジョブで利用できるようにします。
def run_selfplay(config: MuZeroConfig, storage: SharedStorage, actor_num: int=99):

    network = storage.latest_network(actor_num)
    game = play_game(config, network)
    print(game.info)
    return game

# 各ゲームは、最初のボードの位置から開始し、ゲームの終了に達するまで動きを生成するためにモンテカルロツリー検索を繰り返し実行することによって生成されます。
def play_game(config: MuZeroConfig, network: Network) -> Game:

    game = config.new_game()
    i = 0

    while not game.terminal():
        # 検索ツリーのルートでは、表現関数を使用して、現在の観測値を指定して隠し状態を取得します。
        if i >= 32:
            root = Node(0)
            current_observation = game.make_image(-1) # 最新のimageを取得
            expand_node(root, game.legal_actions(),
                        network.initial_inference(current_observation))
            add_exploration_noise(config, root)
            # 次に、アクションシーケンスとネットワークによって学習されたモデルのみを使用してモンテカルロツリー検索を実行します。
            run_mcts(config, root, game.action_history(), network)
            action = select_action(config, len(game.history), root, network)
            game.apply(action)
            game.store_search_statistics(root)
        else:
            game.apply(Action(1))
        i += 1
    return game

# コアモンテカルロツリー検索アルゴリズム。
# アクションを決定するために、Nシミュレーションを実行し、常に検索ツリーのルートから始まり、リーフノードに到達するまでUCB式に従ってツリーを横断します。
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # 検索ツリー内では、ダイナミクス関数を使用して、アクションと前の非表示状態を指定して次の非表示状態を取得します。
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                    history.last_action())
        expand_node(node, history.action_space(), network_output)
        backpropagate(search_path, network_output.value, config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())

    _, action = softmax_sample(visit_counts, t)
    return action

def softmax_sample(distribution, temperature: float):
    '''
    [(30, <__main__.Action object at 0x7fd9fcd5b650>),
     (20, <__main__.Action object at 0x7fd9fcd5b150>)]
    '''
    p, n = np.array([]), np.array([], dtype=int)
    for i, ll in enumerate(distribution):
        p = np.append(p, ll[0])
        n = np.append(n, i)
    f = np.exp(p/temperature)/np.sum(np.exp(p/temperature))
    a = np.random.choice(a=n, p=f)
    return 0, distribution[a][1]

# UCBスコアが最も高い子を選択します。
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
        child) for action, child in node.children.items())
    return action, child


# ノードのスコアは、その値と以前の探査ボーナスに基づいています。
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# ニューラルネットワークから得られた値、報酬、ポリシー予測を使用してノードを拡張します。
def expand_node(node: Node, actions: List[Action],
                network_output: NetworkOutput):
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# シミュレーションの最後に、ツリーからルートまで評価を伝播します。
def backpropagate(search_path: List[Node], value: float,
                  discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value

# 各検索の開始時に、ルートの前にディリクレノイズを追加し、検索が新しいアクションを探索することを奨励します。
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

######### End Self-Play ##########
####### Part 2: Training #########
def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network(True)
    pb = Progbar(config.training_steps)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(network, batch, config)
        pb.add(1)
    storage.save_network(config.training_steps, network)

def update_weights(network: Network, batch, config: MuZeroConfig):
    # Initial step, from the real observation.
    '''
    batch:(1024, 3)
    target_value, target_reward, target_policy = target
    images: (1024, 32, 6)
    actions: (1024, 5)
    target_value: (1024, 6)
    target_reward: (1024, 6)
    target_policy: (1024, 6, 3)
    config.batch_size=1024
    '''

    images, actions, targets = list(zip(*batch))
    images = [i[0] for i in images]
    images = tf.convert_to_tensor(images) # (1024, 32, 6)
    actions = tf.convert_to_tensor(actions) # (1024, 5)

    target_values, target_rewards, target_policys = [],[],[]
    for target in targets:
        mini_value, mini_reward, mini_policy = [],[],[]
        for value, reward, policy in target:
            mini_value.append(value)
            mini_reward.append(reward)
            mini_policy.append(policy)
        target_values.append(mini_value)
        target_rewards.append(mini_reward)
        target_policys.append(mini_policy)

    target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
    target_rewards = tf.convert_to_tensor(target_rewards, dtype=tf.float32)
    target_policys = tf.convert_to_tensor(target_policys)

    loss = 0.
    policy_loss, value_loss, reward_loss = 0., 0., 0.

    with tf.GradientTape() as tape:
        hidden_states = network.rnet_model(images, training=True)
        policy_logits, categorical_values = network.pnet_model(hidden_states, training=True)

        reward = tf.zeros([config.batch_size, 1],dtype=tf.dtypes.float32)
        value  = make_categorical_to_one(categorical_values)

        l_policy = scalar_loss(policy_logits, target_policys[:,0,:])
        v_policy = scalar_loss(value, reshape_to_one(config, target_values[:,0]))
        r_policy = scalar_loss(reward, reshape_to_one(config, target_rewards[:,0]))

        scale = 1.0

        policy_loss += scale_gradient(l_policy, scale)
        value_loss += scale_gradient(v_policy, scale)
        reward_loss += scale_gradient(r_policy, scale)

        hidden_states = scale_gradient(hidden_states, 0.5) # (1024, 8, 96), dtype=float32)

        '''
        hidden_states:  (1024, 8, 96)
        policy_logits:  (1024, 3)
        values:  (1024, 1)
        actions[:, 0], shape=(1024,), dtype=int32)
        categorical_rewards: (1024, 61)
        reward: (1024, 1)
        '''

        scale = 1.0 / config.num_unroll_steps
        for i in range(config.num_unroll_steps):
            hidden_states, categorical_rewards = network.dnet_model([hidden_states,
                                                                     reshape_to_one(config, actions[:, i])], training=True)
            policy_logits, categorical_values = network.pnet_model(hidden_states, training=True)

            reward = make_categorical_to_one(categorical_rewards)
            value  = make_categorical_to_one(categorical_values)

            l_policy = scalar_loss(policy_logits, target_policys[:,i+1,:])
            v_policy = scalar_loss(value, reshape_to_one(config, target_values[:,i+1]))
            r_policy = scalar_loss(reward, reshape_to_one(config, target_rewards[:,i+1]))

            policy_loss += scale_gradient(l_policy, scale)
            value_loss += scale_gradient(v_policy, scale)
            reward_loss += scale_gradient(r_policy, scale)

            hidden_states = scale_gradient(hidden_states, 0.5) # (1024, 8, 96), dtype=float32)

        policy_loss_sum = tf.reduce_mean(policy_loss)
        value_loss_sum = tf.reduce_mean(value_loss)
        reward_loss_sum = tf.reduce_mean(reward_loss)

        loss = policy_loss_sum + value_loss_sum + reward_loss_sum

    #: Gather trainable variables
    models = [network.rnet_model, network.pnet_model]
    variables = [m.trainable_variables for m in models]

    grads = tape.gradient(loss, variables)
    for v, g, m in zip(variables, grads, models):
        tmp_grads, _ = tf.clip_by_global_norm(g, 40.0)
        m.optimizer.apply_gradients(zip(tmp_grads, v))
        #m.optimizer.apply_gradients((grad, var) for (grad, var) in
        #                            zip(tmp_grads, v) if grad is not None)

# ボードゲームのMSE、アタリのカテゴリ値間のクロスエントロピー。
def scalar_loss(prediction, target) -> tf.Tensor:
    return categorical_crossentropy(target, prediction)

def scale_gradient(tensor: tf.Tensor, scale: float) -> tf.Tensor:
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

def reshape_to_one(config: MuZeroConfig, tensor: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tensor, (config.batch_size, 1))

def make_categorical_to_one(categorical_tensor: tf.Tensor) -> tf.Tensor:
    supports = tf.range(-30, 31, dtype=tf.float32)
    one_tensor = tf.reduce_sum(supports * categorical_tensor, axis=1, keepdims=True)
    return one_tensor

def rescaling(x):
    eps = 0.001
    if x == 0:
        return 0
    n = math.sqrt(abs(x)+1) - 1
    return (tf.math.sign(x)*n + eps*x)

def rescaling_inverse(x):
    eps = 0.001
    if x > 0:
        return ((2*eps*x+2*eps+1-
                    (4*eps*(eps+1+x)+1)**0.5)/(2*eps**2))
    else:
        return ((-2*eps*x+2*eps+1-
                    (4*eps*(eps+1-x)+1)**0.5)/(2*eps**2)*(-1))
######### End Training ###########
############################# End of pseudocode ################################

# MuZeroトレーニングは、ネットワークトレーニングとセルフプレイデータ生成の2つの独立した部分に分かれています。
# これら2つの部分は、最新のネットワークチェックポイントをトレーニングからセルフプレイに転送し、
# 完成したゲームをセルフプレイからトレーニングに転送することによってのみ通信します。
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    lock = multiprocessing.Lock()

    worker = []
    for i in range(config.num_actors):
        '''config.num_actors: 4'''
        p = Process(target=launch_job,
                    args=(run_selfplay, config, storage, lock, i))
        worker.append(p)
        p.start()
    for w in worker:
        w.join()

    file_name = game_dir + '/muzero_game.pkl'
    replay_buffer = load(file_name)
    train_network(config, storage, replay_buffer)

    return storage.latest_network()

def launch_job(f, config: MuZeroConfig, storage: SharedStorage, lock, actor_num: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    file_name = game_dir + '/muzero_game.pkl'
    actor_num = 99 # t = 1.0
    # actor_num = 0 # t = 0.25 test環境用

    # for i in range(5):
    for i in range(25):
        start_time = datetime.now()
        game = f(config, storage, actor_num)
        lock.acquire()

        if os.path.isfile(file_name):
            replay_buffer = load(file_name)
            replay_buffer.save_game(game)
        else:
            replay_buffer = ReplayBuffer(config)
            replay_buffer.save_game(game)

        save(file_name, replay_buffer)
        buffer_num = len(replay_buffer.buffer)
        del replay_buffer
        lock.release()
        end_time = datetime.now() - start_time
        print(f'actor_num: No.{str(actor_num + 1)} time: {end_time} game nums: {str(buffer_num)} pieces')


def load(file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save(file_name: str, replay_buffer: ReplayBuffer):
    with open(file_name, 'wb') as f:
        pickle.dump(replay_buffer, f)


net = muzero(make_trade_config())
net.save_network()
