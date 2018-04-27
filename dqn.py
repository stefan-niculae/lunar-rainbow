import random
from collections import deque
import numpy as np
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, AlphaDropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
from base_agent import Agent
# from dataclasses import dataclass


# @dataclass  # not available for python <3.6
# class Transition:
#     state:      np.array
#     action:     int
#     reward:     float
#     next_state: np.array
#     done:       bool
#     error:      float

class DQN(Agent):
    """ TODO: add description for this class
    Simple DQN agent
    """

    """
    OpenAI Lab claim:
    luckiest run: solved in 36min, 34s (359 episodes)
    
    state: concatenated previous and current state
    pre-processing: none
    feature selection: show all

    - discount 0.99
    
    - epsilon-greedy policy
    - initial epsilon: 1
    - final epsilon: 0.1
    - exploration_anneal_episodes: 150
    
    - adam optimizer (params other than lr default)
    - loss: mse
    - lr: 0.005
    - lr decay: divide by 10 every 200 episodes
    
    - layer 0: fc 400 sigmoid
    - layer 1: fc 200 sigmoid
    - out layer: fc 4 (actions) linear
    - initialization: lecun_uniform
    - train epochs: 1
    - train every: 5 time-steps (?)

    - clip Q values: [-10,000; +10,000]

    - un-prioritized memory
    - batch size: 32
    - memory capacity: 50,000
    
    """

    batch_size = 32
    n_epochs = 1
    memory_size = 50000

    q_clip = (-10000, +10000)

    normalize = False

    def __init__(self, env, seed=42,
                 lr=0.005, discount=.99, layer_sizes=(384, 192),
                 exploration_start=1, exploration_min=.01, exploration_anneal_steps=150):
        super().__init__(env, seed=seed)
        self.discount = discount
        self.lr = lr

        # build model
        self.layer_sizes = layer_sizes
        self._model = self._build_model()

        # setup exploration
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self._exploration_eps = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps

        # replay
        self._memory = deque(maxlen=self.memory_size)  # will hold `Transitions`

    def _build_model(self) -> Model:
        model = Sequential()

        model.add(Dense(self.layer_sizes[0], activation='sigmoid', input_shape=(self._state_size,)))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='sigmoid'))

        model.add(Dense(self._n_actions, activation='linear'))
        model.compile(optimizer=Adam(lr=self.lr),
                      loss='mse')
        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        state = np.expand_dims(state, axis=0)
        q = self._model.predict(state)[0]

        # Explore
        if random.random() < self._exploration_eps and not eval_mode:
            return random.randint(0, self._n_actions - 1)

        # Exploit
        else:
            return q.argmax()

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        # remember
        self._memory.append((state, action, reward, next_state, done))
        self._replay()

    def train(self):
        if True or self._episode % 5 == 0:
            super().train()
        else:
            self._episode += 1
        # Perform episode end updates
        self._exploration_eps -= self._exploration_drop
        self._exploration_eps = max(self._exploration_eps, self.exploration_min)
        if self._episode % 200 == 0:
            self.lr /= 10
            K.set_value(self._model.optimizer.lr, self.lr)
        # print('eps', self._exploration_eps, 'lr', self.lr, end=' ')

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        if len(self._memory) < self.batch_size:
            # not enough experience to learn from
            return

        sample_indices = np.random.choice(len(self._memory), self.batch_size)
        sample = [self._memory[i] for i in sample_indices]
        state, action, reward, next_state, done = (np.array(l) for l in zip(*sample))

        q = self._model.predict(state)
        q = np.clip(q, *self.q_clip)
        next_q = self._model.predict(next_state)
        next_q = np.clip(next_q, *self.q_clip)
        next_max = next_q.max(axis=1)

        target = reward + (1-done) * (self.discount * next_max)  # add future discount only if not done
        q[np.arange(len(action)), action] = target

        hist = self._model.fit(state, q, epochs=self.n_epochs, verbose=0)
        # print('fit res', hist)

    def save(self, path: str):
        self._model.save(path + '-model.h5')


class Memory:
    def __init__(self, capacity: int, multi_step: int, state_size: int):
        self.capacity = capacity
        self.multi_step = multi_step
        self.size = 0
        self.cursor = 0
        self.states     = np.zeros((self.capacity, state_size), dtype=float)
        self.actions    = np.zeros(self.capacity, dtype=int)
        self.rewards    = np.zeros(self.capacity, dtype=float)
        self.terminals  = np.zeros(self.capacity, dtype=bool)
        self.priorities = np.zeros(self.capacity, dtype=float)
        self._max_priority = 0

    def __len__(self):
        return self.size

    def add(self, state: [float], action: int, reward: float, terminal: bool):
        self.cursor %= self.capacity

        self.states[self.cursor] = state
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.terminals[self.cursor] = terminal
        self.priorities[self.cursor] = self._max_priority

        self.cursor += 1
        self.size += 1
        self.size = min(self.size, self.capacity)

    def update_priorities(self, indices: [int], values: [float]):
        self.priorities[indices] = values
        self._max_priority = max(self._max_priority, max(values))

    def sample(self, n_samples: int, shift: float, exp: float, discount: float) -> tuple:
        p = (self.priorities + shift) ** exp
        if self.size < self.capacity:
            p = p[:self.size]
        indices = np.random.choice(self.size, size=n_samples, p=p/sum(p))
        # can't be too close to the replacing border, need ensure future transitions are still in memory
        # if c is the cursor (to be replaced next); and multi-step is n (eg 3); then
        # c   is not good (need c+1 and c+2)
        # c-1 is not good (need c   and c+1)
        # c-2 is good (need c-1 and c)
        # so it is safe before c-(n-1)
        too_close = (indices > self.cursor - self.multi_step + 1) & (indices <= self.cursor)
        indices[too_close] -= self.multi_step - 1
        next_indices = (indices + 1) % self.size

        returns = np.zeros(n_samples)
        for sampled_idx in indices:
            for step in range(self.multi_step):
                current_idx = (sampled_idx + step) % self.size
                if self.terminals[current_idx]:  # don't consider further rewards one the episode is marked done
                    break
                returns += discount ** step * self.rewards[current_idx]

        return self.states[indices], self.actions[indices], np.array(returns), self.states[next_indices], self.terminals[indices], indices


class DQNP(Agent):
    """ TODO: add description for this class
    DQN agent with:
     - prioritized experience replay
     - dropout as a stand-in for bnn
     - max-boltzmann exploration: pick random actions weighted by estimated values
     - double: take predictions from target network slowly updating toward main
     - idealization (my contrib)
     - batch normalization
     - huber loss (approximated by logcosh)
     - dueling: split into two streams: state value and action advantage q=v+a-avg(a)
     - multi-step
    """

    batch_size = 32
    memory_size = 5000
    fc_kwargs = dict(kernel_initializer='lecun_normal', activation='selu')
    n_epochs = 2
    q_clip = (-50, 50)

    normalize = True

    def __init__(self, env, seed=42,
                 lr_init=.1, lr_decay=.9999, lr_min=.001,
                 discount=.99, target_update_freq=1000, idealization=.9, n_steps=3,  # rl algorithm
                 layer_sizes=(128, 64), streams_size=32, input_dropout=.2, hidden_dropout=.4,  # model config
                 per_alpha=.5, per_eps=1,  # replay
                 exploration_start=1, exploration_min=.05, exploration_anneal_steps=45000, boltzmann_tau=1,  # policy exploration
                 ):
        super().__init__(env, seed=seed)
        self.n_steps = n_steps
        self.discount = discount
        self.target_update_freq = target_update_freq
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.idealization = idealization
        self._lr = lr_init

        # build model
        self.layer_sizes = layer_sizes
        self.streams_size = streams_size
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self._model = self._build_model()
        self._target_model = self._build_model()
        self._target_model.set_weights(self._model.get_weights())

        # setup exploration
        self.boltzmann_tau = boltzmann_tau
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self._exploration_eps = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps

        # replay
        self.per_alpha = per_alpha
        self.per_eps = per_eps
        self._memory = Memory(capacity=self.memory_size, state_size=self._state_size, multi_step=n_steps)

    def _build_model(self) -> Model:
        inp = layers.Input(shape=(self._state_size,))

        # multiple fully connected layers at the beginning
        fcs = AlphaDropout(self.input_dropout)(inp)
        for size in self.layer_sizes:
            fcs = Dense(size, **self.fc_kwargs)(fcs)
            fcs = BatchNormalization()(fcs)
            fcs = AlphaDropout(self.hidden_dropout)(fcs)

        # action advantage stream
        a = Dense(self.streams_size, **self.fc_kwargs)(fcs)
        a = Dense(self._n_actions,   **self.fc_kwargs)(a)
        baseline = layers.Lambda(lambda adv: K.mean(adv, axis=1, keepdims=True))(a)
        a = layers.subtract([a, baseline])

        # state value stream
        v = Dense(self.streams_size, **self.fc_kwargs)(fcs)
        v = Dense(1, **self.fc_kwargs)(v)

        q = layers.add([v, a])  # combine back streams

        model = Model(inp, q)
        model.compile(optimizer=Adam(lr=self._lr),
                      loss='logcosh')  # similar properties to huber
        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        state = np.expand_dims(state, axis=0)
        q = self._model.predict(state)[0]

        # Explore
        if random.random() < self._exploration_eps and not eval_mode:
            p = np.exp(np.clip(q / self.boltzmann_tau, *self.q_clip))
            return np.random.choice(self._n_actions, p=p/sum(p))

        # Exploit
        else:
            return q.argmax()
            # indices_of_max = np.where(q == q.max())[0]
            # return np.random.choice(indices_of_max)

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        # remember
        self._memory.add(state, action, reward, done)
        self._replay()

    def train(self):
        super().train()
        # Perform episode end updates
        self._exploration_eps -= self._exploration_drop
        self._exploration_eps = max(self._exploration_eps, self.exploration_min)
        self._lr = max(self._lr * self.lr_decay, self.lr_min)
        K.set_value(self._model.optimizer.lr, self._lr)
        if self._episode % self.target_update_freq == 0:
            self._update_target_weights()

    def _update_target_weights(self):
        # new_target_weights = [
        #     (1 - self.target_rl) * target_w + self.target_rl * model_w
        #     for model_w, target_w in zip(self._model.get_weights(), self._target_model.get_weights())
        # ]
        self._target_model.set_weights(self._model.get_weights())

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        if len(self._memory) < self.batch_size:
            # not enough experience to learn from
            return

        state, action, ret, next_state, done, memory_indices = self._memory.sample(
            self.batch_size, shift=self.per_eps, exp=self.per_alpha, discount=self.discount)

        q = self._model.predict(state)
        next_q = self._target_model.predict(next_state)
        next_avg = next_q.mean(axis=1)
        next_max = next_q.max(axis=1)
        next_agg = next_avg + (next_max - next_avg) * self.idealization

        target = ret + (1-done) * (self.discount * next_agg)  # add discounted future discount only if not done
        actions_slice = np.arange(len(action)), action
        q[actions_slice] = target

        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)

        predictions = self._model.predict(state)
        errors = abs(predictions[actions_slice] - target)
        self._memory.update_priorities(memory_indices, errors)

    def save(self, path: str):
        self._model.save(path + '-model.h5')
