import random
from collections import deque
from copy import copy
import numpy as np
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, AlphaDropout, Dropout
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
    normalize = False

    def __init__(self, env, seed=24,
                 lr=0.005, discount=.99, exploration=.25,
                 layer_sizes=(384, 192),
                 loss='mse', hidden_activation='sigmoid', out_activation='linear',
                 batch_size=32, n_epochs=1, memory_size=5000,
                 ):
        super().__init__(env, seed=seed)
        self.discount = discount
        self.lr = lr

        # build model
        self.layer_sizes = layer_sizes
        self.loss = loss
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self._model = self._build_model()

        # setup exploration
        self.exploration = exploration

        # replay
        self.memory_size = memory_size
        self._memory = deque(maxlen=self.memory_size)  # will hold `Transitions`

    @property
    def config(self) -> dict:
        c = super().config
        c.update({attr: getattr(self, attr) for attr in [
            'lr',
            'discount',
            'exploration',
            'layer_sizes',
            'hidden_activation',
            'out_activation',
            'loss',
            'batch_size',
            'n_epochs',
            'memory_size',
            'normalize'
        ]})
        return c

    def _build_model(self) -> Model:
        model = Sequential()

        model.add(Dense(self.layer_sizes[0], activation=self.hidden_activation, input_shape=(self._state_size,)))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation=self.hidden_activation))

        model.add(Dense(self._n_actions, activation=self.out_activation))
        model.compile(optimizer=Adam(lr=self.lr), loss=self.loss)
        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        state = np.expand_dims(state, axis=0)
        q = self._model.predict(state)[0]

        # Explore
        if random.random() < self.exploration and not eval_mode:
            return np.random.choice(self._n_actions)

        # Exploit
        else:
            return q.argmax()

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        # remember
        self._memory.append((state, action, reward, next_state, done))

        # after enough enough experience
        if len(self._memory) >= self.batch_size:
            self._replay()

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        sample_indices = np.random.choice(len(self._memory), self.batch_size)
        sample = [self._memory[i] for i in sample_indices]
        state, action, reward, next_state, done = (np.array(l) for l in zip(*sample))

        q = self._model.predict(state)
        next_q = self._model.predict(next_state)
        next_max = next_q.max(axis=1)

        target = reward + (1-done) * (self.discount * next_max)  # add future discount only if not done
        q[np.arange(len(action)), action] = target

        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)

    def save(self, path: str):
        self._model.save(path + '.h5')


class Memory:
    def __init__(self, capacity: int, state_size: int, multi_step: int, history_len: int, batch_size: int, epsilon: float, prioritize: bool):
        self.capacity = capacity
        self.multi_step = multi_step
        self.history_len = history_len
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.prioritize = prioritize

        self.size = 0
        self.cursor = 0

        self.states     = np.zeros((self.capacity, state_size), dtype=float)
        self.actions    = np.zeros(self.capacity, dtype=int)
        self.rewards    = np.zeros(self.capacity, dtype=float)
        self.terminals  = np.zeros(self.capacity, dtype=bool)
        self.priorities = np.zeros(self.capacity, dtype=float)
        self._max_priority = 0.

    def __len__(self):
        return self.size

    def add(self, state: [float], action: int, reward: float, terminal: bool):
        # print('add', state)
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

    def sample(self, exp: float, discount: float) -> tuple:
        if self.prioritize:
            p = (self.priorities + self.epsilon) ** exp
            if self.size < self.capacity:  # sample just from filled so far
                p = p[:self.size]
        else:
            p = np.ones(self.size)
        # don't select items too close to the erasing border
        # nor to the right, so we can look back at still present history
        # nor to the left, so we can look in to already recorded future
        p[self.cursor-self.history_len-2 : self.cursor+self.multi_step+2] = 0
        indices = np.random.choice(self.size, size=self.batch_size, p=p/sum(p))  # DEBUG

        history_indices = indices + np.expand_dims(range(self.history_len), axis=1)
        history_indices %= self.size
        s = np.concatenate(self.states[history_indices], axis=1)

        history_indices += 1
        history_indices %= self.size
        ns = np.concatenate(self.states[history_indices], axis=1)

        returns = np.zeros(self.batch_size)
        for sample_number, sampled_idx in enumerate(indices):
            for step in range(self.multi_step):
                current_idx = (sampled_idx + step) % self.size
                if self.terminals[current_idx]:  # don't consider further rewards one the episode is marked done
                    break
                returns[sample_number] += discount ** step * self.rewards[current_idx]

        return s, self.actions[indices], returns, ns, self.terminals[indices], indices


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

    normalize = False  # TODO more elegantly

    # TODO try logcosh loss, n_steps, idealization, per, selu, alpha dropout, he_uniform
    def __init__(
        self, env, seed=24,
        lr_init=.005, lr_decay=.25, lr_min=.0001, decay_freq=100,
        discount=.99, idealization=1,
        multi_steps=1, history_len=2,
        layer_sizes=(384, 192), input_dropout=0, hidden_dropout=0, batch_normalization=False,
        loss='mse', hidden_activation='sigmoid', out_activation='linear', weights_init='lecun_uniform',
        double=True, target_update_freq=25,
        dueling=True, streams_size=0,
        prioritize_replay=True, priority_exp=.01, priority_shift=.1,
        policy='max-boltzmann', exploration_start=1, exploration_min=.05, exploration_anneal_steps=150,
        exploration_temp=2, exploration_temp_min=.2,
        batch_size=32, n_epochs=1, memory_size=50000,
        q_clip=(-10000, +10000), exploration_q_clip=(-1000, 1000)
    ):
        """
        :param env:
        :param seed:
        :param lr_init:
        :param lr_decay:
        :param lr_min:
        :param discount:
        :param target_update_freq:
        :param idealization:
        :param multi_steps:
        :param loss:
        :param hidden_activation:
        :param out_activation:
        :param weights_init:
        :param double:
        :param dueling:
        :param layer_sizes:
        :param streams_size:
        :param input_dropout:
        :param hidden_dropout:
        :param batch_normalization:
        :param priority_exp: alpha, the smaller it is, the more "uniform" the sampling distribution
        :param priority_shift: epsilon
        :param history_len:
        :param policy:
        :param exploration_start:
        :param exploration_min:
        :param exploration_anneal_steps:
        :param exploration_temp: temperature (tau) give a chance to others as well: the higher it is, the more "uniform" the sampling distribution
        :param batch_size:
        :param n_epochs:
        :param memory_size:
        :param q_clip:
        :param exploration_q_clip:
        """
        super().__init__(env, seed=seed)
        self.n_steps = multi_steps
        self.discount = discount
        self.target_update_freq = target_update_freq
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.decay_freq = decay_freq
        self.idealization = idealization
        self.dueling = dueling
        self.double = double
        self.history_len = history_len
        self.q_clip = q_clip
        self._lr = lr_init

        # build model
        self.layer_sizes = layer_sizes
        self.streams_size = streams_size
        self.loss = loss
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.batch_normalization = batch_normalization
        self.weights_init = weights_init
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self._model = self._build_model()
        if self.double:
            self._target_model = self._build_model()
            self._target_model.set_weights(self._model.get_weights())
        else:
            self._target_model = None

        # setup exploration
        self.policy = policy
        self._eps_greedy    = (self.policy == 'eps-greedy')
        self._boltzmann     = (self.policy == 'boltzmann')
        self._max_boltzmann = (self.policy == 'max-boltzmann')
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self.exploration_q_clip = exploration_q_clip
        self.exploration_temp = exploration_temp
        self.exploration_temp_min = exploration_temp_min
        self._explore_proba = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps
        self._exploration_tau = exploration_temp
        self._exploration_tau_drop = (exploration_temp - exploration_temp_min) / exploration_anneal_steps

        # replay
        self.prioritize_replay = prioritize_replay
        self.priority_exp = priority_exp
        self.priority_shift = priority_shift
        self.memory_size = memory_size
        self.multi_steps = multi_steps
        self._memory = Memory(memory_size,
                              self._env.state_size,
                              multi_steps,
                              history_len,
                              batch_size,
                              priority_shift,
                              prioritize_replay)

        blank_state = np.zeros(self._state_size)
        self._initial_history = deque([blank_state] * (self.history_len-1), maxlen=self.history_len)  # to be copied

    @property
    def config(self) -> dict:
        c = super().config
        c.update({attr: getattr(self, attr) for attr in [
            'lr_init', 'lr_decay', 'lr_min', 'decay_freq',
            'discount', 'idealization',
            'multi_steps', 'history_len',
            'layer_sizes', 'input_dropout', 'hidden_dropout', 'batch_normalization',
            'loss', 'hidden_activation', 'out_activation', 'weights_init',
            'double', 'target_update_freq',
            'dueling', 'streams_size',
            'prioritize_replay', 'priority_exp', 'priority_shift',
            'policy', 'exploration_start', 'exploration_min', 'exploration_anneal_steps',
            'exploration_temp', 'exploration_temp_min',
            'batch_size', 'n_epochs', 'memory_size',
            'q_clip', 'exploration_q_clip',
        ]})
        return c

    def _build_model(self) -> Model:
        hidden_kwargs = dict(kernel_initializer=self.weights_init, activation=self.hidden_activation)
        out_kwargs    = dict(kernel_initializer=self.weights_init, activation=self.out_activation)
        
        inp = layers.Input(shape=(self._state_size * self.history_len,))

        # multiple fully connected layers at the beginning
        x = inp 
        if self.input_dropout:
            x = Dropout(self.input_dropout)(x)
        for size in self.layer_sizes:
            x = Dense(size, **hidden_kwargs)(x)
            if self.batch_normalization:
                x = BatchNormalization()(x)
            if self.hidden_dropout:
                x = Dropout(self.hidden_dropout)(x)

        if self.dueling:
            # action advantage stream
            a = x
            if self.streams_size:
                a = Dense(self.streams_size, **hidden_kwargs)(a)
            a = Dense(self._n_actions, **out_kwargs)(a)
            baseline = layers.Lambda(lambda adv: K.mean(adv, axis=1, keepdims=True))(a)
            a = layers.subtract([a, baseline])

            # state value stream
            v = x
            if self.streams_size:
                v = Dense(self.streams_size, **hidden_kwargs)(v)
            v = Dense(1, **out_kwargs)(v)

            q = layers.add([v, a])  # combine back streams
        else:
            q = Dense(self._n_actions, **out_kwargs)(x)

        model = Model(inp, q)
        model.compile(optimizer=Adam(lr=self._lr), loss=self.loss)
        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        state = np.expand_dims(state, axis=0)  # make it a "batch" with a single entry
        q = self._model.predict(state)[0]
        best = q.argmax()

        if eval_mode:
            return best

        explore_roll = random.random() < self._explore_proba

        if self._eps_greedy:
            # with eps probability, sample uniformly at random
            if explore_roll:
                return np.random.choice(self._n_actions)
            else:
                return best

        q = np.clip(q, *self.exploration_q_clip)
        p = np.exp(q / self._exploration_tau)
        p[best] = 0  # select something other than the best one
        proportionally_sampled = np.random.choice(self._n_actions, p=p/sum(p))

        if self._max_boltzmann:
            # with eps probability, sample proportionally to estimated value
            if explore_roll:
                return proportionally_sampled
            else:
                return best

        if self._boltzmann:
            # always sample proportionally to estimated value
            return proportionally_sampled

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        # remember
        self._memory.add(state, action, reward, done)
        # after enough enough experience
        if len(self._memory) >= self.batch_size:
            self._replay()

    def train(self):
        """ overridden because we need state history """
        history = copy(self._initial_history)
        current_state = self.initialise_episode()
        done = False

        while not done:
            history.append(current_state)

            action = self._select_action(np.concatenate(history))
            next_state, reward, done = self._env.step(action)
            self._learn_transition(current_state, action, reward, next_state, done)

            current_state = next_state

        self._episode += 1

        # Perform episode end updates
        self._explore_proba = max(self._explore_proba - self._exploration_drop, self.exploration_min)
        self._exploration_tau = max(self._exploration_tau - self._exploration_tau_drop, self.exploration_temp_min)

        if self._episode % self.decay_freq == 0:
            self._lr = max(self._lr * self.lr_decay, self.lr_min)
            K.set_value(self._model.optimizer.lr, self._lr)

        if self.double and self._episode % self.target_update_freq == 0:
            self._update_target_weights()

    def eval(self, n_episodes=100) -> [dict]:
        stats = []
        for episode in range(n_episodes):
            history = copy(self._initial_history)
            state = self.initialise_episode()
            done = False

            while not done:
                history.append(state)
                action = self._select_action(np.concatenate(history), eval_mode=True)
                state, reward, done = self._env.step(action)

            stats.append({
                'reward': self._env.total_reward,
                'steps':  self._env.n_steps,
            })
        return stats

    def _update_target_weights(self):
        # TODO try target rl instead of sudden update
        # new_target_weights = [
        #     (1 - self.target_lr) * target_w + self.target_lr * model_w
        #     for model_w, target_w in zip(self._model.get_weights(), self._target_model.get_weights())
        # ]
        self._target_model.set_weights(self._model.get_weights())

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        state, action, return_, next_state, done, memory_indices = self._memory.sample(
            self.priority_exp, self.discount)

        q = self._model.predict(state)
        q = np.clip(q, *self.q_clip)

        if self.double:
            next_q = self._target_model.predict(next_state)
            main_argmax = q.argmax(axis=1)  # pick according to latest estimations
            next_max = next_q[[range(self.batch_size), main_argmax]]  # but use old estimation instead
        else:
            next_q = self._model.predict(next_state)
            next_max = next_q.max(axis=1)
        next_q = np.clip(next_q, *self.q_clip)

        next_avg = next_q.mean(axis=1)
        next_agg = next_avg + (next_max - next_avg) * self.idealization  # works even when next "max" < next avg

        target = return_ + (1-done) * (self.discount * next_agg)  # add discounted future discount only if not done
        actions_slice = np.arange(len(action)), action
        q[actions_slice] = target

        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)

        predictions = self._model.predict(state)
        errors = abs(predictions[actions_slice] - target)
        self._memory.update_priorities(memory_indices, errors)

    def save(self, path: str):
        self._model.save(path + '.h5')
