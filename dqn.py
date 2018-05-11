import random
from collections import deque
import numpy as np
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, AlphaDropout, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
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

class DQNP(Agent):
    """ TODO: add description for this class
    Simple DQN agent
    """
    def __init__(self, env, seed=42,
                 lr_init=0.005, decay_freq=200, lr_decay=.1, lr_min=.00001, history_len=2,
                 discount=.99, idealization=1,
                 policy='eps-greedy', exploration_start=1, exploration_min=.01, exploration_anneal_steps=150, exploration_temp=1,
                 double=False, target_update_freq=25,
                 dueling=False, streams_size=0,
                 layer_sizes=(384, 192), loss='mse', hidden_activation='sigmoid', out_activation='linear',
                 input_dropout=0, hidden_dropout=0, batch_normalization=False,
                 weights_init='lecun_uniform', optimizer='adam',
                 q_clip=(-10000, +10000), batch_size=32, n_epochs=1,
                 memory_size=50000, min_mem_size=1000, normalize=False,
                 ):
        self.normalize = normalize
        super().__init__(env, seed=seed)
        self.discount = discount
        self.idealization = idealization
        self.history_len = history_len

        self.lr_init = lr_init
        self.decay_freq = decay_freq
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self._lr = lr_init

        if type(q_clip) is not tuple:
            q_clip = (-abs(int(q_clip)), +abs(int(q_clip)))
        self.q_clip = q_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # build model
        self.layer_sizes = layer_sizes
        self.loss = loss
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.batch_normalization = batch_normalization
        if streams_size == -1:
            self.dueling = False
        self.streams_size = streams_size
        self.dueling = dueling
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.weights_init = weights_init
        if not target_update_freq:
            double = False
        self.double = double
        self.target_update_freq = target_update_freq
        self.optimizer = optimizer
        self._model = self._build_model()
        if self.double:
            self._target_model = self._build_model()
            self._target_model.set_weights(self._model.get_weights())
        else:
            self._target_model = None

        # setup exploration
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self._exploration_eps = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps
        self.exploration_temp = exploration_temp
        self.policy = policy
        self._eps_greedy    = (self.policy == 'eps-greedy')
        self._boltzmann     = (self.policy == 'boltzmann')
        self._max_boltzmann = (self.policy == 'max-boltzmann')

        # replay
        memory_size = int(memory_size)
        self.memory_size = memory_size
        self.min_mem_size = min_mem_size
        self._memory = deque(maxlen=memory_size)

    @property
    def config(self) -> dict:
        c = super().config
        c.update({attr: getattr(self, attr) for attr in [
            'batch_size', 'n_epochs',
            'memory_size', 'min_mem_size',
            'q_clip',
            'discount', 'history_len', 'idealization',
            'lr_init', 'decay_freq', 'lr_decay', 'lr_min',
            'layer_sizes', 'hidden_activation', 'out_activation', 'loss', 'optimizer',
            'policy', 'exploration_temp', 'exploration_start', 'exploration_min', 'exploration_anneal_steps', 'weights_init',
            'normalize',
            'dueling', 'streams_size',
            'input_dropout', 'hidden_dropout', 'batch_normalization',
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

        optimizer_class = {
            'rmsprop': RMSprop,
            'adagrad': Adagrad,
            'adadelta': Adadelta,
            'adam': Adam,
            'adamax': Adamax,
            'nadam': Nadam,
        }[self.optimizer]
        model.compile(optimizer=optimizer_class(lr=self._lr), loss=self.loss)

        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        state = np.expand_dims(state, axis=0)
        q = self._model.predict(state)[0]
        best = q.argmax()

        if eval_mode:
            return best

        explore_roll = random.random() < self._exploration_eps

        if self._eps_greedy:
            # with eps probability, sample uniformly at random
            if explore_roll:
                return np.random.choice(self._n_actions)
            else:
                return best

        q = np.clip(q, *self.q_clip)
        p = np.exp(q / self.exploration_temp)
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

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float],
                          done: bool):
        # remember
        self._memory.append((state, action, reward, next_state, done))
        if len(self._memory) >= self.min_mem_size:
            self._replay()

    def train(self):
        blank_state = np.zeros(self._state_size)
        state_history = deque([blank_state] * (self.history_len - 1), maxlen=self.history_len)
        current_state = self.initialise_episode()
        done = False

        while not done:
            state_history.append(current_state)

            action = self._select_action(np.concatenate(state_history))
            next_state, reward, done = self._env.step(action)
            self._learn_transition(current_state, action, reward, next_state, done)

            current_state = next_state

        self._episode += 1

        # Perform episode end updates
        if len(self._memory) >= self.min_mem_size:
            self._exploration_eps -= self._exploration_drop
            self._exploration_eps = max(self._exploration_eps, self.exploration_min)
            if self._episode % self.decay_freq == 0:
                self._lr = max(self._lr * self.lr_decay, self.lr_min)
                K.set_value(self._model.optimizer.lr, self._lr)
            if self.double and self._episode % self.target_update_freq == 0:
                self._target_model.set_weights(self._model.get_weights())  # TODO try not sudden

    def eval(self, n_episodes=100) -> [dict]:
        stats = []
        for episode in range(n_episodes):
            blank_state = np.zeros(self._state_size)
            state_history = deque([blank_state] * (self.history_len - 1), maxlen=self.history_len)
            state = self.initialise_episode()
            done = False

            while not done:
                state_history.append(state)
                action = self._select_action(np.concatenate(state_history), eval_mode=True)
                state, reward, done = self._env.step(action)

            stats.append({
                'reward': self._env.total_reward,
                'steps': self._env.n_steps,
            })
        return stats

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        sample_indices = np.random.choice(len(self._memory), self.batch_size, replace=False)
        sample_indices -= 1

        sample = [self._memory[i] for i in sample_indices]
        state, action, reward, next_state, done = (np.array(l) for l in zip(*sample))

        blank_state = np.zeros(self._state_size)
        prev_state = [np.array(self._memory[i-1][0]) if i > 0 else blank_state for i in sample_indices]

        if self.history_len == 2:
            next_state = np.concatenate([state, next_state], axis=1)
            state = np.concatenate([prev_state, state], axis=1)
        elif self.history_len == 3:
            prev_prev_state = [np.array(self._memory[i - 2][0]) if i > 1 else blank_state for i in sample_indices]
            next_state = np.concatenate([prev_state, state, next_state], axis=1)
            state = np.concatenate([prev_prev_state, prev_state, state], axis=1)

        q = self._model.predict(state)
        q = np.clip(q, *self.q_clip)

        if self.double:
            next_q = self._target_model.predict(next_state)
            next_q = np.clip(next_q, *self.q_clip)
            main_argmax = q.argmax(axis=1)  # pick according to latest estimations
            next_max = next_q[[range(self.batch_size), main_argmax]]  # but use old estimation instead
        else:
            next_q = self._model.predict(next_state)
            next_q = np.clip(next_q, *self.q_clip)
            next_max = next_q.max(axis=1)

        next_avg = next_q.mean(axis=1)
        next_agg = next_avg + (next_max - next_avg) * self.idealization  # works even when next "max" < next avg

        target = reward + (1 - done) * (self.discount * next_agg)  # add future discount only if not done
        q[np.arange(len(action)), action] = target

        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)

    def save(self, path: str):
        self._model.save(path)
