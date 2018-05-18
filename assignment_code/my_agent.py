import random
from collections import deque
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K
from lunarlander_wrapper import EnvWrapper, State


class Agent:
    """ Base agent class, with train and eval functions implemented """

    def __init__(self, env: EnvWrapper, seed=24):
        """ Set reference variables and random seed """
        self._env = env  # environment wrapper that provides extra info
        self._state_size = self._env.state_size
        self._n_actions  = self._env.n_actions

        self._episode = 0

        # set random seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._env._env.seed(seed)

    def initialise_episode(self) -> [float]:
        """
        Reset the environment through the wrapper.
        :return: initial state vector
        """
        return self._env.reset()

    def _select_action(self, state: State, eval_mode=False) -> int:
        """ What action to take in this situation? """
        raise NotImplementedError

    def _learn_transition(self, state: State, action: int, reward: float, next_state: State, done: bool):
        """ Turn this experience into knowledge of how to act in the future. """
        raise NotImplementedError

    def train(self):
        """ Reset the environment and run for the entire episode until it ends. """
        # Initialise the episode and environment
        state = self.initialise_episode()

        # Main loop of training session
        done = False
        while not done:
            # Select an action, take it and observe outcome
            action = self._select_action(state)
            next_state, reward, done = self._env.step(action)
            self._learn_transition(state, action, reward, next_state, done)
            state = next_state

        self._episode += 1


class MyAgent(Agent):
    """ TODO: add description for this class
    Simple DQN agent
    """
    def __init__(self, wrapper, seed=42,
                 lr_init=0.005, decay_freq=200, lr_decay=.1, lr_min=.00001,
                 discount=.99, layer_sizes=(384, 192),
                 exploration_start=1, exploration_min=.01, exploration_anneal_steps=150,
                 loss='mse', hidden_activation='sigmoid', out_activation='linear', init_method='lecun_uniform',
                 q_clip=(-10000, +10000), batch_size=32, n_epochs=1,
                 memory_size=50000, min_mem_size=1000,
                 ):
        super().__init__(wrapper, seed=seed)
        self.discount = discount

        self.lr_init = lr_init
        self.decay_freq = decay_freq
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self._lr = lr_init

        self.q_clip = q_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # build model
        self.layer_sizes = layer_sizes
        self.loss = loss
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.init_method = init_method
        self._model = self._build_model()

        # setup exploration
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self._exploration_eps = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps

        # replay
        self.memory_size = memory_size
        self.min_mem_size = min_mem_size
        self._memory = deque(maxlen=memory_size)

    def _build_model(self) -> Model:
        model = Sequential()

        model.add(Dense(self.layer_sizes[0], activation=self.hidden_activation, kernel_initializer=self.init_method,
                        input_shape=(self._state_size,)))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation=self.hidden_activation, kernel_initializer=self.init_method,))

        model.add(Dense(self._n_actions, activation=self.out_activation, kernel_initializer=self.init_method))
        model.compile(optimizer=Adam(lr=self._lr), loss=self.loss)
        return model

    def _select_action(self, state: [float], eval_mode=False) -> int:
        """ What action to take in this situation? """
        state = np.expand_dims(state, axis=0)
        q = self._model.predict(state)[0]

        # Explore
        if random.random() < self._exploration_eps and not eval_mode:
            return np.random.choice(self._n_actions)

        # Exploit
        else:
            return q.argmax()

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        """ Turn this experience into knowledge of how to act in the future. """
        # remember
        self._memory.append((state, action, reward, next_state, done))
        if len(self._memory) >= self.min_mem_size:
            self._replay()

    def train(self) -> float:
        """ Reset the environment and run for the entire episode until it ends. """
        super().train()

        # Perform episode end updates
        if len(self._memory) >= self.min_mem_size:
            self._exploration_eps -= self._exploration_drop
            self._exploration_eps = max(self._exploration_eps, self.exploration_min)
            if self._episode % self.decay_freq == 0:
                self._lr = max(self._lr * self.lr_decay, self.lr_min)
                K.set_value(self._model.optimizer.lr, self._lr)

        return self._env.total_reward

    def _replay(self):
        """ sample of batch from experience and fit the network to it """
        sample_indices = np.random.choice(len(self._memory), self.batch_size)
        sample = [self._memory[i] for i in sample_indices]
        state, action, reward, next_state, done = (np.array(l) for l in zip(*sample))

        q = self._model.predict(state)
        q = np.clip(q, *self.q_clip)
        next_q = self._model.predict(next_state)
        next_q = np.clip(next_q, *self.q_clip)
        next_max = next_q.max(axis=1)

        target = reward + (1 - done) * (self.discount * next_max)  # add future discount only if not done
        q[np.arange(len(action)), action] = target

        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)
