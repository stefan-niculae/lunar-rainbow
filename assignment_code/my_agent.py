""" Contains the learning agent, DQN. """

import random
from collections import deque

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K

from lunarlander_wrapper import EnvWrapper


class Agent:
    """ Base agent class, with training skeleton implemented """
    def __init__(self, env: EnvWrapper, seed):
        """ Propagate random seed """
        self._env = env  # will be used for interaction during training
        self._episode = 0  # episode counter (used for parameter updates)

        # Propagate seed to all used components
        random.seed(seed)
        np.random.seed(seed)  # also sets for Keras
        self._env._env.seed(seed)

    def _select_action(self, state: [float]) -> int:
        """ What action to take in this situation? """
        raise NotImplementedError

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        """ Turn this experience into knowledge of how to act in the future. """
        raise NotImplementedError

    def train(self):
        """ Reset the environment and run for the entire episode until it ends. """
        state = self._env.reset()  # start new episode and get initial state

        done = False
        while not done:  # run until a terminal state is reached
            action = self._select_action(state)  # the agent picks an action
            next_state, reward, done = self._env.step(action)  # take selected action
            self._learn_transition(state, action, reward, next_state, done)  # tell agent the outcome
            state = next_state  # prepare for next loop iteration

        self._episode += 1


class MyAgent(Agent):
    """
    Deep Q-Network algorithm: uses a neural network to estimate action values given state features.
    Observed action outcomes (transitions) are stored in a circular memory buffer from which random
    samples are selected for training.
    Exploration is decreased linearly and learning rate is decreased after a set number of episodes.
    """
    def __init__(self, wrapper, seed=7,
                 discount=.99,
                 lr_init=0.005, decay_freq=200, lr_decay=.1, lr_min=.00001,
                 exploration_start=1, exploration_min=.01, exploration_anneal_steps=150,
                 layer_sizes=(384, 192), init_method='lecun_uniform',
                 hidden_activation='sigmoid', out_activation='linear', loss='mse',
                 q_clip=(-10000, +10000), batch_size=32, n_epochs=1,
                 memory_size=50000, min_mem_size=1000,
                 ):
        """
        Logistic parameters:
        :param wrapper: an wrapper over a Gym environment
        :param seed: random for reproducibility

        Basic RL algorithm parameters:
        :param discount: how much future value is taken into consideration

        Learning rate of the NN model:
        :param lr_init: initial learning rate
        :param decay_freq: at what episode interval to decrease the learning rate
        :param lr_decay: by what factor to multiply the learning rate (.1 means divide by 10)
        :param lr_min: threshold below which learning rate will not be decreased

        Action selection policy epsilon parameters:
        :param exploration_start: value for first episode
        :param exploration_min: final value after decreasing
        :param exploration_anneal_steps: over how many episodes to decrease

        NN model configuration:
        :param layer_sizes: how many neurons for each layer
        :param init_method: neuron weight initialization method
        :param hidden_activation: nonlinearity function of hidden layer neurons
        :param out_activation: activation function of final layer
        :param loss: loss function based on which the gradient descent is computed

        Training process:
        :param q_clip: minimum and maximum limits of estimated action values
        :param n_epochs: number of passes through a sampled batch

        :param batch_size: number of transitions to sample
        :param memory_size: maximum number of transitions remembered
        :param min_mem_size: minimum number of transitions for training to start
        """
        super().__init__(wrapper, seed=seed)
        self.discount = discount

        self.lr_init = lr_init
        self.decay_freq = decay_freq
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self._lr = lr_init  # will be decreased during training

        # setup exploration
        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.exploration_anneal_steps = exploration_anneal_steps
        self._exploration_eps = exploration_start  # will be decreased during training
        self._exploration_drop = (exploration_start - exploration_min) / exploration_anneal_steps

        # model
        self.layer_sizes = layer_sizes
        self.loss = loss
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.init_method = init_method
        self._model = self._build_model()

        # training process
        self.q_clip = q_clip
        self.n_epochs = n_epochs

        # replay memory
        self.memory_size = memory_size
        self.min_mem_size = min_mem_size
        self.batch_size = batch_size
        self._memory = deque(maxlen=memory_size)

    def _build_model(self) -> Model:
        """ Constructs a NN that predicts state action values in a given state. """
        model = Sequential()
        hidden_kwargs = dict(activation=self.hidden_activation, kernel_initializer=self.init_method)

        # First layer needs to specify input size: one input for state feature
        model.add(Dense(self.layer_sizes[0], input_shape=(self._env.state_size,), **hidden_kwargs))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, **hidden_kwargs))

        # Different activation for the output layer: one output for each possible action
        model.add(Dense(self._env.n_actions,
                        activation=self.out_activation, kernel_initializer=self.init_method))

        model.compile(optimizer=Adam(lr=self._lr), loss=self.loss)
        return model

    def _select_action(self, state: [float]) -> int:
        """ Epsilon-greedy action selection strategy:
            select randomly with epsilon probability, otherwise select action estimated as best. """
        if random.random() < self._exploration_eps:  # explore
            return np.random.choice(self._env.n_actions)
        else:  # exploit
            q = self._model.predict(np.expand_dims(state, axis=0))[0]  # turn into a batch of one
            return q.argmax()

    def _learn_transition(self, state: [float], action: int, reward: float, next_state: [float], done: bool):
        """ Store current observation in memory and train on a sample of previous observations. """
        self._memory.append((state, action, reward, next_state, done))  # remember
        if len(self._memory) >= self.min_mem_size:  # after enough experience, learn from it
            self._replay()

    def train(self) -> float:
        """ After finishing training for one episode, update decreasing parameters. """
        super().train()

        if len(self._memory) < self.min_mem_size:  # only decrease after learning has started
            # Decrease exploration rate
            self._exploration_eps -= self._exploration_drop
            self._exploration_eps = max(self._exploration_eps, self.exploration_min)

            # periodically decrease learning rate
            if self._episode % self.decay_freq == 0:
                self._lr = max(self._lr * self.lr_decay, self.lr_min)
                K.set_value(self._model.optimizer.lr, self._lr)

        return self._env.total_reward

    def _replay(self):
        """ Sample of batch of transitions from experience and fit the network on it. """
        sample_indices = np.random.choice(len(self._memory), self.batch_size)
        sample = [self._memory[i] for i in sample_indices]
        state, action, reward, next_state, done = (np.array(l) for l in zip(*sample))

        q = self._model.predict(state)  # estimated action values in each batch state
        q = np.clip(q, *self.q_clip)

        next_q = self._model.predict(next_state)  # estimated action values in the immediately next state
        next_q = np.clip(next_q, *self.q_clip)
        next_max = next_q.max(axis=1)  # aggregate of next state value

        target = reward + (1 - done) * (self.discount * next_max)  # add future discount only if not done
        q[np.arange(len(action)), action] = target  # update value estimation to computed targets

        # Fit the model on the updated action values in the observed state
        self._model.fit(state, q, epochs=self.n_epochs, verbose=0)
