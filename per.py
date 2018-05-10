import numpy as np


class SumTree:
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with its priority in priority_heap and data frameworks.
    """

    def __init__(self, state_size: int, max_size: int, head_before: int, head_after: int):
        self.max_size = max_size  # for all priority values
        self.priority_heap = np.zeros(2 * max_size - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: max_size - 1                       size: max_size
        self.states     = np.empty((self.max_size, state_size), dtype=float)
        self.actions    = np.empty(self.max_size, dtype=int)
        self.rewards    = np.empty(self.max_size, dtype=float)
        self.terminals  = np.empty(self.max_size, dtype=bool)
        self.priorities = np.empty(self.max_size, dtype=float)

        self.max_priority = 0
        self.write_head = 0
        self.n_entries = 0

        self.head_before = head_before  # history_len
        self.head_after = head_after   # multi_step

    def add(self, priority: float, state, action, reward, terminal):
        self.states[self.write_head] = state
        self.actions[self.write_head] = action
        self.rewards[self.write_head] = reward
        self.terminals[self.write_head] = terminal

        tree_idx = self.write_head + self.max_size - 1
        self.update(tree_idx, priority)  # update tree_frame

        self.write_head += 1
        self.write_head %= self.max_size
        self.n_entries += 1
        self.n_entries = min(self.n_entries, self.max_size)

    def update(self, tree_idx: int, new_priority: float):
        delta = new_priority - self.priority_heap[tree_idx]
        self.priority_heap[tree_idx] = new_priority
        self.max_priority = max(self.max_priority, new_priority)

        # then propagate the change through priority_heap
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.priority_heap[tree_idx] += delta

    def get_leaf(self, v: float) -> int:
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            left_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            right_idx = left_idx + 1
            if left_idx >= len(self.priority_heap):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.priority_heap[left_idx] or self.priority_heap[right_idx] == 0:
                    parent_idx = left_idx
                else:
                    v -= self.priority_heap[left_idx]
                    parent_idx = right_idx

        # priority = self.priority_heap[leaf_idx]
        data_idx = leaf_idx - self.max_size + 1
        is_invalid = self.write_head - self.head_before <= data_idx <= self.write_head + self.head_after or \
                     data_idx <= self.head_after or data_idx >= self.max_size - self.head_before
        
        if is_invalid:
            return None
        else:
            return data_idx

    @property
    def total_priority(self):
        return self.priority_heap[0]  # the root


class Memory:
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    # beta_increment_per_sampling = 0.001

    def __init__(self, state_size: int, max_size: int, multi_step: int, history_len: int, batch_size: int, epsilon: float,
                 prioritize: bool, err_clip: float):
        """

        :param max_size:
        :param multi_step:
        :param history_len:
        :param batch_size:
        :param epsilon: small amount to avoid zero priority
        :param prioritize:
        """
        self.tree = SumTree(state_size, max_size, head_before=history_len, head_after=multi_step)
        self.max_size = max_size
        self.history_len = history_len
        self.multi_step = multi_step
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.err_clip = err_clip
        self.prioritize = prioritize  # TODO

    def __len__(self):
        return self.tree.n_entries

    def store(self, state, action, reward, terminal):
        max_priority = self.tree.max_priority or self.err_clip
        self.tree.add(max_priority, state, action, reward, terminal)  # add with maximum priority

    def sample(self, discount: float):
        segment = self.tree.total_priority / self.batch_size  # priority segment

        indices = np.empty(self.batch_size, int)  # data idxs
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            data_idx = None
            while not data_idx:
                v = np.random.uniform(a, b)
                data_idx = self.tree.get_leaf(v)

            indices[i] = data_idx

        # weighted importance sampling
        # sampling_probabilities = priorities / self.tree.total_priority
        # is_weight = (self.tree.n_entries * sampling_probabilities) ** -self.beta  # importance sampling
        # is_weight /= max(is_weight)

        # self.beta += self.beta_increment_per_sampling
        # self.beta = min(1, self.beta)  # max = 1

        history_indices = indices + np.expand_dims(np.arange(self.history_len), axis=1)
        # history_indices %= self.max_size
        s = np.concatenate(self.tree.states[history_indices], axis=1)

        history_indices += 1
        # history_indices %= self.max_size
        ns = np.concatenate(self.tree.states[history_indices], axis=1)

        if self.multi_step == 1:
            returns = self.tree.rewards[indices]
        else:
            returns = np.zeros(self.batch_size)
            for sample_number, sampled_idx in enumerate(indices):
                for step in range(self.multi_step):
                    current_idx = (sampled_idx + step) #% self.size
                    if self.tree.terminals[current_idx]:  # don't consider further rewards one the episode is marked done
                        break
                    returns[sample_number] += discount ** step * self.tree.rewards[current_idx]

        tree_indices = indices + self.max_size - 1
        return s, self.tree.actions[indices], returns, ns, self.tree.terminals[indices], tree_indices

    def update_priorities(self, indices: [int], errors: [float], alpha: float):
        """
        alpha (0-1): convert the importance of TD error to priority
        """
        errors = abs(errors)
        errors += self.epsilon  # avoid 0
        errors = np.minimum(errors, self.err_clip)

        priorities = errors ** alpha
        for i, p in zip(indices, priorities):
            self.tree.update(i, p)
