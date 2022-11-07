import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import itertools
import os
from abc import abstractmethod


class Env:
    """
    ENV WRAPPER
    """

    def __init__(self, config, acq):
        self.config = config
        self.acq = acq
        self.init_env()

    def init_env(self):
        # so far only aptamers has been implemented
        if self.config.env.main == "aptamers":
            self.env = EnvAptamers(self.config, self.acq)
        elif self.config.env.main == "grid":
            self.env = EnvGrid(self.config, self.acq)
        else:
            raise NotImplementedError


"""
Generic Env Base Class
"""


class EnvBase:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq
        self.device = self.config.device

    @abstractmethod
    def create_new_env(self, idx):
        raise NotImplementedError

    @abstractmethod
    def init_env(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self):
        """
        get all possible actions to get the parents
        """
        raise NotImplementedError

    @abstractmethod
    def get_mask(self):
        """
        for sampling in GFlownet and masking in the loss function
        """
        raise NotImplementedError

    @abstractmethod
    def get_parents(self, backward=False, state=None, done=None):
        """
        to build the training batch (for the inflows)
        """
        raise NotImplementedError

    def get_paths(self, path_list, actions):
        """
        Determines all paths leading to each state in path_list, recursively.
        Args
        ----
        path_list : list
            List of paths (lists)
        actions : list
            List of actions within each path
        Returns
        -------
        path_list : list
            List of paths (lists)
        actions : list
            List of actions within each path
        """
        current_path = path_list[-1].copy()
        current_path_actions = actions[-1].copy()
        parents, parents_actions = self.get_parents(
            state=list(current_path[-1]), done=False, backward=True
        )
        # parents = [self.obs2state(el).tolist() for el in parents]
        if parents == []:
            return path_list, actions
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            if idx > 0:
                path_list.append(current_path)
                actions.append(current_path_actions)
            path_list[-1] += [p]
            actions[-1] += [a]
            path_list, actions = self.get_paths(path_list, actions)
        return path_list, actions

    @abstractmethod
    def step(self, action):
        """
        for forward sampling
        """
        raise NotImplementedError

    @abstractmethod
    def acq2rewards(self, acq_values):
        """
        correction of the value of the AF for positive reward (or to scale it)
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, states, done):
        """
        get the reward values of a batch of candidates
        """
        raise NotImplementedError

    def acq2reward(self, acq_values):
        """
        Args: array of scores

        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.reward_func == "power":
            return np.clip(
                (-1.0 * acq_values / self.reward_norm) ** self.reward_beta,
                self.min_reward,
                None,
            )
        elif self.reward_func == "boltzmann":
            return np.clip(
                np.exp(-1.0 * self.reward_beta * acq_values),
                self.min_reward,
                None,
            )
        else:
            raise NotImplemented

    def reward2acq(self, reward):
        """
        Args: array of boltzmann/power rewards
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return -np.exp(
                (np.log(reward) + self.reward_beta * np.log(self.reward_norm))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return -1.0 * np.log(reward) / self.reward_beta
        else:
            raise NotImplemented


"""
Specific Envs
"""


class EnvAptamers(EnvBase):
    def __init__(self, config, acq) -> None:
        super().__init__(config, acq)
        self.device = self.config.device
        # determine max length of state
        self.max_seq_len = self.config.env.aptamers.max_len
        self.pad_len = self.max_seq_len + 1
        # prepare action space
        self.max_word_len = self.config.env.aptamers.max_word_len
        self.min_word_len = self.config.env.aptamers.min_word_len
        self.n_alphabet = self.config.env.aptamers.dict_size
        self.action_space = self.get_action_space()
        # determine ohe dim, and hence output dim of policy
        self.dict_size = len(self.action_space) + 1
        # calculate input dim to policy
        self.obs_dim = self.pad_len * self.dict_size
        # everything else
        self.min_seq_len = self.config.env.aptamers.min_len
        self.token_eos = self.get_token_eos(self.action_space)
        self.min_reward = 1e-8
        self.reward_beta = self.config.env.reward_beta
        self.reward_norm = self.config.env.reward_norm
        self.reward_func = self.config.env.reward_func
        self.env_class = EnvAptamers
        self.init_env()

    def create_new_env(self, idx):
        env = EnvAptamers(self.config, self.acq)
        env.init_env(idx)
        return env

    def init_env(self, idx=0):
        self.state = np.array([])
        self.n_actions_taken = 0
        self.done = False
        self.id = idx
        self.last_action = None

    def get_action_space(self):
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len + 1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        return actions

    def get_token_eos(self, action_space):
        return len(action_space)

    def acq2reward(self, acq_values):
        return super().acq2reward(acq_values)

    def reward2acq(self, reward):
        return super().reward2acq(reward)

    def get_mask(self):

        mask = [1] * (len(self.action_space) + 1)

        if self.done:
            return [0 for _ in mask]

        seq_len = len(self.state)

        if seq_len < self.min_seq_len:
            mask[self.token_eos] = 0
            return mask

        elif seq_len == self.max_seq_len:
            mask[: self.token_eos] = [0] * len(self.action_space)
            return mask

        else:
            return mask

    def get_parents(self, backward=False, state=None, done=None):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done

        if done:
            if state[-1] == self.token_eos:
                parents_a = [self.token_eos]
                parents = [state[:-1]]
                if backward:
                    # in getpaths, backward id true so we need to modify 
                    # done. but we also need to modify the done of the env 
                    # in case we are using the self thingy later
                    # need to ask bao
                    self.done = False
                return parents, parents_a
            else:
                raise NameError

        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if state[-len(a) :] == list(a):
                    parents.append((state[: -len(a)]))
                    actions.append(idx)

            return parents, actions

    def step(self, action):
        valid = False
        seq = self.state
        seq_len = len(seq)

        if (action == [self.token_eos]) and (self.done == False):
            if seq_len >= self.min_seq_len and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                self.done = True
                self.n_actions_taken += 1
                self.state = next_seq
                self.last_action = self.token_eos

                return next_seq, action, valid

        if self.done == True:
            valid = False
            return None, None, valid

        elif self.done == False and not (action == [self.token_eos]):
            if (
                action in list(map(list, self.action_space))
                and seq_len <= self.max_seq_len
            ):
                valid = True
                next_seq = np.append(seq, action)
                self.n_actions_taken += 1
                self.state = next_seq
                self.last_action = action
                return next_seq, action, valid

        else:
            raise TypeError("invalid action to take")

    def get_reward(self, states, done):
        rewards = np.zeros(len(done), dtype=float)
        final_states = [s for s, d in zip(states, done) if d]
        inputs_af_base = [self.manip2base(final_state) for final_state in final_states]

        final_rewards = (
            self.acq.get_reward(inputs_af_base)
            .view(len(final_states))
            .cpu()
            .detach()
            .numpy()
        )
        final_rewards = self.acq2reward(final_rewards)

        done = np.array(done)
        rewards[done] = final_rewards
        return rewards

    def base2manip(self, state):
        seq_base = state
        seq_manip = np.concatenate((seq_base, [self.token_eos]))
        return seq_manip

    def manip2base(self, state):
        seq_manip = state
        if seq_manip[-1] == self.token_eos:
            seq_base = seq_manip[:-1]
            return seq_base
        else:
            raise TypeError


class EnvGrid(EnvBase):
    def __init__(self, config, acq) -> None:
        super().__init__(config, acq)
        self.device = self.config.device
        # determine max length of state
        self.n_dim = self.config.env.grid.n_dim
        self.pad_len = self.n_dim
        # prepare action space
        self.min_step_len = self.config.env.grid.min_step_length
        self.max_step_len = self.config.env.grid.max_step_length
        self.action_space = self.get_action_space()
        # determine ohe dim, ie, max state len, and hence output dim of policy
        self.max_seq_len = self.config.env.grid.length
        self.dict_size = len(self.action_space) + 1
        # calculate input dim to policy
        self.obs_dim = self.pad_len * self.max_seq_len
        # everything else
        self.token_eos = self.get_token_eos(self.action_space)
        self.min_reward = 1e-8
        self.reward_beta = self.config.env.reward_beta
        self.reward_norm = self.config.env.reward_norm
        self.reward_func = self.config.env.reward_func
        self.env_class = EnvGrid
        self.init_env()

    def create_new_env(self, idx):
        env = EnvGrid(self.config, self.acq)
        env.init_env(idx)
        return env

    def init_env(self, idx=0):
        """
        Resets the environment.
        """
        self.state = np.array([0 for _ in range(self.n_dim)])
        self.n_actions_taken = 0
        self.done = False
        self.id = idx
        self.last_action = None

    def get_action_space(self):
        """
        Returns list with all possible actions.
        Example:
            actions = [(0,), (1,)]
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        dims = [a for a in range(self.n_dim)]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, repeat=r)]
            actions += actions_r
        return actions

    def get_token_eos(self, action_space):
        return len(action_space)

    def acq2reward(self, acq_values):
        return super().acq2reward(acq_values)

    def reward2acq(self, reward):
        return super().reward2acq(reward)

    def get_mask(self):
        """
        Returns a vector of length the action space + 1:
        True if action is invalid given the current state, False otherwise.
        """

        # if state is None:
        # state = self.state.copy()
        # if done is None:
        # done = self.done
        if self.done:
            return [0 for _ in range(len(self.action_space) + 1)]
        mask = [1 for _ in range(len(self.action_space) + 1)]
        for idx, a in enumerate(self.action_space):
            for d in a:
                if self.state[d] + 1 >= self.max_seq_len:
                    mask[idx] = 0
                    break
        return mask

    # TODO: implement obs2state, readable2state, state2readable if required

    def get_parents(self, backward=False, state=None, done=None):
        """
        Determines all parents and actions that lead to state.
        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.
        action : int
            Last action performed
        Returns
        -------
        parents : list
            List of parents as state2obs(state)
        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if self.done:
            return [self.state], [self.token_eos]

        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_aux = state.copy()
                for a_sub in a:
                    if state_aux[a_sub] > 0:
                        state_aux[a_sub] -= 1
                    else:
                        break
                else:
                    parents.append((state_aux))
                    actions.append(idx)
        return parents, actions

    def step(self, action):
        """
        Executes step given an action.
        Args
        ----
        a : list
            Index of action in the action space. a == eos indicates "stop action"
        Returns
        -------
        self.state : list
            The sequence after executing the action
        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # All dimensions are at the maximum length
        if all([s == self.max_seq_len - 1 for s in self.state]):
            self.done = True
            self.n_actions_taken += 1
            self.last_action = self.token_eos
            return self.state, [self.token_eos], True
        if action != [self.token_eos]:
            state_next = self.state.copy()
            # action is already a list so no need to do the following I guess
            # if action.ndim == 0:
            # action = [action]
            for a in action:
                state_next[a] += 1
            if any([s >= self.max_seq_len for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions_taken += 1
                self.last_action = action
            return self.state, action, valid
        else:
            self.done = True
            self.n_actions_taken += 1
            self.last_action = self.token_eos
            return self.state, [self.token_eos], True

    def get_reward(self, states, done):
        """
        Args:
            states: tuple of arrays
            done: tuple of boolen values
            len(states) = len(done)
        Function:
            Calls the desired acquisition function to calculate rewards of final states
        Return:
            rewards: numpy array containing reward of all states terminal and non-terminal
        """
        rewards = np.zeros(len(done), dtype=np.float32)
        final_states = [s for s, d in zip(states, done) if d]

        final_rewards = (
            self.acq.get_reward(final_states)
            .view(len(final_states))
            .cpu()
            .detach()
            .numpy()
        )
        final_rewards = self.acq2reward(final_rewards)

        done = np.array(done)
        rewards[done] = final_rewards
        return rewards

    def base2manip(self, state):
        # state is just a coordinate -- does not make sense to add eos to coordinate
        return state

    def manip2base(self, state):
        return state
