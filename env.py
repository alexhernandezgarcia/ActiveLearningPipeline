import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import itertools
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod


'''
ENV WRAPPER
'''

class Env:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq.acq

        self.init_env()
    
    def init_env(self):
        if self.config.env.main == "aptamers":
            self.env = EnvAptamers(self.config, self.acq)
        else:
            raise NotImplementedError
    
'''
Generic Env Base Class
'''

class EnvBase:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq

        self.device = self.config.device


    @abstractmethod
    def create_new_env(self, idx):
        pass

    @abstractmethod
    def init_env(self, idx):
        pass

    @abstractmethod
    def get_action_space(self):
        '''
        get all possible actions to get the parents
        '''
        pass
 
    @abstractmethod
    def get_mask(self):
        '''
        for sampling in GFlownet and masking in the loss function
        '''
        pass
    
    @abstractmethod
    def get_parents(self, backward = False):
        '''
        to build the training batch (for the inflows)
        '''
        pass
    
    @abstractmethod
    def step(self,action):
        '''
        for forward sampling
        '''
        pass
    
    @abstractmethod
    def acq2rewards(self, acq_values):
        '''
        correction of the value of the AF for positive reward (or to scale it)
        '''

        pass
    
    @abstractmethod
    def get_reward(self, states, done):
        '''
        get the reward values of a batch of candidates
        '''
        pass


'''
Specific Envs
'''

class EnvAptamers(EnvBase):
    def __init__(self, config, acq) -> None:
        super().__init__(config, acq)

        self.device = self.config.device

        self.max_seq_len = self.config.env.max_len
        self.min_seq_len = self.config.env.min_len
        self.max_word_len = self.config.env.max_word_len
        self.min_word_len = self.config.env.min_word_len
        self.n_alphabet = self.config.env.dict_size
        self.total_fidelities = self.config.env.total_fidelities

        self.action_space = self.get_action_space()
        self.token_eos = self.get_token_eos(self.action_space)

        self.env_class = EnvAptamers

        self.init_env()
   
    def create_new_env(self, idx):
        env = EnvAptamers(self.config, self.acq)
        env.init_env(idx)
        return env
    
    def init_env(self, idx=0):
        super().init_env(idx)
        #Same comments than in middle_mf : self.seq, self.fid, to be merged in self.state for compatibility with single fidelity
        self.seq = np.array([])
        self.fid = None
        self.n_actions_taken = 0
        self.done = False
        #TODO : here, as we impose the flows, self.eos is crucial because that is the pseudo final_state on which we compute the reward.
        self.eos = False
        self.id = idx
        self.last_action = None
    
    def get_state(self):
        return (self.seq, self.fid)
    # Contrary to middle-mf, the possible actions are the same as in single fidelity, because we artificially chose the fidelity at the end by imposing the flows
    def get_action_space(self):
        super().get_action_space()
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len +1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat = r)]
            actions += actions_r
        return actions

    def get_token_eos(self, action_space):
        return len(action_space)
    # Same as in SF
    def get_mask(self):
        super().get_mask()

        mask = [1] * (len(self.action_space) + 1)

        if self.done : 
            return [0 for _ in mask]
        
        if self.eos and not(self.done):
            return [0 for _ in mask]
        
        seq_len = len(self.seq)

        if seq_len < self.min_seq_len:
            mask[self.token_eos] = 0
            return mask
        
        elif seq_len == self.max_seq_len:
            mask[:self.token_eos] = [0] * len(self.action_space)
            return mask
        
        else:
            return mask
    # different from middle_mf and single fidelity
    def get_parents(self, backward = False):
        super().get_parents(backward)
        #if self.done, we know the last action taken was the fidelity
        if self.done:
            if (self.eos == False) and (self.fid is not None) and (self.seq[-1] == self.token_eos):
                parents_a = [self.fid]
                parents = [(self.seq, None)]
                if backward:
                    #fid and seq updated in backward_sample():
                    self.done = False
                    self.eos = True
                return parents, parents_a

            else:
                raise TypeError("Not good ending of sequence")
        #if self.eos, we know the last action taken was eos
        elif self.eos:
            if (self.seq[-1] == self.token_eos) and self.done == False and (self.fid is None):
                parents_a = [self.token_eos]
                parents = [(self.seq[:-1], None)]
                if backward:
                    self.eos = False
                return parents, parents_a 
            else:
                raise TypeError("not good last eos action")
        
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if self.seq[-len(a): ] == list(a):
                    parents.append((self.seq[:-len(a)], None))
                    actions.append(idx)
            
            return parents, actions
    # Like get_parents, step() is nearly the same as in single fidelity, but we add the update of the fidelity at the end
    def step(self, action):
        super().step(action)
        valid = False
        seq = self.seq
        seq_len = len(seq)

        if self.eos:
            #if self.eos, we perform a sanity check: the action must be a fidelity choice
            if not(self.done) and (action[0] in range(self.total_fidelities)):
                if seq_len - 1 >= self.min_seq_len and seq_len - 1 <= self.max_seq_len: #-1 for the eos action
        
                    valid = True
                    next_seq = seq
                    next_fid = action[0]
                    self.eos = False
                    self.done = True
                    self.seq = next_seq
                    self.fid = next_fid
                    self.n_actions_taken += 1
                    self.last_action = action[0]
                    return next_seq, action, valid
                else:
                    raise TypeError("constrain min/max len")
            else:
                raise TypeError("problem eos / done")
       
        elif (action == [self.token_eos]) and (self.done == False) and (self.eos == False):
            if seq_len >= self.min_seq_len and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                next_fid = None
                self.eos = True
                self.n_actions_taken += 1
                self.seq = next_seq
                self.fid = next_fid
                self.last_action = self.token_eos
                return next_seq, action, valid
            else:
                raise TypeError("action eos and done and eos state pb")
        

        
        elif self.eos == False and not(action == [self.token_eos]):
            if action in list(map(list, self.action_space)) and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                next_fid = None
                self.n_actions_taken += 1
                self.seq = next_seq
                self.fid = next_fid
                self.last_action = action[0]
                return next_seq, action, valid
        
        elif self.done == True:
            valid = False
            return None, None, valid
        
        else:
            raise TypeError("invalid action to take")
    
    def acq2reward(self, acq_values):
        min_reward = 1e-20
        true_reward = np.clip(acq_values, min_reward, None)
        customed_af = lambda x: x #to favor the higher rewards in a more spiky way, can be customed
        exponentiate = np.vectorize(customed_af)
        return exponentiate(true_reward)
    # The reward is distored : 
    # 1. It is computed for states at self.eos and not self.done
    # 2. It is equal to the sum of the rewards a(x,m) for all fidelities m available
    # 3. This is done after conversion to the right format with a specific method added in acquisition function
    # NB : this is only used to compute the rewards, which are backpropagated : imposing flows.
    def get_reward(self, states, eos):
        rewards = np.zeros(len(eos), dtype = float)
        final_states = [s for s, e in zip(states, eos) if e]

        inputs_af_base = [self.manip2base(final_state) for final_state in final_states]

        final_rewards = self.acq.get_sum_reward_batch(inputs_af_base).view(len(final_states)).cpu().detach().numpy()

        final_rewards = self.acq2reward(final_rewards)

        eos = np.array(eos)
        
        rewards[eos] = final_rewards

        return rewards
    # Specific function to chose the fidelity at the end according to the imposed flows
    # It calls a method in acquisistion function as well, like get_rewards
    def get_logits_fidelity(self, states):
        inputs_base = [self.manip2base(state_eos) for state_eos in states]
        logits = self.acq.get_logits_fidelity(inputs_base)
        #TODO : maybe acq2rewards should be called here.
        return logits
        
    def base2manip(self, state):
        seq_base = state[0]
        fid = state[1]
        seq_manip = np.concatenate((seq_base, [self.token_eos]))
        return (seq_manip, fid)
    
    def manip2base(self, state):
        seq_manip = state[0]
        fid = state[1]
        if seq_manip[-1] == self.token_eos:
            seq_base = seq_manip[:-1]
            return (seq_base, fid)
        else:
            raise TypeError

    
