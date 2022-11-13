import numpy as np
import itertools
from abc import abstractmethod

class Env:
    '''
    Env wrapper class that initializes the specific environment precised in the config
    '''
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq.acq
        self.init_env()
    
    def init_env(self):
        if self.config.env.main == "aptamers": #the only environment currently implemented
            self.env = EnvAptamers(self.config, self.acq)
        else:
            raise NotImplementedError
    

class EnvBase:
    '''
    Base class from which all the environments derive
    '''
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
        pass
 
    @abstractmethod
    def get_mask(self):
        pass
    
    @abstractmethod
    def get_parents(self, backward = False):
        pass
    
    @abstractmethod
    def step(self,action):
        pass
    
    @abstractmethod
    def acq2rewards(self, acq_values):
        pass
    
    @abstractmethod
    def get_reward(self, states, done):
        pass


class EnvAptamers(EnvBase):
    '''
    Environment for DNA Aptamers
    0 : A, 1 : C, 2 : T, 4 : G
    5 : EOS
    6 and more : the different discrete fidelities.
    Can be adapted for more complex actions (AA, AT, CG)
    '''
    def __init__(self, config, acq) -> None:
        super().__init__(config, acq)

        #Specifig to DNA sequences
        self.max_seq_len = self.config.env.max_len
        self.min_seq_len = self.config.env.min_len
        self.max_word_len = self.config.env.max_word_len
        self.min_word_len = self.config.env.min_word_len
        self.n_alphabet = self.config.env.dict_size

        #Fidelities
        #TODO : as the fidelity is an action that the updates the env.state, it is a crucial parameter
        self.total_fidelities = self.config.env.total_fidelities

        #Using the conventions to number the actions
        #TODO : we have to discuss how we represent the fidelities in the space of actions.
        #So far, I modified the method get_action_space() so that it gives a dictionnary that distinguishes the dna action / eos action / fidelitiy action
        self.action_space = self.get_action_space() #dictionnary with keys : "dna_action", "eos_action", "fidelity_action"
        #The following initializations are just utilities for sanity checks.
        self.last_dna_action = self.action_space['dna_action'][-1][0] #from 0 to last_dna_action : DNA Actions
        self.eos_token = self.action_space['eos_action'][0][0] #End of sequence action = Last DNA action_id + 1
        self.first_fidelity_action = self.action_space['fidelity_action'][0][0] #First fidelity action = Eos action + 1

        #Utils to convert the fidelities to actions (0, the worst fidelity corresponds to 0 + self.first_fidelity_action in terms of action number)
        #TODO : we must distinguish : 
        #1. The number of the fidelity action : here the first fidelity (0) is action number token_eos + 1 = token_last_dna_action + 2
        #2. The actual fidelity number and how it updates the state.
        #For example, if the policy network gives the action number token_eos + 1 = token_last_dna_action + 2, it would update the fidelity state from None to 0 (conventions we can discuss)
        self.fid_action2state = lambda x : x - self.first_fidelity_action
        self.fid_state2action = lambda x : x + self.first_fidelity_action

        self.env_class = EnvAptamers
        self.init_env()
   
    def create_new_env(self, idx):
        env = EnvAptamers(self.config, self.acq)
        env.init_env(idx)
        return env
    
    def init_env(self, idx=0):
        super().init_env(idx)
        self.seq = np.array([])
        #TODO : adding self.fid, complementary to self.seq to constitute the self.state
        self.fid = None
        self.n_actions_taken = 0
        #TODO : in addition to self.done, we have to add self.eos.
        #Important : self.eos is True if and only if (the eos action has just been chosen AND the fidelity is still None)
        #self.done and self.eos can never be true at the same time. They are False at first, and then when eos action is chosen, self.eos = True if self.fid = None and self.done = True directly otherwise
        #Remark : self.eos is not indispensable in itself (just like self.done isn't because both can be hard coded, we know when something is done or eos based on their seq / fid in manip format)
        #But self.eos is a useful state so as to know what was the previous action and what is possible. 
        #Besides, it is quite important if we want to integrate full_MVP_mf in the same code.
        self.done = False
        self.eos = False
        self.id = idx
        self.last_action = None
    #Defining self.state would have been tedious, it is easier to update separately self.seq and self.fid
    #In gflownet.py, we access the whole state of en env with the self.get_state() method
    def get_state(self):
        return (self.seq, self.fid)

    #TODO : this method now outputs a dictionnary of actions, to better understand the structure of their representation 
    def get_action_space(self):
        '''
        Makes the link between the number of the action (ie integer index in the list dico_actions["all_action"]) and the actual action [A] or [Eos]
        NB : action eos identifies with its integer index by construction. Whereas action of index 0 could be [00] ie [AA] in some settings.
        '''
        dico_actions = {}

        #DNA ACTIONS
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len +1)
        alphabet = [a for a in range(self.n_alphabet)]
        dna_actions = []
        for r in valid_wordlens:
            actions_r = [list(el) for el in itertools.product(alphabet, repeat = r)]
            dna_actions += actions_r
        dico_actions["dna_action"] = dna_actions

        #EOS ACTIONS
        eos_action = len(dna_actions)
        dico_actions["eos_action"] = [[eos_action]]

        #FIDELITY ACTIONS
        dico_actions["fidelity_action"] = [[i] for i in range(eos_action +1, eos_action + 1 + self.total_fidelities)]

        #ALL ACTIONS
        dico_actions["all_action"] = dico_actions["dna_action"] + dico_actions["eos_action"] + dico_actions["fidelity_action"]

        return dico_actions


    
    def get_mask(self):
        super().get_mask()

        mask = [1] * (len(self.action_space["all_action"]))

        #EXTREME CASES : done or eos
        if self.done:
            return [0 for _ in mask]
        #One of the interests of introducing self.eos : if self.eos = True, we know that we just chose the eos action and we have to chose a fidelity before prompting self.done = True at next action
        elif self.eos:
            assert self.fid == None and self.done == False
            mask[:self.first_fidelity_action] = [0 for _ in range(self.first_fidelity_action)]
            return mask

        else:          
            #FIDELITY EXAMINATION
            if self.fid != None:
                #we can not chose another fidelity
                mask[self.first_fidelity_action:] = [0 for _ in range(self.total_fidelities)]
            
            #Minimal length
            assert self.min_seq_len < self.max_seq_len
            len_seq = len(self.seq)
            assert len_seq <= self.max_seq_len
            if len_seq < self.min_seq_len :
                #eos forbidden
                mask[self.eos_token] = 0
                return mask
            if len_seq >= self.max_seq_len:
                #TODO : checking len_seq >= self.max_seq_len is okay if we chose on [A, C, T, G] action at a time, which is the case right now.
                # When we'll allow [AA, AT, AG, ...] action to be sampled, this will not work anymore
                #the only possible action is the EOS or the choice of the fidelity
                mask[:self.eos_token] = [0 for _ in range(self.eos_token)] #fidelity already forbidden if fid chosen
                return mask
            
            return mask

    #TODO : this would have to be modified cf main-new-al, but the rules are intuitive for multifidelity as well
    #Please note that we are not in the autoregressive case anymore ! That is interesting in terms of flows
    #Please also note that there is not backward = False option in the arguments.
    #Indeed, we can only update env.done and env.eos AFTER chosing which parents we chose (several parents are possible !)
    def get_parents(self):
        parents = []
        parents_a = []

        if self.done:
            assert self.fid != None
            #Choice fid
            parents_a += [self.fid_state2action(self.fid)]
            parents += [(self.seq, None)]
            #Choice eos
            assert self.seq[-1] == self.eos_token
            parents_a += [self.eos_token]
            parents += [(self.seq[:-1], self.fid)]

            return parents, parents_a
        
        elif self.eos:
            assert self.seq[-1] == self.eos_token
            assert self.fid == None
            parents_a += [self.eos_token]
            parents += [(self.seq[:-1], self.fid)]
            return parents, parents_a

        else:
            #DNA action
            for idx, a in enumerate(self.action_space["dna_action"]):
                if self.seq[-len(a): ] == a:
                    parents += [(self.seq[:-len(a)], self.fid)]
                    parents_a += [idx]
            
            #if fidelity was chosen, it is another possible action
            if self.fid != None:
                parents_a += [self.fid_state2action(self.fid)]
                parents += [(self.seq, None)]
            
            return parents, parents_a
    #This replaces the previous backward = True option in get_parents.
    #update_status() is only called in gflownet.py for backward_sample() (to get offline data) : we update the state details after chosing it with a backward policy
    def update_status(self):
        seq = self.seq
        fid = self.fid
        #As previously mentioned, self.done and self.eos can be hardcoded. Defining self.eos and self.done is just to avoid adding an "if" verification each time we want to know more about the state
        if len(seq) == 0:
            self.done = False
            self.eos = False
        elif seq[-1] == self.eos_token and fid != None:
            self.done = True
            self.eos = False
        elif seq[-1] == self.eos_token and fid == None:
            self.done = False
            self.eos = True
        else:
            self.done = False
            self.eos = False


    def step(self, action):
        '''
        action = a interger, not a list !
        '''

        valid = False
        seq = self.seq
        fid = self.fid
        seq_len = len(seq)

        if self.done:
            raise TypeError("can't perform action when env.done")
        #If you are self.eos = True, you can ony chose the fidelity next to complete your candidate definition
        elif self.eos:
            assert fid == None
            if action >= self.first_fidelity_action:
                valid = True
                self.fid = self.fid_action2state(action)
                self.seq = seq
                self.eos = False
                self.done = True
                self.n_actions_taken += 1
                self.last_action = action
                return valid

            else:
                raise TypeError('only possible action when env.eos if fidelity choice')
        
        else:
            #EOS
            if action == self.eos_token:
                #You cannot chose eos action if your seq is to small (we force the choice or eos anyway when the max len is reached)
                assert self.min_seq_len < self.max_seq_len
                assert seq_len >= self.min_seq_len and seq_len <= self.max_seq_len
                valid = True
                self.fid = fid
                self.seq = np.append(seq, self.action_space["eos_action"])
                #Updating self.done and self.eos. Note that this can also be done by calling self.update_status() but we hard coded it here again ...
                if fid != None:
                    self.eos = False
                    self.done = True
                elif fid == None:
                    self.eos = True
                    self.done = False
                self.n_actions_taken += 1
                self.last_action = action
                return valid    

            #FID
            elif action >= self.first_fidelity_action:
                #All these assertion are sanity checks that our rules have been enforced
                assert fid == None and self.done == False
                valid = True
                self.fid = self.fid_action2state(action)
                self.seq = seq
                if self.eos:
                    self.eos = False
                    self.done = True
                elif self.eos == False:
                    self.eos = False
                    self.done = False
                self.n_actions_taken += 1
                self.last_action = action
                return valid

            #DNA 
            elif action in range(0, self.last_dna_action + 1):
                assert self.eos == False and self.done == False
                assert seq_len < self.max_seq_len
                valid = True
                self.fid = fid
                dna_aptamers = self.action_space["dna_action"][action] #it is a list
                self.seq = np.append(seq, dna_aptamers)
                self.done = False
                self.eos = False
                self.n_actions_taken += 1
                self.last_action = action
                return valid
            
            else:
                raise TypeError('not a valid action')


    def acq2reward(self, acq_values):
        min_reward = 1e-10
        true_reward = np.clip(acq_values, min_reward, None)
        customed_af = lambda x: x #to favor the higher rewards in a more spiky way, can be customed
        exponentiate = np.vectorize(customed_af)
        return exponentiate(true_reward)

    def get_reward(self, states, done):
        rewards = np.zeros(len(done), dtype = float)
        final_states = [s for s, d in zip(states, done) if d]
        inputs_af_base = [self.manip2base(final_state) for final_state in final_states]
        final_rewards = self.acq.get_reward_batch(inputs_af_base).view(len(final_states)).cpu().detach().numpy()
        final_rewards = self.acq2reward(final_rewards)
        done = np.array(done)    
        rewards[done] = final_rewards
        return rewards
    #TODO : based on what we discussed, we might need to gather all transitions that are environment specific in the environment ...    
    def base2manip(self, state):
        seq_base = state[0]
        fid = state[1]
        seq_manip = np.concatenate((seq_base, [self.eos_token]))
        return (seq_manip, fid)
    
    def manip2base(self, state):
        seq_manip = state[0]
        fid = state[1]
        assert seq_manip[-1] == self.eos_token
        seq_base = seq_manip[:-1]
        return (seq_base, fid)


    
