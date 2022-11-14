import torch
import torch.nn.functional as F
from torch.utils import data
from torch.distributions.categorical import Categorical
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pd
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod

#Utils function for the whole file, fixed once and for all
global tf_list, tl_list, to, _dev
_dev = [torch.device("cpu")]
tf_list = lambda x: torch.FloatTensor(x).to(_dev[0])
tl_list = lambda x: torch.LongTensor(x).to(_dev[0])
to = lambda x : x.to(_dev[0])

def set_device(dev):
    _dev[0] = dev

'''
GFlownet Wrapper, callable with key methods
'''
class GFlowNet:
    def __init__(self, config, logger, env):
        self.config = config
        self.logger = logger
        self.env = env.env

        self.init_gflownet()
    
    def init_gflownet(self):
        #for now there is only one GFlowNet - configuration called A, but there will be several with multifidelity (several possibilities of sampling m)
        self.gflownet = GFlowNet_A(self.config, self.logger, self.env)
        
    
    def train(self):
        self.gflownet.train()
    
    def sample(self, nb_queries):
        queries = self.gflownet.sample_queries(nb_queries)
        return queries
        
        
'''
GFlowNet Objects
'''
class GFlowNetBase:
    def __init__(self, config, logger, env, load_best_model = False):
        self.config = config
        self.logger = logger
        self.env = env

        self.device = self.config.device
        set_device(self.device)
 
        #set loss function, device, buffer ...
        self.path_model = self.config.path.model_gfn


        if self.config.gflownet.loss.function == "flowmatch":
            self.loss_function = self.flowmatch_loss
        elif self.config.gflownet.loss.function == "trajectory_balance":
            self.loss_function = self.trajectory_balance
        else:
            raise NotImplementedError  

        self.model_class = NotImplemented
        self.model = NotImplemented
        self.best_model = NotImplemented
        self.sampling_model = NotImplemented
        self.get_model_class()
        if load_best_model:
            self.make_model(best_model=True)

        self.load_hyperparameters()

        self.loginf = tf_list([1e6])
        self.buffer = Buffer(self.config)
            
    @abstractmethod
    def load_hyperparameters(self):
        self.flowmatch_eps = tf_list([self.config.gflownet.loss.flowmatch_eps])
        self.rng = np.random.default_rng(self.config.gflownet.sampling.seed)
        self.random_action_prob = self.config.gflownet.sampling.random_action_prob
        self.temperature = self.config.gflownet.sampling.temperature
        self.pct_batch_empirical = self.config.gflownet.training.pct_batch_empirical
        self.training_steps = self.config.gflownet.training.training_steps
        self.view_progress = self.config.gflownet.view_progress
        self.batch_size = self.config.gflownet.training.batch_size
        self.ttsr = self.config.gflownet.training.ttsr
        self.clip_grad_norm = self.config.gflownet.training.clip_grad_norm


    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def make_model(self, new_model = False, best_model = False):
        '''
        Initializes the GFN policy network (separate class for MLP for now), and load the best one (random if not best GFN yet)
        '''
        def make_opt(params, config):
            params = list(params)
            if not len(params):
                return None
            if config.gflownet.training.opt == "adam":
                opt = torch.optim.Adam(
                    params,
                    config.gflownet.training.learning_rate, #to incorpore in load_hp
                    betas=(
                        config.gflownet.training.adam_beta1,
                        config.gflownet.training.adam_beta2,
                    ),
                )

            elif config.gflownet.training.opt == "msgd":
                opt = torch.optim.SGD(
                    params,
                    config.gflownet.training.learning_rate,
                    momentum=config.gflownet.training.momentum,
                )

            else:
                raise NotImplementedError
            return opt

        def make_lr_scheduler(optimizer, config):
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.gflownet.training.lr_decay_period,
                gamma=config.gflownet.training.lr_decay_gamma,
            )
            return lr_scheduler
        
        if new_model:
            self.model = self.model_class(self.config)
            self.opt = make_opt(self.model.parameters(), self.config)

            if self.device == "cuda":
                self.model.cuda()  # move net to GPU
                for state in self.opt.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            
            self.lr_scheduler = make_lr_scheduler(self.opt, self.config)
            print("self.model new model created ! ")

        if best_model:
            path_best_model = self.path_model
            if os.path.exists(path_best_model):
                checkpoint = torch.load(path_best_model)
                self.best_model = self.model_class(self.config)
                self.best_model.load_state_dict(checkpoint["model_state_dict"])
                self.best_opt = make_opt(self.best_model.parameters(), self.config)
                self.best_opt.load_state_dict(checkpoint["optimizer_state_dict"])
                print("self.best_model best gfn loaded") 
  
            else:
                print("the best previous model could not be loaded, random gfn for best model")
                self.best_model = self.model_class(self.config)
                self.best_opt = make_opt(self.best_model.parameters(), self.config)
                self.best_lr_scheduler = make_lr_scheduler(self.best_opt, self.config)
            
            if self.device == "cuda":
                self.best_model.cuda()  # move net to GPU
                for state in self.best_opt.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()   

            self.best_lr_scheduler = make_lr_scheduler(self.best_opt, self.config)  

    # Exactly the same as in single_fidelity, because called in the same context
    @abstractmethod
    def forward_sample(self, envs, policy, temperature = 0):
        """
        Performs a forward action on each environment of a list.

        Args
        ----
        env : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        policy : string
            - model: uses self.model to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if policy="model"

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        if temperature == 0:
            temperature = self.temperature

        if self.sampling_model == NotImplemented:
            print("weird, the sampling model should be initialized already")
            self.sampling_model = self.best_model
        
        self.sampling_model.eval()
        #instead of calling env.get_state(), we might define directly env.state = (seq, fid) to match env.state called in single fidelity
        states = [env.get_state() for env in envs]
        states_ohe = torch.stack(list(map(self.manip2policy, states))).view(len(states), -1)
        masks = tf_list([env.get_mask() for env in envs])

        if policy == "model":
            with torch.no_grad():
                action_logits = self.sampling_model(states_ohe)
            action_logits /= temperature
        elif policy == "uniform":
            action_logits = tf_list(np.ones((len(states), len(self.env.action_space) + 1))) #actions + eos
        
        elif policy == "mixt":
            random_probas = [self.rng.uniform() for _ in range(len(envs))]
            envs_random = [env for i, env in enumerate(envs) if random_probas[i] <= self.random_action_prob]
            envs_no_random = [env for i, env in enumerate(envs) if random_probas[i] > self.random_action_prob]

            if envs_random: 
                envs_random, actions_random, valids_random = self.forward_sample(envs_random, policy = "uniform", temperature = self.temperature)
                # print('random', actions_random)
                # print(type(valids_random))

            else:
                envs_random, actions_random, valids_random = [], to(torch.tensor([])), ()

            if envs_no_random:
                envs_no_random, actions_no_random, valids_no_random = self.forward_sample(envs_no_random, policy = "model", temperature=self.temperature)
                # print('no random', actions_no_random)
                # print(valids_no_random)
                
            else:
                envs_no_random, actions_no_random, valids_no_random = [], to(torch.tensor([])), ()
            final_envs = envs_random + envs_no_random
            final_actions = torch.cat((actions_random, actions_no_random), dim = 0)
            final_valids = valids_random + valids_no_random
            
            return final_envs, final_actions, final_valids
        
        else:
            raise NotImplemented
        
        
        action_logits = torch.where(masks == 1, action_logits, - self.loginf)
        if all(torch.isfinite(action_logits).flatten()):
            actions = Categorical(logits=action_logits).sample()
        else:
            raise ValueError("Action could not be sampled from model!")
        
        assert len(envs) == actions.shape[0]
  
        # Execute actions
        _, _, valids = zip(*[env.step([action.tolist()]) for env, action in zip(envs, actions)])
        
        return envs, actions, valids

    #There would be a small difference with single_fidelity, with the specific annoying case of imposing the flows instead of calling model when env.done == True
    @abstractmethod
    def backward_sample(self, env, policy, temperature = 0):
        if temperature == 0:
            temperature = self.config.gflownet.sampling.temperature

        parents, parents_a = env.get_parents(backward = True) #remplacer state par seq

        parents_ohe = torch.stack(list(map(self.manip2policy, parents))).view(len(parents), -1)

        if policy == "model":
            self.best_model.eval()
            with torch.no_grad():
                #TODO : maybe call the imposed logits here if env.done == True ... for the fidelity choice
                action_logits = self.best_model(parents_ohe)[
                    torch.arange(len(parents)), parents_a
                ]

            if all(torch.isfinite(action_logits).flatten()):
                action_idx = Categorical(logits=action_logits).sample().item()
         
            else:
                raise ValueError("Action could not be sampled from model!")
        
        elif policy == "uniform":
            action_idx = self.rng.integers(low=0, high=len(parents_a))
        else:
            raise NotImplemented

        state = parents[action_idx]
        #That would change with env.state = (seq, fid) to match single fidelity exactly
        env.seq = state[0] #state ou fonction set state
        env.fid = state[1]
        env.last_action = parents_a[action_idx]

        return env, parents, parents_a

    @abstractmethod
    def get_training_data(self, batch_size):
        '''
        Calls the buffer to get some interesting training data
        Performs backward sampling for off policy data and forward sampling
        Calls the utils method forward sampling and backward sampling
        '''
        pass
    

    @abstractmethod
    def flowmatch_loss(self, data):
        pass
    

    @abstractmethod
    def trajectory_balance(self, data):
        pass
    
    # Train function never changes !
    @abstractmethod
    def train(self):
        all_losses = []

        self.make_model(new_model=True, best_model=True)
        self.model.train()

        for it in tqdm(range(self.training_steps), disable = not self.view_progress):

            data = self.get_training_data(self.batch_size)

            for sub_it in range(self.ttsr):
                self.model.train()
                loss = self.loss_function(data)
                
                if not torch.isfinite(loss):
                    print("loss is not finite - skipping iteration")

                else:
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)     
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                
                if sub_it == 0:
                    all_losses.append(loss.item())
                 
        #save model
        path = self.path_model
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.opt.state_dict()}, path)
        print("new gfn model saved ! ")
        return
    
    @abstractmethod
    def sample_queries(self, nb_queries):
        '''
        Just performs forward sampling with the trained GFlownet
        '''
        print("we sample for query !")
        self.make_model(best_model=True)
        self.sampling_model = self.best_model

        batch = []
        envs = [self.env.create_new_env(idx = idx) for idx in range(nb_queries)]

        while envs:
            envs, actions, valids = self.forward_sample(
                envs, policy="model", temperature= self.temperature)
                
                
            remaining_envs = []
            for env in envs:
                if env.done:
                    batch.append(
                            self.env.manip2base(env.state)
                        )
                else:
                    remaining_envs.append(env)
            envs = remaining_envs
        
        return batch

    @abstractmethod
    def manip2policy(self, state):
        pass


###-----------SUBCLASS OF SPECIFIC GFLOWNET-----------------------
class GFlowNet_A(GFlowNetBase):
    def __init__(self, config, logger, env):
        super().__init__(config, logger, env)

    def load_hyperparameters(self):
        super().load_hyperparameters()
    
    def get_model_class(self):
        if self.config.gflownet.policy_model == "mlp":
            self.model_class = MLP
        else:
            raise NotImplementedError
        
    def make_model(self, new_model=False, best_model=False):
        super().make_model(new_model, best_model)
        

    def forward_sample(self, envs, policy="model", temperature=0):
        return super().forward_sample(envs, policy, temperature)


    def backward_sample(self, env, policy = "model", temperature = 0):
        return super().backward_sample(env, policy, temperature)


    def forward_sample_eos(self, envs, temperature = 0):
        """
        Performs a forward action on each environment of a list.

        Args
        ----
        env : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        policy : string
            - model: uses self.model to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if policy="model"

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        if temperature == 0:
            temperature = self.temperature
        
        assert all(env.eos for env in envs) and all(env.seq[-1] == env.token_eos for env in envs)

        states = [env.get_state() for env in envs]

        #call method of env that calls method of acquisition
        fidelity_logits = self.env.get_logits_fidelity(states)


        if all(torch.isfinite(fidelity_logits).flatten()):
            fidelities = Categorical(logits=fidelity_logits).sample()
  
        else:
            raise ValueError("Fidelity could not be sampled from model!")
        
        assert len(envs) == fidelities.shape[0]
  
        # Execute actions
        _, _, valids = zip(*[env.step([fidelity.tolist()]) for env, fidelity in zip(envs, fidelities)])
        
        return envs, fidelities, valids
    
    def get_training_data(self, batch_size):
        super().get_training_data(batch_size)

        batch = []

        envs = [
            self.env.create_new_env(idx)
            for idx in range(batch_size)
        ]

        #OFFLINE DATA
        self.buffer.make_train_test_set()
        offline_samples = int(self.pct_batch_empirical * len(envs))

        for env in envs[:offline_samples]:
            state = self.rng.permutation(self.buffer.train.samples.values)[0]
            state_manip = self.env.base2manip(state)
            env.done = True
            env.eos = False
            env.seq = state_manip[0]
            env.fid = state_manip[1]
            env.last_action = env.fid
            #we don't use last_action_taken



            while len(env.seq) > 0:
                previous_state = env.get_state()
                previous_eos = env.eos
                previous_done = env.done
                previous_mask = env.get_mask()
                
                env, parents, parents_a = self.backward_sample(
                        env,
                        policy="model",
                        temperature=self.temperature
                    )
                #for backward sampling, the last action is updated after
                previous_action = env.last_action 
                seq_ohe = self.manip2policy(previous_state)       
                parents_ohe = torch.stack(
                    list(map(self.manip2policy, parents))
                    )
                
                batch.append(
                    [
                        seq_ohe.unsqueeze(0),
                        tl_list([previous_action]),
                        tf_list([previous_mask]),
                        previous_state,
                        parents_ohe.view(len(parents), -1),
                        tl_list(parents_a),
                        previous_eos,
                        previous_done,
                        tl_list([env.id]*len(parents)),
                        tl_list(
                            [
                                len(previous_state[0]) - 1 if not previous_done
                                else len(previous_state[0])
                            ]
                        )
                    ]
                )


                # print(
                #     [
                #         seq_ohe.unsqueeze(0),
                #         tl_list([previous_action]),
                #         tf_list([previous_mask]),
                #         previous_state,
                #         parents_ohe.view(len(parents), -1),
                #         tl_list(parents_a),
                #         previous_eos,
                #         previous_done,
                #         tl_list([env.id]*len(parents)),
                #         tl_list(
                #             [
                #                 len(previous_state[0]) - 1 if not previous_done
                #                 else len(previous_state[0])
                #             ]
                #         )
                #     ]
                # )
            

            env.done = True


        envs = [env for env in envs if not env.done]
        self.sampling_model = self.best_model
        self.sampling_model.eval()
        
        while envs:
            envs_eos = [env for env in envs if env.eos]
            envs_no_eos = [env for env in envs if not(env.eos)]
            #default : 
            #ENVS NO EOS : CLASSIC
            if len(envs_no_eos):
                envs_no_eos, actions_no_eos, valids_no_eos = self.forward_sample(
                        envs_no_eos,
                        policy="mixt",
                        temperature=self.temperature
                    )
            else:
                envs_no_eos, actions_no_eos, valids_no_eos = [], to(torch.tensor([])), ()

    
            #ENVS EOS : 
            if len(envs_eos):

                envs_eos, actions_eos, valids_eos = self.forward_sample_eos(
                        envs_eos,
                        temperature=self.temperature
                    )
            else:
                envs_eos, actions_eos, valids_eos = [], to(torch.tensor([])), ()


            envs = envs_no_eos + envs_eos
            actions = torch.cat((actions_no_eos, actions_eos), dim = 0)
            valids = valids_no_eos + valids_eos

            # Add to batch
            for env, action, valid in zip(envs, actions, valids):
                if valid:
                    parents, parents_a = env.get_parents()
                    state_ohe = self.manip2policy(env.get_state())
                    parents_ohe = torch.stack(list(map(self.manip2policy, parents)))
                    mask = env.get_mask()
                    batch.append(
                            [
                                state_ohe.unsqueeze(0),
                                tl_list([int(action)]), #don't know why it is a scalar sometime ...
                                tf_list([mask]),
                                env.get_state(),
                                parents_ohe.view(len(parents), -1),
                                tl_list(parents_a),
                                env.eos,
                                env.done,
                                tl_list([env.id] * len(parents)),
                                tl_list([env.n_actions_taken - 1]), #convention, we start at 0
                            ]
                        )
            

                    # print(
                    #         [
                    #             state_ohe.unsqueeze(0),
                    #             tl_list([int(action)]), #don't know why it is a scalar sometime ...
                    #             tf_list([mask]),
                    #             env.get_state(),
                    #             parents_ohe.view(len(parents), -1),
                    #             tl_list(parents_a),
                    #             env.eos,
                    #             env.done,
                    #             tl_list([env.id] * len(parents)),
                    #             tl_list([env.n_actions_taken - 1]), #convention, we start at 0
                    #         ]
                    # )

                    
            envs = [env for env in envs if not env.done]

        
        (
            states,
            actions,
            masks,
            input_reward,
            parents,
            parents_a,
            eos,
            done,
            path_id,
            state_id,
        ) = zip(*batch)
        


        rewards = self.env.get_reward(input_reward, eos)


        rewards = [tf_list([r]) for r in rewards]
        eos = [tl_list([e]) for e in eos]
        done = [tl_list([d]) for d in done]


        batch = list(
            zip(
                states,
                actions,
                masks,
                rewards,
                parents,
                parents_a,
                eos,
                done,
                path_id,
                state_id,
            )
        )

        return batch

    def flowmatch_loss(self, data):
        super().flowmatch_loss(data)
        self.model.train()

        #for inputflow, id of parents
        parents_incoming_flow_idxs = tl_list(
            sum(
                [
                    [i] * len(parents)
                    for i, (_, _, _, _, parents, _, _, _, _, _) in enumerate(data)
                ],
                [],
            )
        )

        seqs, _, masks, rewards, parents, actions, eos, done, _, _ = map(torch.cat, zip(*data))

        #IN FLOW
        q_in = self.model(parents)[torch.arange(parents.shape[0]), actions]
        parents_Qsa = q_in * (1 - done) - self.loginf*done

        in_flow = torch.log(self.flowmatch_eps + to(torch.zeros((seqs.shape[0],))).index_add_(0, parents_incoming_flow_idxs, torch.exp(parents_Qsa)))

        # in_flow = torch.logaddexp(
        #     torch.log(self.flowmatch_eps),
        #     tf_tensor(torch.zeros((seqs.shape[0],))).index_add_(
        #         0, parents_incoming_flow_idxs, parents_Qsa
        #     ),
        # )
        #OUTFLOW
        q_out = self.model(seqs)
        q_out = torch.where(masks ==1, q_out, -self.loginf)
        q_out = torch.logsumexp(q_out, 1)
        q_out = q_out * (1 - done) - self.loginf * done
        child_Qsa = q_out * (1 - eos) - self.loginf*eos

        out_flow = torch.log(self.flowmatch_eps + rewards + torch.exp(child_Qsa))
        
        #out_flow = torch.logaddexp(torch.log(rewards + self.flowmatch_eps), child_Qsa)
        #LOSS
        loss = (in_flow - out_flow).pow(2).mean()
        print("loss gfn", loss.item())
        return loss
    
    def train(self):
        super().train()

    #Classic sampling until eos. And then calling forward_sample_eos to chose the fidelity artificially.
    def sample_queries(self, nb_queries):
        print("we sample for query !")
        self.make_model(best_model=True)
        self.sampling_model = self.best_model

        batch = []
        envs = [self.env.create_new_env(idx = idx) for idx in range(nb_queries)]

        while envs:
            envs_eos = [env for env in envs if env.eos]
            envs_no_eos = [env for env in envs if not(env.eos)]
            #default : 
            #ENVS NO EOS : CLASSIC
            if len(envs_no_eos):
                envs_no_eos, actions_no_eos, valids_no_eos = self.forward_sample(
                        envs_no_eos,
                        policy="mixt",
                        temperature=self.temperature
                    )
            else:
                envs_no_eos, actions_no_eos, valids_no_eos = [], to(torch.tensor([])), ()

    
            #ENVS EOS : 
            if len(envs_eos):

                envs_eos, actions_eos, valids_eos = self.forward_sample_eos(
                        envs_eos,
                        temperature=self.temperature
                    )
            else:
                envs_eos, actions_eos, valids_eos = [], to(torch.tensor([])), ()


            envs = envs_no_eos + envs_eos
            # actions = torch.cat((actions_no_eos, actions_eos), dim = 0)
            # valids = valids_no_eos + valids_eos
                
            remaining_envs = []
            for env in envs:
                if env.done:
                    batch.append(
                            self.env.manip2base(env.get_state())
                        )
                else:
                    remaining_envs.append(env)
            envs = remaining_envs
        
        return batch
                


    def manip2policy(self, state):
        seq_manip = state[0]
        fid = state[1]
        initial_len = len(seq_manip)

        seq_tensor = torch.from_numpy(seq_manip)
        seq_ohe =  F.one_hot(seq_tensor.long(), num_classes=self.env.n_alphabet+1)
        input_policy = seq_ohe.reshape(1, -1).float()

        number_pads = (self.env.max_seq_len + 1 - initial_len)
        if number_pads:
            padding =  torch.cat(
                [torch.tensor([0] * (self.env.n_alphabet + 1))] * number_pads
            ).view(1, -1)
            input_policy = torch.cat((input_policy, padding), dim = 1)
        
        return to(input_policy)[0]






        
        




'''
Utils Buffer
'''

class Buffer:
    '''
    BUffer of data : 
    - loads the data from oracle and put the best ones as offline training data
    - maintains a replay buffer composed of the best trajectories sampled for training
    '''
    def __init__(self, config):
        self.config = config
        self.path_data_oracle = self.config.path.data_oracle

        self.rng = np.random.default_rng(47) #to parametrize
    

    def np2df(self):
        data_dict = np.load(self.path_data_oracle, allow_pickle = True).item()
        seqs = data_dict["samples"]
        energies = data_dict["energies"]
        df = pd.DataFrame(
            {
                "samples" : seqs,
                "energies" : energies,
                "train" : [False] * len(seqs),
                "test" :  [False] * len(seqs)
            }
        )

        return df
    
    def make_train_test_set(self):
        df = self.np2df()
        indices = self.rng.permutation(len(df.index))
        n_tt = int(0.1 * len(indices))
        indices_tt = indices[:n_tt]
        indices_tr = indices[n_tt:]
        df.loc[indices_tt, "test"] = True
        df.loc[indices_tr, "train"] = True

        self.train = df.loc[df.train]
        self.test = df.loc[df.test]






'''
Model Zoo
'''



class Activation(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "gelu":
            self.activation = F.gelu

    def forward(self, input):
        return self.activation(input)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
     
        self.config = config
        act_func = "relu"

        # Architecture
        self.input_max_length = self.config.env.max_len + 1  
        self.input_classes = self.config.env.dict_size + 1  
        self.init_layer_size = int(self.input_max_length * self.input_classes)
        self.final_layer_size = int(self.input_classes)  

        self.filters = 256
        self.layers = 16

        prob_dropout = self.config.gflownet.training.dropout

        # build input and output layers
        self.initial_layer = nn.Linear(
            self.init_layer_size, self.filters
        )  
        self.initial_activation = Activation(act_func)
        self.output_layer = nn.Linear(
            self.filters, self.final_layer_size, bias=False
        )  

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        # self.norms = []
        self.dropouts = []

        for _ in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters, self.filters))
            self.activations.append(Activation(act_func))
            # self.norms.append(nn.BatchNorm1d(self.filters))#et pas self.filters
            self.dropouts.append(nn.Dropout(p=prob_dropout))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        # self.norms = nn.ModuleList(self.norms)
        self.dropouts = nn.ModuleList(self.dropouts)
        return

    def forward(self, x):
        x = self.initial_activation(
            self.initial_layer(x)
        )  
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
            # seq = self.norms[i](seq)
        return self.output_layer(x)
