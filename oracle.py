import numpy as np
import pandas as pd
#for transition functions base2oracle, specific to each oracle
import torch
from torch import nn
import torch.nn.functional as F
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod

'''
Oracle Wrapper, callable in the AL pipeline, with key methods
'''
class Oracle:
    '''
    Generic Class for the oracle. 
    The different oracles (can be classes (MLP-toy oracle eg) or just a single function calling another annex program)
    can be called according to a config param in the method score
    '''
    def __init__(self, config):
        self.config = config
        #path to load and save the scored data
        self.path_data = config.path.data_oracle
        self.init_oracle()
    
    def init_oracle(self):
        #calling the relevant oracle. All oracles should follow the model of OracleBase
        if self.config.oracle.main == "mlp":
            self.oracle = OracleMLP(self.config)
        elif self.config.oracle.main == "toy":
            self.oracle = OracleToy(self.config)
        else:
            raise NotImplementedError

    def initialize_dataset(self, save = True, return_data = False):
        #the method to initialize samples in the BASE format is specific to each oracle for now. It can be changed.
        #the first samples are in the "base format", so as to be directly saved as such and sent for query (base2oracle transition) 

        samples = self.oracle.initialize_samples_base()

        data = {}
        data["samples"] = samples
        data["energies"] = self.score(samples)
        #print("data initialized", data)

        if save:
            np.save(self.path_data, data)
        if return_data:
            return data
       
    
    def score(self, queries):
        '''
        Calls the specific oracle (class/function) and apply its "get_score" method on the dataset
        '''
        scores = self.oracle.get_score(queries)
        return scores

    def update_dataset(self, queries, energies):
        dataset = np.load(self.path_data, allow_pickle = True)
        dataset = dataset.item()
        dataset["samples"] += queries
        dataset["energies"] += energies
        np.save(self.path_data, dataset)
        return



#For generalization purposes, the oracles are outside the main class of oracle
#First, general a general wrapper for oracles. All the other files.py follow the same formalism for modularity 
'''
BaseClass model for all other oracles
'''
class OracleBase:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def initialize_samples_base(self):
        pass

    @abstractmethod
    def base2oracle(self, state):
        '''
        Transition from base format (format of the queries and stored data) to a format callable by the oracle
        '''
        pass
    
    @abstractmethod
    def get_score(self, queries):
        '''
        - Transforms the queries to a good format with base2oracle
        - Calls the oracle on this data and return it in a good format --> for the proxy to train on.
        '''
        pass


'''
Different Oracles Implemented
'''

class OracleMLP(OracleBase):
    def __init__(self, config):
        super().__init__(config)
        #parameters useful for the format of this custom trained MLP
        self.dict_size = self.config.env.dict_size 
        self.max_len_mlp = 40 #this MLP was trained with seqs of len max 40
        self.path_oracle_mlp = self.config.path.model_oracle_MLP
        self.device = torch.device(self.config.device)
        self.total_fidelities = self.config.env.total_fidelities
    
    def initialize_samples_base(self):
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len
        self.init_len = self.config.oracle.init_dataset.init_len
        self.random_seed = self.config.oracle.init_dataset.seed
        np.random.seed(self.random_seed)
        
        #random samples in base format
        samples = []
        for i in range(self.min_len, self.max_len + 1):
            #in order to have more long samples, we generate more random samples of longer sequences : it is arbitrary so far
            samples.extend(np.random.randint(0, self.dict_size, size = (int(self.init_len * i), i))) 
        
        np.random.shuffle(samples)
        samples = samples[:self.init_len]

        full_samples = []
        for molecule in samples:
            full_samples += [[molecule, np.random.randint(0, self.total_fidelities)]]

        return full_samples

    def base2oracle(self, state):
        seq_array, fidelity = state[0], state[1]
        initial_len = len(seq_array)
        #transform the array as a tensor
        seq_tensor = torch.from_numpy(seq_array)
        #perform one hot encoding on it
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size + 1) #+1 for eos token
        seq_ohe = seq_ohe.reshape(1, -1).float()
        #adding the end of sequence token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()

        oracle_input = torch.cat((seq_ohe, eos_ohe), dim = 1)
        #adding 0-padding until max_len
        number_pads = self.max_len_mlp - initial_len
        if number_pads:
            padding = torch.cat([torch.tensor([0] * (self.dict_size + 1))] * number_pads).view(1, -1)
            oracle_input = torch.cat((oracle_input, padding), dim = 1)
        
        return (oracle_input.to(self.device)[0], fidelity)

    def get_score(self, queries):
        '''
        Input : list of arrays=seqs
        Outputs : list of floats=energies
        '''
        #list of tensors in ohe format
        list_queries =  list(map(self.base2oracle, queries))
        # #turn this into a single big tensor to call the MLP once
        # tensor4oracle = torch.stack(list4oracle).view(len(queries), -1)
        df_queries = pd.DataFrame(list_queries, columns = ["seq_oracle", "fidelity"])
        df_energies = np.zeros(len(df_queries.index))

        for fidelity in range(0, self.total_fidelities):
            #we filter the queries per fidelity to call a specific MLP on each
            indexes = df_queries.index[df_queries["fidelity"] == fidelity].tolist()
            sub_samples = df_queries["seq_oracle"][indexes].reset_index(drop = True).tolist()
            
            
            #load self.model MLP - fidelity
            self.load_MLP(fidelity)

            
            #format the sub_samples and call the self.model
            inputs_model = torch.stack(sub_samples).view(len(sub_samples), -1)
            sub_energies = self.model(inputs_model)

            for i, index in enumerate(indexes):
                df_energies[index] = sub_energies[i]
        
        return df_energies.tolist()

    def load_MLP(self, fidelity):
        file_path = self.path_oracle_mlp + "_" + str(fidelity)
        
        if os.path.exists(file_path):
            self.model = MLP(fidelity)
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

        else:
            raise NotImplementedError


class OracleToy(OracleBase):
    def __init__(self, config):
        super().__init__(config)
        self.total_fidelities = config.env.total_fidelities
    
    def initialize_samples_base(self):
        self.dict_size = self.config.env.dict_size
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len

        self.init_len = self.config.oracle.init_dataset.init_len
        self.random_seed = self.config.oracle.init_dataset.seed
        np.random.seed(self.random_seed)
        
        #random samples in base format
        samples = []
        for i in range(self.min_len, self.max_len + 1):
            #in order to have more long samples, we generate more random samples of longer sequences : it is arbitrary so far
            samples.extend(np.random.randint(0, self.dict_size, size = (int(self.init_len * i), i))) 
        
        np.random.shuffle(samples)
        samples = samples[:self.init_len]

        full_samples = []
        for molecule in samples:
            full_samples += [[molecule, np.random.randint(0, self.total_fidelities)]]

        return full_samples

    def base2oracle(self, state):
        '''
        Input : seq, fid
        Output : seq, fid
        '''

        return state

    def get_score(self, queries):
        dico_std = {
            0: 1,
            1 : 0.5,
            2 : 0,
        }
        
        def toy_function(state):
            seq = state[0]
            fid = state[1]
            mean = float(np.count_nonzero(seq == 0))
            std = dico_std[int(fid)]
            result = np.random.normal(mean, std)
            if result < 0:
                return 0
            else:
                return result
        
        return list(map(toy_function, queries))

        


### Diverse Models of Oracle for now
'''
ORACLE MODELS ZOO
'''
###

class Activation(nn.Module):
    def __init__(self, activation_func = "relu"):
        super().__init__()
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "gelu":
            self.activation = F.gelu

    def forward(self, input):
        return self.activation(input)


#This MLP was specifically trained on a special dataset
class MLP(nn.Module):
    def __init__(self, fidelity):
        super().__init__()

        self.init_architecture_params(fidelity)
        self.init_layer_depth = int(
                (40 + 1) * (4 + 1)
            ) 

        # build input and output layers
        self.initial_layer = nn.Linear(
            int(self.init_layer_depth), self.filters
        )
        self.activation1 = Activation("gelu")
        self.output_layer = nn.Linear(self.filters, 1, bias=False)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.dropouts = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters, self.filters))
            self.activations.append(Activation("gelu"))
            self.dropouts.append(nn.Dropout(p=0.1))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        self.dropouts = nn.ModuleList(self.dropouts)

    def init_architecture_params(self, fidelity):

        if fidelity == 0:
            self.layers = 2
            self.filters = 128
        
        elif fidelity == 1:
            self.layers = 2 
            self.filters = 256
        
        elif fidelity == 2:
            self.layers = 4
            self.filters = 256
 
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.activation1(
            self.initial_layer(x)
        )  # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        y = self.output_layer(x)

        return y

