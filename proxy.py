import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod

'''
GLOBAL CLASS PROXY WRAPPER, callable with key methods
'''
class Proxy:
    def __init__(self, config, oracle, logger):
        self.config = config
        self.logger = logger
        self.oracle2proxy = oracle.oracle2proxy
        self.init_proxy()

    def init_proxy(self):
        if self.config.proxy.model == "mlp":
            self.proxy = ProxyMLP(self.config, self.logger)
        else:
            raise NotImplementedError
    
    def train(self):
        self.data_handler = BuildDataset(self.config, self.proxy, oracle2proxy = self.oracle2proxy)
        self.proxy.converge(self.data_handler)
        return


'''
PROXY OBJECTS
'''    
class ProxyBase:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.device = self.config.device
        self.path_data = self.config.path.data_oracle
        self.path_model = self.config.path.model_proxy  

        #Training Parameters
        self.training_eps = self.config.proxy.training.eps
        self.max_epochs = self.config.proxy.training.max_epochs
        self.history = self.config.proxy.training.history
        assert self.history <= self.max_epochs
        self.dropout = self.config.proxy.training.dropout
        self.batch_size = self.config.proxy.training.training_batch

        #Dataset management
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed

        self.model_class = NotImplemented  #will be precised in child classes

 
    @abstractmethod
    def init_model(self):
        '''
        Initialize the proxy we want (cf config). Each possible proxy is a class (MLP, transformer ...)
        Ensemble methods will be another separate class
        '''
        self.model = self.model_class(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad = True)

    @abstractmethod
    def load_model(self, dir_name = None):
        '''
        will not have to be used normally because the global object ActiveLearning.proxy will be continuously updated
        '''
        if dir_name == None:
            dir_name = self.config.path.model_proxy
        
        self.init_model()
        
        if os.path.exists(dir_name):
            checkpoint = torch.load(dir_name)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.device == "cuda":
                self.model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        
        else:
            raise NotImplementedError
    
    @abstractmethod
    def converge(self, data_handler):
        '''
        will call getDataLoaders/train_batch/ test / checkConvergencce 
        '''
        #we reset the model, cf primacy bias, here we train on more and more data
        self.init_model()

        #for statistics we save the tr and te errors
        [self.err_tr_hist, self.err_te_hist] = [[], []]
        
        #get training data in torch format
        tr, te = data_handler.get_data_loaders()

        self.converged = 0
        self.epochs = 0

        while self.converged != 1:

            if (
                self.epochs > 0
            ):  #  this allows us to keep the previous model if it is better than any produced on this run
                self.train(tr)  # already appends to self.err_tr_hist
            else:
                self.err_tr_hist.append(0)

            self.test(te)


            if self.err_te_hist[-1] == np.min(
                self.err_te_hist
            ):  # if this is the best test loss we've seen
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.path_model,
                )  # we update only the best, not keep the previous ones
                print(
                    "new best found at epoch {}, of value {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )

            # after training at least "history" epochs, check convergence
            if self.epochs >= self.history + 1:
                self.check_convergence()

            if (self.epochs % 10 == 0) and self.config.debug:
                print(
                    "Model epoch {} test loss {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )  

            self.epochs += 1

            # if self.converged == 1:
            #     self.statistics.log_comet_proxy_training(
            #         self.err_tr_hist, self.err_te_hist
            #     )
        # dataset = np.load(self.path_data, allow_pickle = True)
        # dataset = dataset.item()

        # samples = list(map(lambda x: self.base2proxy(x), dataset['samples']))
        # samples = np.array(samples)

        # tr = data.DataLoader(samples, batch_size=len(samples), shuffle=True, num_workers= 0, pin_memory=False)
        # for _, trainData in enumerate(tr):
        #     outputs = self.model(trainData)

        # print(outputs)

    @abstractmethod
    def train(self, tr):
        ''' 
        will call getLoss
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.get_loss(trainData)
            err_tr.append(loss.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.err_te_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
    
    @abstractmethod
    def test(self, te):
        err_te = []
        self.model.eval()
        with torch.no_grad():
            for i, testData in enumerate(te):
                loss = self.get_loss(testData)
                err_te.append(loss.data)
        
        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
    
    @abstractmethod
    def get_loss(self, data):
        inputs = data[0]
        targets = data[1]
        if self.device == "cuda":
            inputs = inputs.cuda()
            targets = targets.cuda()
        output = self.model(inputs.float())
        return F.mse_loss(output[:, 0], targets.float())
        
    @abstractmethod
    def check_convergence(self):
        eps = self.training_eps
        history = self.history
        max_epochs = self.max_epochs

        if all(
            np.asarray(self.err_te_hist[-history + 1 :]) > self.err_te_hist[-history]
        ):  # early stopping
            self.converged = 1  # not a legitimate criteria to stop convergence ...
            print(
                "Model converged after {} epochs - test loss increasing at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )  
            )

        if (
            abs(self.err_te_hist[-history] - np.average(self.err_te_hist[-history:]))
            / self.err_te_hist[-history]
            < eps
        ):
            self.converged = 1
            print( 
                "Model converged after {} epochs - hit test loss convergence criterion at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )

        if self.epochs >= max_epochs:
            self.converged = 1
            print(
                "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )
      
    @abstractmethod
    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data).cpu().detach().numpy()
            return output  

    @abstractmethod
    def base2proxy(self, state):
        pass
    


'''
In the child Classes, the previous abstract methods can be overwritten. In what follows, the minimum is done to precise the proxy, ie
- the Network is precised
- The conversion format is given for its input
'''
class ProxyMLP(ProxyBase):
    def __init__(self, config, logger, init_model = False):
        super().__init__(config, logger)
        self.model_class = MLP
        if init_model:
            self.init_model()

    def init_model(self):
        super().init_model()
            
    def load_model(self, dir_name = None):
        super().load_model(dir_name)
   
    def converge(self, data_handler):
        super().converge(data_handler)

    def train(self, tr):
        super().train(tr)

    def test(self, te):
        super().test(te)

    def get_loss(self, data):
        return super().get_loss(data)
    
    def check_convergence(self):
        super().check_convergence()

    def evaluate(self, data):
        return super().evaluate(data)
  
    def base2proxy(self, state):

        #useful format
        self.dict_size = self.config.env.dict_size
        self.max_len = self.config.env.max_len
        self.total_fidelities = self.config.env.total_fidelities

        seq = state[0]
        fid = state[1]
        initial_len = len(seq)
        #into a tensor and then ohe
        seq_tensor = torch.from_numpy(seq)
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size +1)
        seq_ohe = seq_ohe.reshape(1, -1).float()
        #addind eos token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()

        input_proxy = torch.cat((seq_ohe, eos_ohe), dim = 1)
        #adding 0-padding
        number_pads = self.max_len - initial_len
        if number_pads:
            padding = torch.cat(
                [torch.tensor([0] * (self.dict_size +1))] * number_pads
            ).view(1, -1)
            input_proxy = torch.cat((input_proxy, padding), dim = 1)
        
        #fid
        fid_tensor = torch.tensor([fid])
        fid_tensor = F.one_hot(fid_tensor.long(), num_classes = self.total_fidelities)
        fid_tensor = fid_tensor.reshape(1, -1).float()
        
        input_proxy = torch.cat((input_proxy, fid_tensor), dim = 1)
        return input_proxy.to("cpu")[0] #weird, it always has to be cpu ...


'''
BuildDataset utils
'''
class BuildDataset:
    '''
    Will load the dataset scored by the oracle and convert it in the right format for the proxy with transition.base2proxy
    '''
    def __init__(self, config, proxy, oracle2proxy):
        self.config = config
        self.proxy = proxy
        self.path_data = self.config.path.data_oracle
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed
        
        self.oracle2proxy = oracle2proxy

        self.load_dataset()
    
    
    def load_dataset(self):
        dataset = np.load(self.path_data, allow_pickle = True)
        dataset = dataset.item()

        #Targets of training
    
        targets = list(map(self.oracle2proxy, dataset["energies"]))
        self.targets = np.array(targets)


        #Samples of training
        samples = list(map(self.proxy.base2proxy, dataset["samples"]))
        self.samples = np.array(samples)


    def reshuffle(self):
        self.samples, self.targets = shuffle(
            self.samples, self.targets, random_state=self.seed_data
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


    def get_data_loaders(self):

        if self.shuffle_data:
            self.reshuffle()
        
        train_size = int(0.8 * self.__len__())
        test_size = self.__len__() - train_size

        train_dataset = []
        test_dataset = []

        for i in range(
            test_size, test_size + train_size
        ):  # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
            train_dataset.append(self.__getitem__(i))
        for i in range(test_size):  # test data is drawn from oldest datapoints
            test_dataset.append(self.__getitem__(i))

        tr = data.DataLoader(
            train_dataset,
            batch_size = self.proxy.batch_size,
            shuffle = True,
            num_workers= 0,
            pin_memory= False
        )

        te = data.DataLoader(
            test_dataset,
            batch_size = self.proxy.batch_size,
            shuffle = False,
            num_workers=0,
            pin_memory=False
        )
    
        return tr, te
    


###
'''
MODEL ZOO
'''
###

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
        super(MLP, self).__init__()
        # initialize constants and layers
        self.config = config

        act_func = "gelu"

        # Architecture
        self.input_max_length = self.config.env.max_len
        self.input_classes = self.config.env.dict_size + 1  # +1 for eos
        self.total_fidelities = self.config.env.total_fidelities
        self.dropout = self.config.proxy.training.dropout

        self.layers_before_fid = 2
        self.layers_after_fid = 4
        self.filters_before_fid = 256
        self.filters_after_fid = 256
        self.latents = 200

        self.init_layer_depth = int(
            (self.input_classes) * (self.input_max_length + 1)
        )  # +1 for eos token

        # build input and output layers
        self.initial_layer = nn.Linear(
            int(self.init_layer_depth), self.filters_before_fid
        )  # layer which takes in our sequence in one-hot encoding
        self.initial_activation = Activation(act_func)
        # intermediate before id : later : Linear and activation
        # from intermediate_1 to latent : linear and activation
        self.towards_latent_layer = nn.Linear(self.filters_before_fid, self.latents)
        self.towards_latent_activation = Activation(act_func)
        # from latent to intermediate_2 : no activation ?
        self.fidelity_layer_depth = self.latents + self.total_fidelities
        self.from_latent_layer = nn.Linear(
            self.fidelity_layer_depth, self.filters_after_fid
        )
        self.from_latent_activation = Activation(
            act_func)
        # final
        self.output_layer = nn.Linear(self.filters_after_fid, 1, bias=False)

        # build hidden layers
        self.lin_layers_before_fid = []
        self.activations_before_fid = []
        self.norms_before_fid = []  # pas pour l'instant
        self.dropouts_before_fid = []

        for _ in range(self.layers_before_fid):
            self.lin_layers_before_fid.append(
                nn.Linear(self.filters_before_fid, self.filters_before_fid)
            )
            self.activations_before_fid.append(
                Activation(act_func)
            )
            self.norms_before_fid.append(nn.BatchNorm1d(self.filters_before_fid))
            self.dropouts_before_fid.append(nn.Dropout(p=self.dropout))

        # initialize module lists
        self.lin_layers_before_fid = nn.ModuleList(self.lin_layers_before_fid)
        self.activations_before_fid = nn.ModuleList(self.activations_before_fid)
        self.norms_before_fid = nn.ModuleList(self.norms_before_fid)
        self.dropouts_before_fid = nn.ModuleList(self.dropouts_before_fid)

        self.lin_layers_after_fid = []
        self.activations_after_fid = []
        self.norms_after_fid = []
        self.dropouts_after_fid = []

        for _ in range(self.layers_after_fid):
            self.lin_layers_after_fid.append(
                nn.Linear(self.filters_after_fid, self.filters_after_fid)
            )
            self.activations_after_fid.append(
                Activation(act_func)
            )
            self.norms_after_fid.append(nn.BatchNorm1d(self.filters_after_fid))
            self.dropouts_after_fid.append(nn.Dropout(p=self.dropout))

        # initialize module lists
        self.lin_layers_after_fid = nn.ModuleList(self.lin_layers_after_fid)
        self.activations_after_fid = nn.ModuleList(self.activations_after_fid)
        self.norms_after_fid = nn.ModuleList(self.norms_after_fid)
        self.dropouts_after_fid = nn.ModuleList(self.dropouts_after_fid)

    def forward(self, x):
        x_seq = x[:, : -self.total_fidelities]
        x_fid = x[:, self.init_layer_depth :]  # +1 non convention python

        # BEFORE FID
        x_before_fid = self.initial_activation(self.initial_layer(x_seq))
        
        for i in range(self.layers_before_fid):
            x_before_fid = self.lin_layers_before_fid[i](x_before_fid)
            x_before_fid = self.activations_before_fid[i](x_before_fid)
            x_before_fid = self.dropouts_before_fid[i](x_before_fid)
            x_before_fid = self.norms_before_fid[i](x_before_fid)

        # TO LATENT
        x_latent = self.towards_latent_layer(x_before_fid)
        x_latent = self.towards_latent_activation(x_latent)

        # ADDING FID
        x_after_fid = torch.cat((x_latent, x_fid), dim=1)
        x_after_fid = self.from_latent_layer(x_after_fid)
        x_after_fid = self.from_latent_activation(x_after_fid)

        # AFTERFID
        for i in range(self.layers_after_fid):
            x_after_fid = self.lin_layers_after_fid[i](x_after_fid)
            x_after_fid = self.activations_after_fid[i](x_after_fid)
            x_after_fid = self.dropouts_after_fid[i](x_after_fid)
            x_after_fid = self.norms_after_fid[i](x_after_fid)

        y = self.output_layer(x_after_fid)

        return y
