import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np
#for MES
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.acquisition.cost_aware import CostAwareUtility 
from botorch.models.cost import AffineFidelityCostModel
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior


class AcquisitionFunction:
    '''
    ACQUISITION WRAPPER
    '''
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy.proxy

        self.init_acquisition()
    
    def init_acquisition(self):
        if self.config.acquisition.main == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        elif self.config.acquisition.main == "MES":
            self.acq = AcquisitionFunctionMES(self.config, self.proxy)
        else:
            raise NotImplementedError

    
class AcquisitionFunctionBase:
    '''
    BASE CLASS FOR ACQUISITION
    '''
    @abstractmethod
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.device = self.config.device
        self.total_fidelities = self.config.env.total_fidelities


    @abstractmethod
    def load_best_proxy(self):
        '''
        In case, loads the latest version of the proxy (no need normally)
        '''
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        
        else:
            raise FileNotFoundError

    @abstractmethod
    def get_reward_batch(self, inputs_af_base):
        '''
        calls the get_reward method of the appropriate Acquisition Class (MUtual Information, Expected Improvement, ...)
        '''
        pass


'''
ACQUISITION FUNCTION ZOO
'''

class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
    
    def load_best_proxy(self):
        super().load_best_proxy()

    def get_reward_batch(self, inputs_af_base): 
        '''
        So far, we expect inputs_af_base of shape nb_inputs x 1
        '''
        inputs_af = list(map(self.base2af, inputs_af_base))
        inputs = torch.stack(inputs_af).view(len(inputs_af_base), -1)

        self.load_best_proxy()
        self.proxy.model.eval()

        with torch.no_grad():
            outputs = self.proxy.model(inputs)

        return outputs #no need to .view, it is already the good shape

    def base2af(self, state):
        #useful format
        self.dict_size = self.config.env.dict_size
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len

        seq = state[0]
        initial_len = len(seq)
        fid = state[1]

        #into a tensor and then ohe
        seq_tensor = torch.from_numpy(seq)
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size +1)
        seq_ohe = seq_ohe.reshape(1, -1).float()

        #addind eos token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()
        input_af = torch.cat((seq_ohe, eos_ohe), dim = 1)

        #adding 0-padding
        number_pads = self.max_len - initial_len
        if number_pads:
            padding = torch.cat(
                [torch.tensor([0] * (self.dict_size +1))] * number_pads
            ).view(1, -1)
            input_af = torch.cat((input_af, padding), dim = 1)
        
        #Adding the fidelity
        fid_tensor = torch.tensor([fid])
        fid_tensor = F.one_hot(fid_tensor.long(), num_classes = self.total_fidelities)
        fid_tensor = fid_tensor.reshape(1, -1).float()
        
        input_af = torch.cat((input_af, fid_tensor), dim = 1)

        return input_af.to(self.device)[0]
    


class AcquisitionFunctionMES(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
    
    def load_best_proxy(self):
        super().load_best_proxy()

    def make_botorch_model(self):
        return ProxyBotorch(self.config, self.proxy)
    
    def make_candidate_set(self):
        #load the npy file with current evaluated candidates
        loading_path = self.config.path.data_oracle
        dataset = np.load(loading_path, allow_pickle = True).item()
        samples = dataset["samples"] 
        #convert into list of af tensors
        candidates = list(map(self.base2af, samples))
        #turn into single tensor
        candidates = torch.stack(candidates)
        candidates = candidates.view(len(samples), 1, -1)

        return candidates
    
    def make_cost_utility(self):
        #loads the weights from oracle -> proxy -> acq
        #object with customed weights of model
        return None


    def project_max_fidelity(self, tensor_af):
        #tensor in af format and then put them to max fidelity for the proxy
        #useful format
        
        nb_inputs = len(tensor_af)
        #isolate the non_fidelity part
        tensor_af = tensor_af.view(nb_inputs, -1)
        inputs_without_fid = tensor_af[:, :-self.total_fidelities]
       
    
        #get ohe representations of the highest fidelity
        max_fid = torch.tensor([self.total_fidelities - 1]) #convention : 0 to total_fidelities - 1
        max_fid = F.one_hot(max_fid.long(), num_classes = self.total_fidelities)
        max_fid = max_fid.reshape(1, -1).float()

        max_fids = max_fid.repeat(nb_inputs, 1)
        input_max_fid = torch.cat((inputs_without_fid, max_fids), dim = 1) 
        return input_max_fid.to(self.device).view(nb_inputs, 1, -1)#[0]

    def get_reward_batch(self, inputs_af_base):
        #create the model
        model = self.make_botorch_model()
        #make candidate set
        candidates = self.make_candidate_set()
        #we have the project method
        projection = self.project_max_fidelity
        #we load the weights : right now discrete fidelities 
        cost_utility = self.make_cost_utility()
        #transform inputs_af_base into good format
        inputs_af = list(map(self.base2af, inputs_af_base))
        inputs_af = torch.stack(inputs_af).view(len(inputs_af_base),1, -1)

        MES = qMultiFidelityMaxValueEntropy(
            model = model, 
            candidate_set = candidates, 
            cost_aware_utility = cost_utility,
            project = projection
            )
   
        acq_values = MES(inputs_af)
        
        return acq_values


    def base2af(self, state):
        #useful format
        self.dict_size = self.config.env.dict_size
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len

        seq = state[0]
        initial_len = len(seq)
        fid = state[1]

        #into a tensor and then ohe
        seq_tensor = torch.from_numpy(seq)
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size +1)
        seq_ohe = seq_ohe.reshape(1, -1).float()

        #addind eos token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()

        input_af = torch.cat((seq_ohe, eos_ohe), dim = 1)

        #adding 0-padding
        number_pads = self.max_len - initial_len
        if number_pads:
            padding = torch.cat(
                [torch.tensor([0] * (self.dict_size +1))] * number_pads
            ).view(1, -1)
            input_af = torch.cat((input_af, padding), dim = 1)

        #adding the fidelity 
        fid_tensor = torch.tensor([fid])
        fid_tensor = F.one_hot(fid_tensor.long(), num_classes = self.total_fidelities)
        fid_tensor = fid_tensor.reshape(1, -1).float()
        
        input_af = torch.cat((input_af, fid_tensor), dim = 1)

        return input_af.to(self.device)[0]



'''
ZOO
'''

class ProxyBotorch(Model):
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.nb_samples = 20
    
    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X)
        #loading the best proxy
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        else:
            raise FileNotFoundError
        #depending on the dimension the input, we wil have different formats
        dim_input = X.dim()    
        #for each element X, compute several proxy values (with dropout), to deduce the mean and std   
        self.proxy.model.train(mode = True)
        with torch.no_grad():
            outputs = torch.hstack([self.proxy.model(X) for _ in range(self.nb_samples)]).cpu().detach().numpy()
            mean_1 = np.mean(outputs, axis = 1) 
            std_1 = np.std(outputs, axis = 1) 
        #For the mean
        mean = torch.from_numpy(mean_1)
        #For the variance
        list_std = torch.from_numpy(std_1)
        nb_inputs = list_std.shape[0]
        nb_cofid = list_std.shape[1]
        list_covar = [torch.diag(list_std[i, ...].view(nb_cofid)) for i in range(nb_inputs)]
        covar= torch.stack(list_covar, 0)
        #tinkering to fit the correct input format of MultivariateNormal
        if dim_input == 3:
            mean = mean.unsqueeze(1)
            covar = covar.unsqueeze(1)   
        if dim_input == 4:
            mean = mean.view(mean.shape[0], mean.shape[2], mean.shape[1])
            covar = covar.unsqueeze(1)    
        #creating the mvn    
        mvn = MultivariateNormal(mean = mean, covariance_matrix = covar)
        #deducing the posterior
        posterior = GPyTorchPosterior(mvn)

        return posterior
    
    @property
    def batch_shape(self):
        #not sure about this, read the docs x Moksh
        #self.batch_shape = torch.Size()
        #cls_name = self.__class__.__name__
        return torch.Size([])
    
    @property
    def num_outputs(self):
        #self.num_outputs = 1
        #cls_name = self.__class__.__name__
        return 1
    



