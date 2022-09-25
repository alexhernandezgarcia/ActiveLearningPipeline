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


'''
ACQUISITION WRAPPER
'''

class AcquisitionFunction:
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
    
    # def get_reward(self, inputs_af_base):
    #     outputs = self.acq.get_reward_batch(inputs_af_base)
    #     return  outputs
    
    

'''
BASE CLASS FOR ACQUISITION
'''
class AcquisitionFunctionBase:
    '''
    Cf Oracle class : generic AF class which calls the right AF sub_class
    '''
    @abstractmethod
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.device = self.config.device
        self.total_fidelities = self.config.env.total_fidelities
        
        #the specific class of the exact AF is instantiated here
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
SUBCLASS SPECIFIC ACQUISITION
'''

class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
    
    def load_best_proxy(self):
        super().load_best_proxy()

    
    def get_reward_batch(self, inputs_af_base): #inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)

        inputs_af = list(map(self.base2af, inputs_af_base))
        inputs = torch.stack(inputs_af).view(len(inputs_af_base), -1)

        self.load_best_proxy()
        self.proxy.model.eval()
        with torch.no_grad():
            outputs = self.proxy.model(inputs)
        return outputs

    def get_logits_fidelity(self, inputs_eos_base):
        inputs_all_fidelities = [(input[0], fid) for input in inputs_eos_base for fid in range(self.total_fidelities)]
        # inputs_af = list(map(self.base2af, inputs_all_fidelities))
        # inputs_af = torch.stack(inputs_af).view(len(inputs_eos_base) * self.total_fidelities, -1)
       
        outputs = self.get_reward_batch(inputs_all_fidelities)

        return outputs.view(len(inputs_eos_base), self.total_fidelities)

    def get_sum_reward_batch(self, inputs_af_base):
        inputs_indices = torch.LongTensor(
            sum([[i] * self.total_fidelities for i in range(len(inputs_af_base))], [])
        ).to(self.device)
        inputs_base_all_fidelities = [(input[0], fid) for input in inputs_af_base for fid in range(self.total_fidelities)]
        
        # inputs_af_all_fidelities = list(map(self.base2af, inputs_base_all_fidelities))
        # inputs_af_all_fidelities = torch.stack(inputs_af_all_fidelities).view(len(inputs_af_base) * self.total_fidelities, -1)
    
        outputs = self.get_reward_batch(inputs_base_all_fidelities).view(len(inputs_base_all_fidelities))
        # print(outputs)
        # print(inputs_indices)
        # print(torch.zeros(
        #     (len(inputs_af_base,))
        # ))
        # return
        sum_rewards = torch.zeros(
            (len(inputs_af_base,))
        ).to(self.device).index_add_(0, inputs_indices, outputs)

        return sum_rewards


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

    def get_reward_batch(self, inputs_af_base): #inputs_af = list of ...

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

        # super().get_reward_batch(inputs_af_base)

        # inputs_af = list(map(self.base2af, inputs_af_base))
        # inputs = torch.stack(inputs_af).view(len(inputs_af_base), -1)

        # self.load_best_proxy()
        # self.proxy.model.eval()
        # with torch.no_grad():
        #     outputs = self.proxy.model(inputs)
        # return outputs

    def get_logits_fidelity(self, inputs_eos_base):
        inputs_all_fidelities = [(input[0], fid) for input in inputs_eos_base for fid in range(self.total_fidelities)]
        # inputs_af = list(map(self.base2af, inputs_all_fidelities))
        # inputs_af = torch.stack(inputs_af).view(len(inputs_eos_base) * self.total_fidelities, -1)
       
        outputs = self.get_reward_batch(inputs_all_fidelities)

        return outputs.view(len(inputs_eos_base), self.total_fidelities)

    def get_sum_reward_batch(self, inputs_af_base):
        inputs_indices = torch.LongTensor(
            sum([[i] * self.total_fidelities for i in range(len(inputs_af_base))], [])
        ).to(self.device)
        inputs_base_all_fidelities = [(input[0], fid) for input in inputs_af_base for fid in range(self.total_fidelities)]
        
        # inputs_af_all_fidelities = list(map(self.base2af, inputs_base_all_fidelities))
        # inputs_af_all_fidelities = torch.stack(inputs_af_all_fidelities).view(len(inputs_af_base) * self.total_fidelities, -1)
    
        outputs = self.get_reward_batch(inputs_base_all_fidelities).view(len(inputs_base_all_fidelities))
        # print(outputs)
        # print(inputs_indices)
        # print(torch.zeros(
        #     (len(inputs_af_base,))
        # ))
        # return
        sum_rewards = torch.zeros(
            (len(inputs_af_base,))
        ).to(self.device).index_add_(0, inputs_indices, outputs)

        return sum_rewards


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
        
        fid_tensor = torch.tensor([fid])
        fid_tensor = F.one_hot(fid_tensor.long(), num_classes = self.total_fidelities)
        fid_tensor = fid_tensor.reshape(1, -1).float()
        
        input_af = torch.cat((input_af, fid_tensor), dim = 1)

        return input_af.to(self.device)[0]

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

        #for each element X, compute several proxy values (with dropout), to deduce the mean and std
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        else:
            raise FileNotFoundError
        original_shape = X.shape
        nb_queries = original_shape[0]

        if X.dim() == 4:
            nb_queries = nb_queries * X.shape[2]
            X = X.unsqueeze(2)
            
        X = X.view(nb_queries, -1)

        self.proxy.model.train(mode = True)
        with torch.no_grad():
            outputs = torch.hstack([self.proxy.model(X) for _ in range(self.nb_samples)]).cpu().detach().numpy()
            mean = np.mean(outputs, axis = 1)
            std = np.std(outputs, axis = 1)
        
        #build the matrix std
        # print("original shape", original_shape)
        #print("mean originai shap", original_shape,  torch.reshape(torch.from_numpy(mean), tuple(original_shape[:-1])))
        # print("std originai shap", torch.reshape(torch.from_numpy(std), tuple(original_shape[:-1])))
        # mean = torch.reshape(torch.from_numpy(mean), tuple(original_shape[:-1]))
        # covar = torch.reshape(torch.from_numpy(std), tuple(original_shape[:-1]))
        #print("covar matrice original shape", np.diag())
        mean = torch.from_numpy(mean).unsqueeze(0)
        covar = torch.from_numpy(np.diag(std))
        
        #return a Gypotorch Multivariate object
        
        # print("posterior created with mean", mean)
        # print("posterior created with variance", covar)

        posterior = MultivariateNormal(mean = mean, covariance_matrix = covar)

     
        posterior.mvn = posterior
        posterior.device = self.config.device
        posterior.dtype = mean.dtype

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
    

    



