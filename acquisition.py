import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
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
    


