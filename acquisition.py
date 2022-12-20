import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler

"""
ACQUISITION WRAPPER
"""


class AcquisitionFunction:
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy.proxy

        self.init_acquisition()

    def init_acquisition(self):
        # so far, only proxy acquisition function has been implemented, only add new acquisition class inheriting from AcquisitionFunctionBase to innovate
        if self.config.acquisition.main.lower() == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "oracle":
            self.acq = AcquisitionFunctionOracle(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "ucb":
            self.acq = AcquisitionFunctionUCB(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "ei":
            self.acq = AcquisitionFunctionEI(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "botorch-ucb":
            self.acq = AcquisitionFunctionBotorchUCB(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "mes":
            self.acq = AcquisitionFunctionMES(self.config, self.proxy)
        else:
            raise NotImplementedError

    def get_reward(self, inputs_af_base):
        outputs = self.acq.get_reward_batch(inputs_af_base)
        return outputs


"""
BASE CLASS FOR ACQUISITION
"""


class AcquisitionFunctionBase:
    """
    Cf Oracle class : generic AF class which calls the right AF sub_class
    """

    @abstractmethod
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.device = self.config.device

        # the specific class of the exact AF is instantiated here

    @abstractmethod
    def load_best_proxy(self):
        """
        In case, loads the latest version of the proxy (no need normally)
        """
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)

        else:
            raise FileNotFoundError

    @abstractmethod
    def get_reward_batch(self, inputs_af_base):
        """
        Args:
            inputs_af_base: list of arrays
        Returns:
            tensor of outputs
        Function:
            calls the get_reward method of the appropriate Acquisition Class (MI, EI, UCB, Proxy, Oracle etc)
        """
        pass


"""
SUBCLASS SPECIFIC ACQUISITION
"""


class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)

    def load_best_proxy(self):
        super().load_best_proxy()

    def get_reward_batch(self, inputs_af_base):  # inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)

        inputs_af = list(map(torch.tensor, inputs_af_base))
        input_afs_lens = list(map(len, inputs_af_base))
        inputs = pad_sequence(inputs_af, batch_first=True, padding_value=0.0)

        self.load_best_proxy()
        self.proxy.model.train()
        with torch.no_grad():
            outputs = self.proxy.model(inputs, input_afs_lens, None)
        return outputs


class AcquisitionFunctionOracle(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)

    def get_reward_batch(self, inputs_af_base):
        super().get_reward_batch(inputs_af_base)
        outputs = self.proxy.get_score(inputs_af_base)
        return torch.FloatTensor(outputs)


class AcquisitionFunctionUCB(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
        self.nb_samples = 20

    def load_best_proxy(self):
        super().load_best_proxy()

    def get_reward_batch(self, inputs_af_base):  # inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)

        inputs_af = list(map(torch.tensor, inputs_af_base))
        input_afs_lens = list(map(len, inputs_af_base))
        inputs = pad_sequence(inputs_af, batch_first=True, padding_value=0.0)

        self.load_best_proxy()
        self.proxy.model.train()
        with torch.no_grad():
            outputs = (
                torch.hstack([self.proxy.model(inputs, input_afs_lens, None) for _ in range(self.nb_samples)])
                .cpu()
                .detach()
                .numpy()
            )

        mean = np.mean(outputs, axis=1)
        std = np.std(outputs, axis=1)
        score = mean + self.config.acquisition.ucb.kappa * std
        score = torch.Tensor(score)
        score = score.unsqueeze(1)
        return score


class AcquisitionFunctionEI(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)

    def load_best_proxy(self):
        super().load_best_proxy()

    def getMinF(self, inputs, input_len, mask):
        # inputs = torch.Tensor(inputs).to(self.config.device)
        outputs = self.proxy.model(inputs, input_len, mask).cpu().detach().numpy()
        self.best_f = np.percentile(outputs, self.config.acquisition.ei.max_percentile)

    def get_reward_batch(self, inputs_af_base):  # inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)
        inputs_af = list(map(torch.tensor, inputs_af_base))
        input_afs_lens = list(map(len, inputs_af_base))
        inputs = pad_sequence(inputs_af, batch_first=True, padding_value=0.0)

        self.getMinF(inputs, input_afs_lens, None)

        self.load_best_proxy()
        self.proxy.model.train()
        with torch.no_grad():
            outputs = (
                torch.hstack([self.proxy.model(inputs, input_afs_lens, None)])
                .cpu()
                .detach()
                .numpy()
            )
        mean = np.mean(outputs, axis=1)
        std = np.std(outputs, axis=1)
        mean, std = torch.from_numpy(mean), torch.from_numpy(std)
        u = (mean - self.best_f) / (std + 1e-4)
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = std * (updf + u * ucdf)
        ei = ei.cpu().detach()
        return ei


class ProxyBotorchMES(Model):
    def __init__(self, config, nn):
        super().__init__()
        self.proxy = nn
        self.config = config
        self._num_outputs = 1
        self.nb_samples = 20

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X, observation_noise, posterior_transform)

        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        else:
            raise FileNotFoundError
        self.proxy.model.train()
        dim = X.ndim

        if dim==4:
            X = X.squeeze(-3).squeeze(-2)
        elif dim == 3:
            X = X.squeeze(-2)

        with torch.no_grad():
             outputs = torch.hstack([self.proxy.model(X, None, None) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1)
        var = torch.var(outputs, axis=1)

        if dim==2:
            # covar = torch.cov(outputs)
            covar = torch.diag(var)
        elif dim==4:
            var = var.unsqueeze(-1).unsqueeze(-1)
            mean = mean.unsqueeze(-1).unsqueeze(-1)
            covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
            covar = covar.unsqueeze(-1)
        elif dim==3:
            var = var.unsqueeze(-1)
            mean = mean.unsqueeze(-1)
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
        
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

class ProxyBotorchUCB(Model):
    def __init__(self, config, nn):
        super().__init__()
        self.proxy = nn
        self.config = config
        self._num_outputs = 1
        self.nb_samples = 20

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X, observation_noise, posterior_transform)

        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        else:
            raise FileNotFoundError
        self.proxy.model.train()
        dim = X.ndim

        # if dim==4:
        #     X = X.squeeze(-3).squeeze(-2)
        if dim == 3:
            X = X.squeeze(-2)

        with torch.no_grad():
             outputs = torch.hstack([self.proxy.model(X, None, None) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1).unsqueeze(-1)
        var = torch.var(outputs, axis=1).unsqueeze(-1)

        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis = 0)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

class AcquisitionFunctionBotorchUCB(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
	
    def load_best_proxy(self):
        super().load_best_proxy()

    def make_botorch_model(self):
        return ProxyBotorchUCB(self.config, self.proxy)

    def get_reward_batch(self, inputs_af_base):
        #create the model
        model = self.make_botorch_model()
        #make candidate set
        #transform inputs_af_base into good format
        inputs_af = list(map(torch.tensor, inputs_af_base))
        inputs_af = pad_sequence(inputs_af, batch_first=True, padding_value=0.0).to('cuda').unsqueeze(-2)
        # inputs_af = list(map(self.base2af, inputs_af_base))
        # inputs_af = torch.stack(inputs_af).view(len(inputs_af_base),1, -1)
        sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)
        UCB = qUpperConfidenceBound(
            model = model, beta = 0.1, sampler = sampler)
        acq_values = UCB(inputs_af)
        print(acq_values)
        return acq_values
        
class AcquisitionFunctionMES(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
	
    def load_best_proxy(self):
        super().load_best_proxy()

    def make_botorch_model(self):
        return ProxyBotorchMES(self.config, self.proxy)

    def make_candidate_set(self):
        #load the npy file with current evaluated candidates
        loading_path = self.config.path.data_oracle
        dataset = np.load(loading_path, allow_pickle = True).item()
        samples = dataset["samples"] 
        #convert into list of af tensors
        candidates = list(map(torch.tensor, samples))
        candidates = pad_sequence(candidates, batch_first=True, padding_value=0.0)
        # candidates = list(map(self.base2af, samples))
        #turn into single tensor
        # candidates = torch.stack(candidates)
        # candidates = candidates.view(len(samples), 1, -1)
        return candidates

    def get_reward_batch(self, inputs_af_base):
        #create the model
        model = self.make_botorch_model()
        #make candidate set
        candidates = self.make_candidate_set().float().to('cuda')
        #transform inputs_af_base into good format
        inputs_af = list(map(torch.tensor, inputs_af_base))
        inputs_af = pad_sequence(inputs_af, batch_first=True, padding_value=0.0).to('cuda').unsqueeze(-2)
        # inputs_af = list(map(self.base2af, inputs_af_base))
        # inputs_af = torch.stack(inputs_af).view(len(inputs_af_base),1, -1)
        MES = qLowerBoundMaxValueEntropy(
            model = model, 
            candidate_set = candidates)
   
        acq_values = MES(inputs_af)
        print(acq_values)
        return acq_values
    
