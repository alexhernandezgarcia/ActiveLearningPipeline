"""
Active learning with GFlowNets
"""

from omegaconf import DictConfig, OmegaConf
from oracle import Oracle
from proxy import Proxy
from acquisition import AcquisitionFunction
from env import Env
from gflownet import GFlowNet
from querier import Querier
from utils.logger import Logger


# TODO : instantiate the config with hydra . Once we have the config object we can pass
# it to other functions
class ActiveLearning:
    """
    AL Global pipeline : so far it is a class (the config is easily passed on to all
    methods), very simple, but it can be a single simple function

    Bao's comments:
    assume we have a config object that we can pass to all the components of the
    pipeline like in the previous code, eg "config_test.yaml"
    setup function that creates the directories to save data, ...'

    """

    def __init__(self, config):
        self.config = config
        self.setup()
        self.logger = Logger(self.config)
        self.oracle = Oracle(self.config, self.logger)
        self.proxy = Proxy(self.config, self.logger)
        self.acq = AcquisitionFunction(self.config, self.proxy)
        self.gfn_agent = GflownetAgent(self.config, self.logger, self.acq)
        # self.env = Env(self.config, self.acq)
        # self.gflownet = GFlowNet(self.config, self.logger, self.env)
        # self.querier = Querier(self.config, self.gflownet)

    def run_pipeline(self):
        # intialise iter so that doesnt lead to error in logging stats of initial dataset
        # self.iter = None
        self.oracle.initialize_dataset()
        for self.iter in range(self.config.al.n_iter):
            self.logger.set_context("iter{}".format(self.iter + 1))
            self.iterate()
        self.logger.finish()

    def iterate(self):
        self.proxy.train()
        queries, energies = self.gfn_agent.run_pipeline()
        self.oracle.update_dataset(queries, energies)

    def setup(self):
        """
        Creates the working directories to store the data.
        """
        return


class GflownetAgent:
    """
    Trains the policy network on the oracle rewards in the absence of the active learning setting
    """

    def __init__(self, config, logger=None, acq=None):
        self.config = config
        self.setup()
        # initalise logger object
        if logger == None:
            self.logger = Logger(self.config)
        # use same logger object as that of active learning
        else:
            self.logger = logger

        self.oracle = Oracle(self.config, self.logger)

        if acq == None:
            self.acq = AcquisitionFunction(self.config, self.oracle)
        else:
            # in case it is an active learning pipeline, you need proxy in acq
            # proxy is only in al pipeline
            # if no al, then acq is oracle
            self.acq = acq

        self.env = Env(self.config, self.acq)
        self.gflownet = GFlowNet(self.config, self.logger, self.env)
        self.querier = Querier(self.config, self.gflownet)

    def run_pipeline(self):
        self.gflownet.train()
        queries = self.querier.build_query()
        energies = self.oracle.score(queries)
        # self.logger.finish()
        return queries, energies

    def setup(self):
        """
        Creates the working directories to store the data.
        """
        return


if __name__ == "__main__":
    # TODO : activelearning pipeline as a simple function, without the class ?
    config_test_name = "./config_test.yaml"
    config = OmegaConf.load(config_test_name)
    if config.al.mode == True:
        al = ActiveLearning(config=config)
        al.run_pipeline()
    else:
        gfn = GflownetAgent(config=config)
        queries, energies = gfn.run_pipeline()
        gfn.logger.finish()
