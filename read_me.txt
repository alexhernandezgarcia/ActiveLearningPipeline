'''
activelearning.py 
'''
- loads the config file
- loads the components of the active learning pipeline : 

-> The oracle is independant
-> The proxy is independant, just uses the transition function oracle2proxy to modify the target_values of an oracle if needed
-> The acquisition function uses the proxy
-> The environment uses the acquisition function (to compute the rewards)
-> The gflownet uses the environment (for the transitions, etc...)
-> The querier uses the gflownet (to sample before enhancing the raw results) 


- runs the pipeline : 
1. Initializing the first known dataset (random for now, but domain knowledge in the future)
2. Each iteration of the pipeline :
	a. Training the proxy 
	b. Training the GFlownet with the new reward function (generating off-policy and on-policy data and training on it)
	c. Building the query sampling the GFlownet
	d. Scoring our queries with the oracles 
	e. Updating the data to train the proxy

TO DO :  
- handle the config with hydra
- create directories for data management (setup)
- initialize a logger with methods to store the statistics during the whole pipeline
- no need for a class for ActiveLearning ? just a class for Logger / a setup function and then we just run the pipeline with the simple bricks.

---------------------------------------------------------------------------------------------------
Note on the formats : 
ORACLE
- the data [(dna sequence, fidelity), oracle-based energy] is stored in a "base_format" : for now ([0, 1, 2, 3], integer for fidelity)
PROXY
- "base2proxy" is specific to each proxy and transforms the stored inputs to the input of the proxy architecture.
- the values from the oracle can be converted for training the proxy with the method "oracle2proxy", specific to each oracle. It is lambda x:-x in the case of negative energies to minimize
ENV
- to manipulate the state transitions (seq, fid) --> (next_seq, next_fid), we define a format called "manip_format". Generally it is [0, 1, 2, 3, 4] when we add the eos action 4 for convenience in the representation. The method "base2manip" is used to convert the offline data stored in the base_format to our "manip" format handled by the methods of the environment.
- the reciprocal method "manip2base" is used to send final states in the "base_format" for evaluation by the acquisition function.
ACQUISITION
- "base2af" transforms sampled candidates from the GFlowNet (online or offline policy) in the correct input format for the acquisition function. It is specific to every acquisition, and is ususally the same as "base2proxy" since the acquisition is based on the proxy. But not necessarily in general.
GFLOWNET
- the training data is generated in the "manip_format". But at each step, we have to send our state to the policy network to have the next action. The method "manip2policy" is specific to each policy network and creates the inputs for the policy network. From the output of the network, we sample an integer that identifies an action and serves to directly update the "manip_format".

-----------------------------------------------------------------------------------------------------------
Organization of the files of each component of the pipeline :
I used the same organization for all the components of the pipeline that can be changed for testing : the proxy / the acquisition function / the oracle. For example, for the oracle :
- A Wrapper Class ("Oracle") : calls the specific oracle we want (Oracle.oracle) and uses the relevant method from the oracle to perform the tasks for our pipeline.
- Each specific oracle is a class that derives from the mother class : eg OracleNupack is a child class of OracleBase. Abstract methods are defined in the Base Class if they are common to all child instances.

----------------------------------------------------------------------------------------------------------
'''
oracle.py
'''
WRAPPER : Oracle
- "initialize_dataset" : initializes the first dataset that is sent for initial scoring (data to train the proxy for the first time). Calls the method "initialize_samples_base" that creates the first candidate dataset : so far it is random, and we can set it specific for each Oracle child class. Then it calls the "score" method, different for each oracle.
- "score" : calls the "score" method of the current oracle to score the candidates later in the pipeline.
- "update_dataset" : given new (queries, energies), we update our database of knowledge to train a more precise proxy.

Oracle classes deriving from OracleBase:
- "initialize_samples_base" : the way we create the initial dataset is different for each oracle
- "get_score" : scoring the queries in the "oracle_format" with the procedure specific to each oracle
- "base2oracle" : see the notes on the formats above
- "oracle2proxy" : cf notes on format

So far, I have implemented : 
- OracleMLP, stored on the git, and trained on data from nupack (when nupack wasn't working for me). Those models are wrapped in the "Model Zoo" at the bottom of the file
- OracleToy : counting the numbers of 0 ie A in the sequences 
- OracleNupack : computing the energy of the sequence with nupack
See the code to see how I tinkered to simulate oracles with different fidelities for each type of oracle.
-----------------------------------------------------------------
'''
proxy.py
'''
WRAPPER : 
- calls the proxy we use : Proxy.proxy
- trains this proxy : 
--> first building the training dataset with the object BuildDataset (using oracle2proxy method from the oracle class to convert the oracle values to actual targets for training the proxy).
--> feeds this data_handler obtained in the method "converge" = specific optimization method for each proxy architecture

Proxy instances inheriting from ProxyBase:
- See the code. The current methods are highly inspired from the previous pipeline. Maybe they should change to be more modular. 

So far, just a MLP proxy is implemented.

TO DO : 
- work with Nikita to reshape the ProxyBase class and its abstract methods so that it is more modular. Implement more proxies.
- Integrate more training parameters for each proxy in the config file, to launch many experiments.

------------------------------------------------------------------
'''
acquisition.py
'''
WRAPPER :
- Just loads the wanted acquisition function. "AcquisitionFunction.acq" will be passed to the Environment to directly use the methods of the AcquisitionFunctionBase children.

Acquisition Functions, children of AcquisitionFunctionBase
- method "load_best_proxy" : loads the best proxy to compute the AF (normally no need to because we update our proxy continously and store it in the variable self.proxy in the pipeline)
- method "get_reward_batch" : takes as input a list of candidates on which we have to compute the reward, converts the format, and processes the acquisition function computationsn.
- "base2af" : for the mentioned format conversion : specific to each AF

So far, I implemented the stupid AcquisitionFunctionProxy, just giving the proxy value as a reward, and the AcquisitionFunctionMES, with Botorch.
For MES, there are many annex methods because we need extra_information to call the built-in MES. It requires a BotorchModel, that I customed in the Model Zoo at the bottom of the file.

TO DO : 
- The Wrapper is very simple here : any better and more elegant way to put it ? 
- Let's debug MES !!!
------------------------------------------------------------------
'''
env.py
'''
WRAPPER, just calling the relevant environment.
So far, just the EnvAptamers has been implemented.

- The methods of EnvBase are quite classic, and it was the interesting work to decline them for the choice of the fidelity at the midle. I will detail more the modelling later.
-----------------------------------------------------------------



------------------------------------------------------------------
TO DO : 
- implement a gflownet folder/module? 