class Querier:
    '''
    Samples with the GFlownet latest model and then do post post processing to enhance the candidates and get statistics on them
    '''
    # The basic querier just samples candidate = (seq, fid) with the trained gflownet, no change compared to single_fidelity !
    # But as we'll add duplicates filtering, ... all these enhancements will differ a bit from previous_code with the fidelity (We'll see as we code them)
    def __init__(self, config, gflownet):
        self.config = config
        self.gflownet = gflownet
        self.n_queries = self.config.al.queries_per_it

    
    def build_query(self):
        '''
        calls gflownet.sample() through sampleForQuery and then do post processing on it
        '''
        queries = self.sample4query() #as a list of tuples [(seq, fid)]
        queries = self.enhance_queries(queries)
        queries = self.construct_query(queries)
        return queries
    
    def sample4query(self):
        queries = self.gflownet.sample(self.n_queries)
        return queries
  
    def enhance_queries(self, queries):
        '''
        runs filtering, annealing, ...
        '''
        return queries

    def construct_query(self, queries):
        '''
        Final filtering with fancy tools : clustering ...
        '''
        return queries