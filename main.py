'''import statements'''
import activeLearner
from utils import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # annoying numpy error


'''
This code implements an active learning protocol for global minimization of some function

To-Do
==> incorporate gFlowNet
==> incorporate RL

low priority /long term
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs
==> think carefully about how we split test and train datasets
==> augmentation and other regularization
==> maybe print outputs at the end of each iteration as a lovely table
==> add a debug mode for saving training results
==> characterize and track specific local minima, including suboptimal ones, over iterations
==> optimize transformer initialization

known issues
==> training parallelism hangs on iteration #2 on linux

'''

# get command line input
parser = argparse.ArgumentParser()
# high level
parser.add_argument('--run_num', type=int, default=0)
parser.add_argument('--sampler_seed', type=int, default=0)
parser.add_argument('--model_seed', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=0)
parser.add_argument('--device', type = str, default = 'local')
parser.add_argument('--GPU', type = bool, default = True)
parser.add_argument('--explicit_run_enumeration', type = bool, default = False)
# dataset settings
parser.add_argument('--dataset_type', type = str, default = 'toy')
parser.add_argument('--dataset', type=str, default='potts')
parser.add_argument('--init_dataset_length', type = int, default = int(1e2))
parser.add_argument('--dict_size', type = int, default = 4)
parser.add_argument('--variable_sample_length', type = bool, default = True)
parser.add_argument('--min_sample_length', type = int, default = 10)
parser.add_argument('--max_sample_length', type = int, default = 40)
# AL settings
parser.add_argument('--query_mode', type=str, default='uncertainty')
parser.add_argument('--test_mode', type = bool, default = False)
parser.add_argument('--pipeline_iterations', type = int, default = 10)
parser.add_argument('--minima_dist_cutoff', type = float, default = 0.25)
parser.add_argument('--queries_per_iter', type = int, default = 100)
parser.add_argument('--mode', type = str, default = 'training')
parser.add_argument('--debug', type = bool, default = True)
# querier settings
parser.add_argument('--model_state_size', type = int, default = 30)
# model settings
parser.add_argument('--model_type', type = str, default = 'mlp')
parser.add_argument('--training_parallelism', type = bool, default = False)
parser.add_argument('--model_ensemble_size', type = int, default = 10)
parser.add_argument('--model_filters', type = int, default = 256)
parser.add_argument('--embedding_dim', type = int, default = 256)
parser.add_argument('--model_layers', type = int, default = 2)
parser.add_argument('--training_batch_size', type = int, default = 10)
parser.add_argument('--max_epochs', type = int, default = 200)
#sampler settings
parser.add_argument('--sampling_time', type = int, default = int(1e4))
parser.add_argument('--num_samplers', type = int, default = 20)
parser.add_argument('--min_gamma', type = float, default = -3)
parser.add_argument('--max_gamma', type = float, default = 1)

args = parser.parse_args()
params = getParamsDict(args)

#====================================
if params['mode'] == 'evaluation':
    params['pipeline iterations'] = 1

if params['test mode']:
    params['pipeline iterations'] = 2
    params['init dataset length'] = 100
    params['queries per iter'] = 100
    params['sampling time'] = int(1e3)
    params['num samplers'] = 2
    params['model ensemble size'] = 2
    params['max training epochs'] = 5
    params['model filters'] = 12
    params['model layers'] = 1  # for cluster batching
    params['embed dim'] = 12  # embedding dimension
    params['batch size'] = 10  # model training batch size
    params['min sample length'], params['max sample length'] = [10, 20]  # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false
    params['dict size'] = 4  # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4

# paths
if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = 'C:/Users\mikem\Desktop/activeLearningRuns'#'/home/mkilgour/learnerruns'#

#=====================================
if __name__ == '__main__':
    al = activeLearner.activeLearning(params)
    if params['mode'] == 'initalize':
        printRecord("Initialized!")
    elif params['mode'] == 'training':
        al.runPipeline()
    elif params['mode'] == 'evaluation':
        ValueError("No function for this! Write a function to load torch models and evaluate inputs.")


