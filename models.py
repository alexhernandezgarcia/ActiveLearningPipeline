'''Import statements'''
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends

from sklearn.utils import shuffle

import os
import sys
from utils import *

'''
This script contains models for fitting DNA sequence data

> Inputs: list of DNA sequences in letter format
> Outputs: predicted binding scores, prediction uncertainty 

To-do's
==> upgrade to uncertainty-enabled architecture
==> implement ensembling (model of models? need to stop training for each model individually)
==> add noisey augmentation and/or few-shot dimension reduction
==> add positional embedding

Problems
==> we need to think about whether or not to shuffle test set between runs, or indeed what to use in the test set at all - right now we shuffle
'''


class model():
    def __init__(self, params, ensembleIndex):
        self.params = params
        self.ensembleIndex = ensembleIndex
        self.params['history'] = min(20, self.params['max training epochs']) # length of past to check
        self.params['plot training results'] = self.params['plot results'] # plot loss curves
        self.initModel()
        torch.random.manual_seed(params['random seed'])


    def initModel(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        self.model = MLP(self.params)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
        datasetBuilder = buildDataset(self.params)
        self.mean, self.std = datasetBuilder.getStandardization()


    def save(self, best):
        if best == 0:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex)+'_final')
        elif best == 1:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex))


    def load(self,ensembleIndex):
        '''
        Check if a checkpoint exists for this model - if so, load it
        :return:
        '''
        dirName = getModelName(ensembleIndex)
        if os.path.exists('ckpts/' + dirName):  # reload model
            checkpoint = torch.load('ckpts/' + dirName)

            if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(checkpoint['model_state_dict']):
                    checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #prev_epoch = checkpoint['epoch']

            if self.params['GPU'] == 1:
                model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            self.model.eval()
            #print('Reloaded model: ', dirName)
        else:
            pass
            #print('New model: ', dirName)


    def converge(self):
        '''
        train model until test loss converges
        :return:
        '''
        [self.err_tr_hist, self.err_te_hist] = [[], []] # initialize error records

        tr, te, self.datasetSize = getDataloaders(self.params)

        #print(f"Dataset size is: {bcolors.OKCYAN}%d{bcolors.ENDC}" %self.datasetSize)

        self.converged = 0 # convergence flag
        self.epochs = 0


        while (self.converged != 1):
            if self.epochs > 0: #  this allows us to keep the previous model if it is better than any produced on this run
                model.train_net(self, tr)
            else:
                self.err_tr_hist.append(0)

            model.test(self, te) # baseline from any prior training

            if self.err_te_hist[-1] == np.min(self.err_te_hist): # if this is the best test loss we've seen
                self.save(best=1)

            # after training at least 10 epochs, check convergence
            if self.epochs >= self.params['history']:
                self.checkConvergence()

            self.epochs += 1
            #self.save(self, best=0) # save ongoing checkpoint

            #sys.stdout.flush()
            #sys.stdout.write('\repoch={}; train loss={:.5f}; test loss={:.5f};\r'.format(self.epochs, self.err_tr_hist[-1], self.err_te_hist[-1]))

            #print('epoch={}; train loss={:.5f}; test loss={:.5f};'.format(self.epochs, self.err_tr_hist[-1], self.err_te_hist[-1]))

        if self.params['plot training results'] == 1:
            self.plotResults()


    def plotResults(self):
        '''
        plot train and test loss as function of epochs, and combine subplots of each pipeline iteration
        :return:
        '''
        columns = min(5,self.params['pipeline iterations'])

        plt.figure(1)
        rows = max([1,(self.params['pipeline iterations'] // 5)])
        plt.subplot(rows, columns, self.params['iteration'])
        plt.plot(self.err_tr_hist,'o-', label='Train Loss')
        plt.plot(self.err_te_hist,'d-', label='Test Loss')
        plt.title('Iteration #%d' % self.params['iteration'])
        plt.xlabel('Epochs')
        if self.params['iteration'] == 1:
            plt.legend()


    def train_net(self, tr):
        '''
        perform one epoch of training
        :param tr: training set dataloader
        :return: n/a
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.getLoss(trainData)
            err_tr.append(loss.data)  # record the loss

            self.optimizer.zero_grad()  # run the optimizer
            loss.backward()
            self.optimizer.step()

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())


    def test(self, te):
        '''
        get the loss over the test dataset
        :param te: test set dataloader
        :return: n/a
        '''
        err_te = []
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            for i, testData in enumerate(te):
                loss = self.getLoss(testData)
                err_te.append(loss.data)  # record the loss

        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())


    def getLoss(self, train_data):
        """
        get the regression loss on a batch of datapoints
        :param train_data: sequences and scores
        :return: model loss over the batch
        """
        inputs = train_data[0]
        targets = train_data[1]
        if self.params['GPU'] == 1:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # convert our inputs to a one-hot encoding
        #one_hot_inputs = F.one_hot(torch.Tensor(letters2numbers(inputs)).long(), 4)
        # flatten inputs to a 1D vector
        #one_hot_inputs = torch.reshape(one_hot_inputs, (one_hot_inputs.shape[0], self.params['input length']))
        # evaluate the model
        output = self.model(inputs.float())
        # loss function - some room to choose here!
        targets = (targets - self.mean)/self.std # standardize the targets, but only during training
        return F.smooth_l1_loss(output[:,0], targets.float())


    def checkConvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        if all(np.asarray(self.err_te_hist[-self.params['history']+1:])  > self.err_te_hist[-self.params['history']]):
            self.converged = 1

        # check if test loss is unchanging
        if abs(self.err_te_hist[-self.params['history']] - np.average(self.err_te_hist[-self.params['history']:]))/self.err_te_hist[-self.params['history']] < eps:
            self.converged = 1

        if self.epochs >= self.params['max training epochs']:
            self.converged = 1


        if self.converged == 1:
            print(f'{bcolors.OKCYAN}Model training converged{bcolors.ENDC} after {bcolors.OKBLUE}%d{bcolors.ENDC}' %self.epochs + f" epochs and with a final test loss of {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % np.amin(np.asarray(self.err_te_hist)))


    def evaluate(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(torch.Tensor(Data).float())
            if output == 'Average':
                return np.average(out,axis=1) * self.std + self.mean
            elif output == 'Variance':
                return np.var(out.detach().numpy(),axis=1)

    def loadEnsemble(self,models):
        '''
        load up a model ensemble
        :return:
        '''
        self.model = modelEnsemble(models)


class modelEnsemble(nn.Module): # just for evaluation of a pre-trained ensemble
    def __init__(self,models):
        super(modelEnsemble, self).__init__()
        self.models = models
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        output = []
        for i in range(len(self.models)): # get the prediction from each model
            output.append(self.models[i](x.clone()))

        output = torch.cat(output,dim=1) #
        return output # return mean and variance of the ensemble predictions


class buildDataset():
    '''
    build dataset object
    '''
    def __init__(self, params):
        dataset = np.load('datasets/' + params['dataset']+'.npy', allow_pickle=True)
        dataset = dataset.item()
        self.samples = dataset['sequences']
        self.targets = dataset['scores']

        self.samples, self.targets = shuffle(self.samples, self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def getFullDataset(self):
        return self.samples, self.targets

    def getStandardization(self):
        return np.mean(self.targets), np.var(self.targets)


def getDataloaders(params): # get the dataloaders, to load the dataset in batches
    '''
    creat dataloader objects from the dataset
    :param params:
    :return:
    '''
    training_batch = params['batch_size']
    dataset = buildDataset(params)  # get data
    train_size = int(0.8 * len(dataset))  # split data into training and test sets

    test_size = len(dataset) - train_size

    # construct dataloaders for inputs and targets
    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size): # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
        train_dataset.append(dataset[i])
    for i in range(test_size):
        test_dataset.append(dataset[i])

    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 2, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 2, pin_memory=True)

    return tr, te, dataset.__len__()


def getDataSize(params):
    dataset = np.load('datasets/' + params['dataset'] + '.npy', allow_pickle=True)
    dataset = dataset.item()
    samples = dataset['sequences']

    return len(samples[0])


class MLP(nn.Module):
    def __init__(self,params):
        super(MLP,self).__init__()
        # initialize constants and layers

        if True:
            act_func = 'relu'
        #elif params['activation']==2:
        #    act_func = 'kernel'

        params['input length'] = int(getDataSize(params))

        self.layers = params['model layers']
        self.filters = params['model filters']

        # build input and output layers
        self.initial_layer = nn.Linear(params['input length'], self.filters) # layer which takes in our sequence
        self.activation1 = Activation(act_func,self.filters,params)
        self.output_layer = nn.Linear(self.filters, 1)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.norms = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func, self.filters))
            #self.norms.append(nn.BatchNorm1d(self.filters))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        #self.norms = nn.ModuleList(self.norms)


    def forward(self, x):
        x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            #x = self.norms[i](x)

        x = self.output_layer(x) # linear transformation to output
        return x


class kernelActivation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        gamma = 1/(6*(self.dict[-1]-self.dict[-2])**2) # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma',torch.ones(1) * gamma) #

        #self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)


    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        return torch.exp(-self.gamma*(x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


