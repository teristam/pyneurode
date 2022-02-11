import numpy as np 
from sklearn import model_selection
import matplotlib.pylab as plt 
from palettable.colorbrewer.qualitative import Dark2_4 as p4color
from sklearn.model_selection import StratifiedKFold
from types import SimpleNamespace
from multiprocessing import Manager
from sklearn.utils import safe_mask


def pos_bin_data(pos,bins=50):
    # bin pos 
    bins = np.linspace(pos.min(),pos.max(),bins)
    binned_idx = np.digitize(pos,bins)-1
    return binned_idx, bins

def get_binned_mean(binned_idx, bins, x):
    # the average value of x in each bins
    x_mean = np.zeros((len(bins),))
    for b in range(len(bins)):
        x_mean[b] = np.nanmean(x[binned_idx==b])

    return x_mean

def crossValidate(model,norm_spiketrain, position, cv=10):
    model = model.fit(norm_spiketrain,position)
    score = model_selection.cross_validate(model,norm_spiketrain,position,cv=cv, 
        scoring=['explained_variance','r2'])
    return score

def stripplot(x,x_predicted=None,lines=5,figsize=(15,5),title=None,first_label='ground_truth', second_label='predicted'):
    # plot x in multiple lines of subplot
    fig = plt.figure(figsize=figsize)
    segment = len(x)//lines
    for i in range(lines):
        ax = fig.add_subplot(lines,1,i+1)
        ax.plot(x[i*segment:(1+i)*segment-1],color=p4color.mpl_colors[0],label=first_label)
        if x_predicted is not None:
            ax.plot(x_predicted[i*segment:(1+i)*segment-1],color=p4color.mpl_colors[1], label=second_label)

    if title:
        fig.suptitle(title)
    ax.legend(loc="lower center",bbox_to_anchor=(0.5, -1),ncol=2)
    # fig.tight_layout()

    fig.subplots_adjust(hspace=0.3)

    return fig


def initStateNS(managerNS):
    managerNS.start_timestamp = None
    managerNS.regressor = None
    managerNS.pos_scaler = None
    managerNS.spk_scaler = None
    managerNS.buffer_is_valid = False
    managerNS.decodeResultIsReady = False
    managerNS.decoderReady = False  
    managerNS.neuron_num = None
    managerNS.kill = False
    managerNS.pca_transformer = None
    managerNS.cluster_id = None

    managerNS.sort_thread_delay = -1
    managerNS.sync_thread_delay = -1
    managerNS.train_delay = -1
    managerNS.decode_delay = -1
    managerNS.zmq_delay = -1


    return managerNS

class State:
    def __init__(self):
        self.buffer_is_valid = False
        self.decodeResultIsReady = False
        self.decoderReady = False
        self.regressor = None
        self.spk_scaler = None
        self.pos_scaler = None
        self.neuron_num = None

class SharedState(State):
    #A state backed by shared memory
    # The state class cannot be added to managed namespace directly. Because any change to the state object can be overwritten by other
    # processes
    def __init__(self, manager:Manager):
        self._buffer_is_valid = manager.Value('i',False)
        self._decodeResultIsReady = manager.Value('i',False)
        self._decoderReady = manager.Value('i',False)
        self._neuron_num = manager.Value('i',-1)

    
    @property
    def buffer_is_valid(self):
        return self._buffer_is_valid.value

    @buffer_is_valid.setter
    def buffer_is_valid(self,value):
        self._buffer_is_valid.value = value
    
    @property
    def decodeResultIsReady(self):
        return self._decodeResultIsReady.value

    @decodeResultIsReady.setter
    def decodeResultIsReady(self,value):
        self._decodeResultIsReady.value = value

    @property
    def decoderReady(self):
        return self._decoderReady.value

    @decoderReady.setter
    def decoderReady(self, value):
        self._decoderReady.value = value

    @property
    def neuron_num(self):
        return self._neuron_num.value

    @neuron_num.setter
    def neuron_num(self, value):
        self._neuron_num.value = value
        





def balanced_cross_val_predict(classifier,resampler, X,y,cv=5,scorer=None,return_prob=False):
    # First balance the dataset in training and then do prediction

    predicted = np.zeros_like(y)
    predicted_prob = np.zeros((len(y),len(np.unique(y))))
    skf = StratifiedKFold(n_splits=cv)
    
    for train_index, test_index in skf.split(X,y):
        # Only do re-balancing in the training set
        X_rs, y_rs = resampler.fit_resample(X[train_index,:],y[train_index])
        
        # train and predict
        classifier.fit(X_rs,y_rs)
        predicted[test_index] = classifier.predict(X[test_index,:])

        if return_prob:
            predicted_prob[test_index,:] = classifier.predict_proba(X[test_index,:])

    results = SimpleNamespace()

    results.predicted = predicted
    
    if scorer is not None:
        score = scorer(y,predicted)
        results.score = score
    
    
    if return_prob:
        results.predict_prob = predicted_prob
    
    return results


def makeSlidingWinFeature(x,win):
    # vector sliding window into feature
    feat = np.zeros((x.shape[0]-win+1,x.shape[1]*win))
    
    for i in range(x.shape[0]-win+1):
        feat[i,:] = x[i:i+win,:].ravel()
    return feat