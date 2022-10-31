import pickle
from random import sample

# import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas.core.computation import align
# import seaborn as sns

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import time
from isosplit5 import isosplit5
from scipy import spatial
import warnings
from .spike_sorter_cy import align_spike_cy
from scipy import stats

def sort_spikes(spike_waveforms, eps=1,pca_component=6, min_samples=5,clusterMethod=None):
    start = time.time()
    # Sort spikes

    # pca
    if spike_waveforms.shape[0]>pca_component:
        pca_transformer = PCA(n_components=pca_component).fit(spike_waveforms)
        pc = pca_transformer.transform(spike_waveforms)
        standard_scaler = StandardScaler().fit(pc)
        pc_norm = standard_scaler.transform(pc) #normalize the pca data
        labels = isosplit5(pc_norm.T)

        print('use normalized pc')
    else:
        print('Not enough spikse to sort waveform')
        pca_transformer = None
        pc = None
        standard_scaler = None
        pc_norm = None
        labels = None

    #clustering
    # db = DBSCAN(eps=eps,min_samples=min_samples).fit(pc_norm)
    # db = cluster.OPTICS(min_samples=min_samples,max_eps=eps).fit(pc_norm)
    # db = cluster.MeanShift(bin_seeding=True).fit(pc_norm)
    # labels = db.labels_
    # print(f'Clustering took {time.time()-start:.3f} s')

    return labels, pc_norm, pca_transformer, standard_scaler


# def plot_pc_pair(pc_norm, labels):
#     df_pc = pd.DataFrame(pc_norm)
#     df_pc['label'] = labels
#     sns.pairplot(df_pc,hue='label')


def generate_spike_templates(df):
    #Calculate template
    df_cluster = df.groupby('cluster_id').mean().reset_index()
    cluster_electrode_ids = dict(zip(df_cluster.cluster_id.values, df_cluster.electrode_ids.values))

    spike_waveforms = np.stack(df.spike_waveform_aligned.values)
    cluster_id = df.cluster_id.values

    # Calculate spike templates
    # template_cluster_id contains the cluster_id for each template
    templates,template_cluster_id = calculate_spike_template(spike_waveforms,cluster_id)
    template_electrode_id = np.array([int(cluster_electrode_ids[i]) for i in template_cluster_id]) #which electrode the template are from
    
    # build model for distance
    models = estimate_distance_model(templates, template_cluster_id, spike_waveforms, cluster_id)
    
    return (templates, template_cluster_id, template_electrode_id, models)

def estimate_distance_model(templates, template_cluster_id, spike_waveform_aligned, cluster_ids, metric='mse'):
    '''
    Build a normal distribution for each cluster
    '''
    
    
    cluster_dist = []
    for i,tid in enumerate(template_cluster_id):
        
        spike_waveform = spike_waveform_aligned[cluster_ids==template_cluster_id[i]]
        
        assert spike_waveform.shape[0] > 0
                
        if metric=='mse':
            d = np.mean((spike_waveform - templates[i,:])**2,axis=1) # MSE is equivalent to Euclidean distance
        else:     
            d = spatial.distance.cdist(spike_waveform, templates[[i],:], metric='cosine') #to keep 2D dimension
                

        # Fit normal distribution
        mean = np.mean(d)
        std = np.std(d)
        cluster_dist.append({'mean':mean, 'std':std})
        
    return cluster_dist
        

def calculate_spike_template(spike_waveforms, cluster_ids):
    # Calculate the templates for neurons
    # spike_waveforms should be [n x time]
    # Will remove outliner cluster (-1) automatically
    spike_waveforms = spike_waveforms[cluster_ids!=-1] 
    cluster_ids = cluster_ids[cluster_ids!=-1]
    label_unique = np.unique(cluster_ids)
    template = np.zeros((len(label_unique), spike_waveforms.shape[1]),dtype=np.float32)
    for i,lbls in enumerate(label_unique):
        template[i,:] =spike_waveforms[cluster_ids==lbls,:].mean(axis=0)

    return template, label_unique


def guassmf(x, mean, sigma):
    # Guassian membership function to estimate the probability
    
    return np.exp(-((x-mean)**2) / sigma**2)

def template_matching(templates, spike_waveform,metrics='mse', dist_model=None, dist_threshold=0.05, 
                      return_prob=True):
    # Match a spike waveform to template
    # return the index of minimum template, as well as the distance
    # TODO: reject outliner
    
    
    if dist_model is not None:
        assert len(templates) == len(dist_model)

    if metrics=='mse':
        d = np.mean((spike_waveform - templates)**2,axis=1) # MSE is equivalent to Euclidean distance
    else:
     
        d = spatial.distance.cdist(spike_waveform, templates, metric='cosine')

    # d = np.sum(spike_waveform*templates)
    idx = np.argmin(d)
    min_d = np.min(d)
    
    prob = None
    
    # check the probability
    if dist_model is not None:
        prob = guassmf(min_d, dist_model[idx]['mean'], dist_model[idx]['std'])
        if prob<dist_threshold:
            # distance outside range
            idx = -1 # mark as outliner
            
    if return_prob:
        return idx, prob
    else:
        return idx


def align_spike(spikes, chan_per_electrode=4, search_span = 15, pre_peak_span=15, post_peak_span = 15,
    align_sign=-1):
    # align spikes to max absolute spike
    # spikes is in the form (index x time), if it is from tetrode, then the waveform of all channel should be already concatenated in the time axis
    # it finds the maximum point along the time axis, then shift all channel in the tetrode such that maximum is at the centre
    #TODO: potential racing condition when spike from two channel are of similar height

    if spikes.ndim <2:
        spikes = spikes.reshape(-1,len(spikes))
    spike_length = spikes.shape[1]//chan_per_electrode
    
    # fig,ax = plt.subplots(3,1,figsize=(4,6))
    # ax[0].plot(spikes.T)

    if align_sign == -1:
        spikes2 = spikes.copy()
        spikes2[spikes2>0] = 0

    abs_spike = np.abs(spikes2)
    # ax[1].plot(abs_spike.T)

    #Create a mask to restrict the search region
    # set the signal output the mask to be zero
    mask = np.zeros_like(abs_spike)
    search_region = [np.arange(spike_length/2-search_span,spike_length/2+search_span)
        +i*spike_length for i in range(chan_per_electrode) ]
    # print(search_region)
    search_region = np.concatenate(search_region).astype(int)
    mask[:,search_region] = 1

    #only search the masked region
    masked_spike = abs_spike*mask 
    # ax[2].plot(masked_spike.T)


    peak_idx = masked_spike.argmax(axis=1)
    # print(peak_idx)
    # plt.figure()
    # plt.plot(masked_spike)


    # find out which channel it is 
    chan_idx = peak_idx//spike_length
    idx_rel_chan = peak_idx%spike_length #index of peak within that channel
    spikes_channel = spikes.reshape(-1,chan_per_electrode,spike_length) # n x time x channel
    spikes_aligned = np.zeros((spikes.shape[0], chan_per_electrode*(pre_peak_span+post_peak_span)))

    # align each channel
    # pad the aligned signal with zero if the peak is too skewed to one side
    for i,idx in enumerate(idx_rel_chan):
        single_spike_aligned = np.zeros_like(spikes_aligned[0,:])

        start_idx = max(idx-pre_peak_span,0) #first half of spike
        end_idx =  min(idx+post_peak_span, spike_length) #second half of spike
        spk = spikes_channel[i, :, start_idx:end_idx]
        
        #pad the spike with zero if it is too skewed to one side
        first_half_length = idx-start_idx
        second_half_length = end_idx-idx
        if first_half_length > second_half_length:
            #pad the second half
            spk = np.pad(spk, [[0,0],[0, first_half_length - second_half_length]])
        elif first_half_length < second_half_length:
            spk = np.pad(spk, [[0,0],[second_half_length - first_half_length,0]])

        spikes_aligned[i,:] = spk.ravel() #flatten

    return spikes_aligned


def makeSpikeDataframe(spikeEvent):
    """Convert received spike event into a dataframe for sorting

    Args:
        spikeEvents (list): list of spike events received from ZMQ
    """
    spikedata = [s.data for s in spikeEvent]
    spikedata = np.stack(spikedata)
    electrode_ids = [s.electrode_id for s in spikeEvent]
    timestamps = [s.timestamp for s in spikeEvent]
    acq_timestamps = [s.acq_timestamp for s in spikeEvent]
    # print(f'Total spike detected {len(spikeEvent)}')
    # %%
    spikedata_flat = spikedata.reshape(spikedata.shape[0],-1) 

    #%%
    spikelength = spikedata.shape[2]
    spikeN = spikedata.shape[0]
    chanN = spikedata.shape[1]
    assert chanN < spikelength, f'Error, the spike shape does not seem right, it should be (N, chanN, samples) now it is {spikedata.shape}'
    channel_id = np.repeat(np.arange(chanN),spikelength)
    channel_ids = np.tile(channel_id, (spikeN,1) )

    t = np.arange(spikedata_flat.shape[1])
    t = np.tile(t, (spikeN,1))
    df = pd.DataFrame({
        'spike_waveform': pd.Series(list(spikedata_flat)),
        'time': pd.Series(list(t)),
        'electrode_ids': electrode_ids,
        'channel_ids':pd.Series(list(channel_ids)),
        'spike_id': np.arange(spikeN),
        'timestamps':timestamps,
        'acq_timestamps': acq_timestamps
    })

    return df 


def makeChannelIdsCol(spikeLength, chan_per_electrode,spikeN):
    #make new channel ids column with new spike length
    channel_ids = np.repeat(np.arange(spikeLength//chan_per_electrode),chan_per_electrode)
    channel_ids = np.tile(channel_ids, (spikeN,1) )
    return pd.Series(list(channel_ids))

def makeSpikeTimeCol(spikeLength,spikeN):
    # make new time col with new spike length
    t = np.arange(spikeLength)
    t = np.arange(spikeLength)
    t = np.tile(t, (spikeN,1))
    return pd.Series(list(t))

def addAlignedSpike2df(df,channel_per_electrode=4):
    #Aligned spike and add to dataframe
    spike_waveforms = np.stack(df.spike_waveform.values)
    spike_waveforms = align_spike(spike_waveforms)
    spikeN,spikeLength = spike_waveforms.shape
    df['spike_waveform_aligned'] = spike_waveforms.tolist() #don't create pd.Series first, will lead to bug if df index has duplicates
    df['time_aligned'] = makeSpikeTimeCol(spikeLength,spikeN)
    df['channel_ids_aligned'] = makeChannelIdsCol(spikeLength, channel_per_electrode,spikeN)
    # assert False
    return df


def checkDupliate(x):
    u,c = np.unique(x[:,0], return_counts=True)
    dup_val = u[c>1]
    if len(dup_val)>0:
        return x[x[:,0]==dup_val[0]]


def sort_all_electrodes(df,channel_per_electrode=4,do_align_spike=True, eps = 1, pca_component=6, verbose=False):
    elect_ids = df.electrode_ids.unique()

    if do_align_spike:
        if verbose:
            print('Aligning spikes')
        df = addAlignedSpike2df(df, channel_per_electrode)

    cluster_start_count = 0

    #make pca columns
    for i in range(pca_component):
        df[f'PC{i}'] = np.nan
        
    df['pc_norm'] = np.zeros((len(df),pca_component)).tolist()

    pca_transformers = {}
    standard_scalers = {}
    pc_norm_list = []

    for e_ids in elect_ids:
        if verbose:
            print(f'Sorting electrode {e_ids}')
        # Sort each tetrode respectively
        # Sorting is done on each tetrode
        df_elec = df[df.electrode_ids==e_ids]

        if do_align_spike:
            spike_waveforms = np.stack(df_elec.spike_waveform_aligned.values)
        else:
            spike_waveforms = np.stack(df_elec.spike_waveform.values)
            
        # sort spikes
        print(spike_waveforms.shape)
        labels,pc_norm,pca_transformer,standard_scaler = sort_spikes(spike_waveforms,eps=eps, pca_component=pca_component)
        pca_transformers[e_ids] = pca_transformer
        standard_scalers[e_ids] = standard_scaler
        
        #
        if labels is not None:
            df.loc[df.electrode_ids==e_ids,'cluster_id'] = [int((e_ids+1)*100 +(l+1)) for l in labels] #make unique id for each electrode
        else:
           df.loc[df.electrode_ids==e_ids,'cluster_id'] = -1
        
        # Need to make an array of list, so that we can assign it to the dataframe column
        if pc_norm is not None:
            # sometimes there may not be enough spike to calcualte the PCA
            pc_norm_list = np.empty(pc_norm.shape[0], dtype=object)
            pc_norm_list[:] = pc_norm.tolist()
            df.loc[df.electrode_ids==e_ids,'pc_norm'] = pc_norm_list
        
        # assert np.all([len(df.iloc[i].pc_norm)==pca_component for i in range(len(df))])
            

    return df, pca_transformers,standard_scalers


def sort_spike_row(row, templates, template_cluster_id, template_electrode_id, pca_transformers, standard_scalers, 
                   dist_model=None, dist_threshold = 0.05):
    # print(row)
    t = time.perf_counter()
    spike = row.spike_waveform
    electrode_id = row.electrode_ids
    spike = align_spike(spike)
    aligned_waveform = spike.squeeze()
    neigbour_idx = np.where(template_electrode_id == electrode_id)[0] # do sorting independently for each electrode
    dist_model_neighbour = [dist_model[i] for i in neigbour_idx]
    
    if len(neigbour_idx) > 0:
        idx, prob = template_matching(templates[neigbour_idx,:], spike, dist_model=dist_model_neighbour,
                                dist_threshold=dist_threshold,
                                return_prob=True)
        if idx>=0:
            labels_template = template_cluster_id[neigbour_idx[idx]]
        else:
            #outliner
            labels_template = '-1'
    else:
        # no template found for a particular electrode
        # make them unclassified
        labels_template = '-1'
        prob = 0

    if pca_transformers is not None:
        #Also do the PCA transform for monitoring purpose
        if electrode_id in pca_transformers and pca_transformers[electrode_id] is not None:
            pca_transformer = pca_transformers[electrode_id]
            pc = pca_transformer.transform(spike.reshape(1,-1))
            pc_norm = standard_scalers[electrode_id].transform(pc).ravel()
        else:
            pc_norm = None
    else:
        pc_norm = None
            
           

    return labels_template,aligned_waveform, time.perf_counter() - t, pc_norm, prob

def template_match_all_electrodes_fast(df, templates, template_electrode_id,template_cluster_id,
                                       dist_model=None, dist_threshold=0.05,
                                       pca_transformers=None, standard_scalers=None):
    ''' 
    Match input spike wavefrom to the closest template, matching is done on each tetrode independently
    '''
    # apply is faster than iterrows()
    results = df.apply(sort_spike_row,axis=1, 
        args=(templates, template_cluster_id, template_electrode_id, pca_transformers, standard_scalers, dist_model, dist_threshold), result_type='expand')
    df[['cluster_id', 'spike_waveform_aligned','sorting_time', 'pc_norm', 'prob']] = results

    df.cluster_id = df.cluster_id.astype(int)

    return df 

def template_match_all_electrodes_np(df, templates, template_electrode_id, template_cluster_id, 
                pca_transformers = None, standard_scalers = None):
    ''' 
    Match input spike wavefrom to the closest template, matching is done on each tetrode independently
    '''

    labels_template=[]
    norm_dist_template = []
    pc_norms = []
    
    
    spikes = np.stack(df.spike_waveform.to_numpy())
    electrode_id = df.electrode_ids.to_numpy()
    aligned_spikes = align_spike_cy(spikes)
    neigbour_idx = np.where(template_electrode_id == electrode_id)[0] # do sorting independently for each electrode


    for _,row  in df.iterrows():
        # align the spike, separate them into their respective tetrodes
        # then match with templates
        # spike = row.spike_waveform
        # electrode_id = row.electrode_ids
        # spike = align_spike(spike)
        # aligned_waveforms.append(spike.squeeze())
        neigbour_idx = np.where(template_electrode_id == electrode_id)[0] # do sorting independently for each electrode

        if len(neigbour_idx) > 0:
            idx,norm_dist = template_matching(templates[neigbour_idx,:], spike)
            labels_template.append(template_cluster_id[neigbour_idx[idx]])
            norm_dist_template.append(norm_dist)
        else:
            # no template found for a particular electrode
            # make them unclassified
            labels_template.append('noise')
            norm_dist_template.append(np.nan)

        if pca_transformers is not None:
            #Also do the PCA transform for monitoring purpose

            if electrode_id in pca_transformers and pca_transformers[electrode_id] is not None:
                pca_transformer = pca_transformers[electrode_id]
                pc = pca_transformer.transform(spike.reshape(1,-1))
                pc_norm = standard_scalers[electrode_id].transform(pc)
                pc_norms.append(pc_norm.ravel())
           
            else:
                pc_norms.append(None)


    df['cluster_id'] = labels_template #make plotting program aware this is a categorical variable
    df['dist_tm'] = norm_dist_template
    df['spike_waveform_aligned'] = aligned_spikes.tolist()
    if pca_transformers is not None:
        df['pc_norm'] = pc_norms

    return df 

def template_match_all_electrodes(df, templates, template_electrode_id, template_cluster_id, 
                pca_transformers = None, standard_scalers = None):
    ''' 
    Match input spike wavefrom to the closest template, matching is done on each tetrode independently
    '''

    labels_template=[]
    norm_dist_template = []
    aligned_waveforms = []
    pc_norms = []

    for _,row  in df.iterrows():
        # align the spike, separate them into their respective tetrodes
        # then match with templates
        spike = row.spike_waveform
        electrode_id = row.electrode_ids
        spike = align_spike(spike)
        aligned_waveforms.append(spike.squeeze())
        neigbour_idx = np.where(template_electrode_id == electrode_id)[0] # do sorting independently for each electrode

        if len(neigbour_idx) > 0:
            idx,norm_dist = template_matching(templates[neigbour_idx,:], spike)
            labels_template.append(template_cluster_id[neigbour_idx[idx]])
            norm_dist_template.append(norm_dist)
        else:
            # no template found for a particular electrode
            # make them unclassified
            labels_template.append('noise')
            norm_dist_template.append(np.nan)

        if pca_transformers is not None:
            #Also do the PCA transform for monitoring purpose

            if electrode_id in pca_transformers and pca_transformers[electrode_id] is not None:
                pca_transformer = pca_transformers[electrode_id]
                pc = pca_transformer.transform(spike.reshape(1,-1))
                pc_norm = standard_scalers[electrode_id].transform(pc)
                pc_norms.append(pc_norm.ravel())
           
            else:
                pc_norms.append(None)


    df['cluster_id'] = labels_template #make plotting program aware this is a categorical variable
    df['dist_tm'] = norm_dist_template
    df['spike_waveform_aligned'] = aligned_waveforms
    if pca_transformers is not None:
        df['pc_norm'] = pc_norms

    return df 


def sort2spiketrain(unique_cluster_id, cluster_ids, spiketime,bins):
    #convert sorted clusters to spiketrain
    #TODO: better way to handle channels that has no spikes

    spike_train = np.zeros((len(unique_cluster_id),
        len(bins)-1))
    spike_time_event = []
    for i,cid in enumerate(unique_cluster_id):
        st = spiketime[cluster_ids==cid]
        spike_time_event.append(st)
        spike_train[i,:] = np.histogram(st,bins)[0]
        if not spike_train[i,:].sum() == len(st):
            warnings.warn(f'Some spikes not processed. id: {spike_train[i,:].sum()} vs {len(st)} \n {st} {bins}')

    return spike_train, spike_time_event


def simpleDownSample(x, segment_size):
    # x should be in (time, channel)
   # perform down sample by taking the mean over a segment indicated by segment_size
    # leftover data will be returned 
    (length,channel) = x.shape

    p = length//segment_size

    if p == 0:
        # if input is too small, return all as leftover
        ds_x = None
        leftover = x
    else:
        x2 = x[:segment_size*p,:]
        leftover = x[segment_size*p:x.shape[0],:]
        
        ds_x = x2.reshape(-1,segment_size,channel).mean(1)

    return (ds_x,leftover)

def is_sorting_converged(id_old, id_new, thres):
    # Determine if the two sorting are similar
    pass

def makeTuningDataframe(position,fr,time_bin):
    """
    fr should be in (time x neuron)
    position should be in (time x 1)
    """
    length2concat = min(position.shape[0],fr.shape[0])

    fr = fr[:length2concat,:]
    position = position[:length2concat,:]
    
    #make dataframe for easier plotting
    df = pd.DataFrame(fr/time_bin)
    pos = np.around((position-position.min())/(position.max()-position.min())*200)
    pos = pos.squeeze()
    df['position'] = pd.cut(pos,50,labels=False)*4 # convert to cm,bin into 50 bins
    df = pd.melt(df,id_vars=['position'],var_name='neuron',value_name='firing_rate')

    return df 

def sort_spikes_online(df_ref, df2sort, eps=1, pca_component=6):
    df_ref = df_ref.copy()
    df2sort = df2sort.copy()
    df_sorted,pca_transformers,standard_scalers = sort_all_electrodes(df_ref, eps=eps, pca_component=pca_component)

    # calculate templates
    df_cluster = df_sorted.groupby('cluster_id').mean().reset_index()
    cluster_electrode_ids = dict(zip(df_cluster.cluster_id.values, df_cluster.electrode_ids.values))

    spike_waveforms = np.stack(df_sorted.spike_waveform_aligned.values)
    cluster_id = df_sorted.cluster_id.values

    # Calculate spike templates
    templates,template_cluster_id = calculate_spike_template(spike_waveforms,cluster_id)
    template_electrode_id = np.array([int(cluster_electrode_ids[i]) for i in template_cluster_id])
    

    # template matching
    df_sorted_all = template_match_all_electrodes(df2sort, templates, template_electrode_id,template_cluster_id)


    #also calculate the principle component 

    # Add principle component columns
    for i in range(pca_component):
        df_sorted_all[f'PC{i}'] = np.nan

    for e_ids in df_sorted_all.electrode_ids.unique():
        spike_waveforms = np.stack(df_sorted_all[df_sorted_all.electrode_ids==e_ids].spike_waveform.values)
        spike_waveforms = align_spike(spike_waveforms)
        # print(spike_waveforms.shape)
        pc = pca_transformers[e_ids].transform(spike_waveforms)
        df_sorted_all.loc[df_sorted_all.electrode_ids==e_ids, [f'PC{i}' for i in range(pca_component)]] = pc

    return df_sorted, df_sorted_all,template_cluster_id



def findMatch(y,x):
    #match position of x in y
    # assuming it is a tetrode
    for j in range(4):
        for i in range(y.shape[1]-x.shape[1]):
            if np.allclose(y[j*4:j*4+4,i:i+x.shape[1]],x):
                return i,j
    
    raise ValueError("Cannot find the sub-matrix")

def syncSpikeTime(df_spike_in,data,spikeIdx = 0, searchLength=5000,noChan=4):
    #synchronize the time stamp of the spike and data
    
    df_spike = df_spike_in.copy()
    first_spike_waveform = df_spike.iloc[spikeIdx].spike_waveform.reshape(noChan,-1)
    first_spike_ts = df_spike.iloc[spikeIdx].timestamps
    spike_length = first_spike_waveform.shape[1]

    idx,electrode = findMatch(data[:,:searchLength], first_spike_waveform)
    print(f'Match found at {idx} on tetrode {electrode}')
    df_spike.timestamps = df_spike.timestamps - first_spike_ts +  idx + spike_length/2

    return df_spike, idx + spike_length/2

def plot_multichannel(data,scale=30,normalized=True):
    # data should be in (channel x time)
    for i in range(data.shape[0]):
        if normalized:
            x= data[i,:]
            x = x/x.std()
            plt.plot(x+i*scale)
        else:
            plt.plot(data[i,:]/scale+i*scale)

def plot_spikeEvent(data,spiketime,plot_range,electrodes=None,
        scale = 10, figsize=(15,6),**kwargs):
    plt.figure(figsize=figsize)
    spiketime = spiketime[spiketime<plot_range]

    for i in range(len(spiketime)):

        ymin = electrodes[i]*0.25
        ymax = electrodes[i]*0.25 + 0.25
        plt.axvline(spiketime[i], ymin=ymin,
            ymax=ymax,**kwargs)

    plot_multichannel(data[:,:plot_range],scale)

class SpikeEvent:
    def __init__(self,electrode_id, timestamp,spikedata):
        self.electrode_id = electrode_id
        self.timestamp = timestamp
        self.data = spikedata
        self.acq_timestamp = None
    
    def __str__(self):
        return f'electrode_id: {self.electrode_id}, timestamp: {self.timestamp}, spikedata: {self.data.shape}'
        

def detectSpike(data,factor=4, method='median', useAbsoluteThres=False, preSample=20, postSample=20, ntetrode=4):
    # Detect spike based on threshold, can either be median or absolute
    # input data should be in the shape of (channel x time)
    
    spikes_list = []
    sampleIndex = preSample
    absData = np.abs(data)
    
    if method=='median':
        m = np.median(absData,axis=1) 
        print(f'Spike threshold: {m}')
    
    for electrode in tqdm(range(ntetrode)):
        sampleIndex = preSample
                
        # with tqdm(total=absData.shape[1]) as pbar:
        while(sampleIndex<absData.shape[1]-postSample-1):

            for channel in range(4):
                idx = electrode*4+channel

                if method=='median':
                    thres = factor*m[idx]
                else:
                    thres = factor
                
                # At the current sample index, check all channel to see if any data point exceed the threshold
                if(data[idx,sampleIndex]<-thres): #only detect negative spike

                    #trigger spike
                    #find maximum
#                         peakIdx = np.argmax(absData[idx,(sampleIndex-preSample):(sampleIndex+postSample)])
                    # peakIdx = np.argmin(data[idx,(sampleIndex-preSample):(sampleIndex+postSample)])
                    # peakIdx = sampleIndex-preSample+peakIdx #shift to refer to the whole array
                    
                    # do not align spike at this stage, to avoid overlapping spike causing issue
                    # otherwise two spike may have the same timestamp
                    peakIdx = sampleIndex 
                    
#                     print(peakIdx)
                    # record down the spike waveform
                    spikedata = data[electrode:electrode+4, (peakIdx-preSample):(peakIdx+postSample)]

                    spikeEvent = SpikeEvent(electrode,peakIdx,spikedata)
                    # if peakIdx<1000:
                    #     print(peakIdx, sampleIndex)
                    spikes_list.append(spikeEvent)

                    sampleIndex += postSample


                    break #break the channel loop
                else:
                    sampleIndex += 1

        # pbar.n = absData.shape[1]
        # pbar.refresh()  
            
    return spikes_list

def getSpikeWaveform(spiketime,electrode,neuroData,preSample=20,postSample=20):
    # extract waveform from neuro data given spiketiming
    spikes = []
    spiketime = spiketime[(spiketime>preSample) & (spiketime< (neuroData.shape[1]-postSample))] #make sure the spiketime is within extractable range
    #extract spikew waveform
    for t in spiketime:
        s = neuroData[electrode*4:electrode*4+4, t-preSample:t+postSample]
        spikes.append(s)


    return np.stack(spikes), spiketime

def spikeWaveform2DF(waveforms):
    # waveforms: list containing spike waveform array of (index x channel x time)
    df_waveform = []
    for cluster_id, waveform in enumerate(waveforms):

        spike_num = np.repeat(np.arange(waveform.shape[0]),waveform.shape[1]*waveform.shape[2])
        channel = np.tile(np.repeat(np.arange(waveform.shape[1]),waveform.shape[2]), waveform.shape[0])
        time = np.tile(np.arange(waveform.shape[2]),waveform.shape[0]*waveform.shape[1])
        time_linear = np.tile(np.arange(waveform.shape[1]*waveform.shape[2]),waveform.shape[0])

        df_temp = pd.DataFrame({'waveform':waveform.ravel(), 
                                'spike_num': spike_num, 'channel':channel,
                                'time':time, 'time_linear':time_linear})
        df_temp['cluster_id'] = cluster_id
        df_waveform.append(df_temp)

    return pd.concat(df_waveform)

def loadPackets(filename):
    #load saved data packet
    data_list = []
    segment_count = 0
    with open(filename,'rb') as f:
        while True:
            try:
                data_list.append(pickle.load(f))
                segment_count +=1
            except EOFError:
                break

        print(f'Reached end. Total segment: {segment_count}')
    return data_list



def compute_whitenMatrix(data, return_intermediate=False):
    # some code adopted from spikeinterface
    data = data - np.mean(data, axis=1, keepdims=True)
    AAt = data @ np.transpose(data)
    AAt = AAt / data.shape[1]
    U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
    W = (U @ np.diag(1 / np.sqrt(S))) @ Ut

    if return_intermediate:
        # return intermidate results for debugging purpose
        return W, AAt, U, S, Ut
    else:
        return W

def applyWhitening(data,W):
    data = data - np.mean(data, axis=1, keepdims=True)
    return W@data

def detectSpikePerChannel(data,detect_span=40,factor=3, use_median=True):
    total_length = int(np.ceil(data.shape[1]/detect_span)*detect_span)
    absData = np.abs(data)
    m = np.median(absData,axis=1)
    # print(m)
    spiketime = []
    for i in tqdm(range(data.shape[0])):
        
        x = np.pad(absData[i,:],(0,total_length-absData.shape[1])) #make sure it is divided by the detect_span
        
        # break into segment then find the minimum
        x = x.reshape(-1,detect_span)
        max_idx = np.argmax(x,axis=1)
        maxValue = x[range(len(max_idx)),max_idx]
        
        #convert the max_idx to linear index by adding the index of the
        # begining of the window
        max_idx += np.arange(len(max_idx))*detect_span
        
        #filter by max size
        if use_median:
            max_idx = max_idx[maxValue>m[i]*factor]
        else:
            max_idx = max_idx[maxValue>factor]
        
        
        #remove close peak
        max_idx = max_idx[np.diff(max_idx,prepend=0)>detect_span]
        
        spiketime.append(max_idx)
        
    return spiketime

