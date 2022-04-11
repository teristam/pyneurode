# cython: infer_types=True

import numpy as np 
cimport numpy as np
cimport cython 

@cython.boundscheck(False)
@cython.wraparound(False)
def align_spike_cy(spikes, int chan_per_electrode=4, int search_span = 15, int pre_peak_span=15, int post_peak_span = 15,
    int align_sign=-1):
    # align spikes to max absolute spike
    # spikes is in the form (index x time), if it is from tetrode, then the waveform of all channel should be already concatenated in the time axis
    # it finds the maximum point along the time axis, then shift all channel in the tetrode such that maximum is at the centre
    #TODO: potential racing condition when spike from two channel are of similar height
    
  

    assert spikes.ndim == 2, 'Spikes must be a 2d array'
    assert spikes.dtype == np.float32, 'Spike datatype must be np.float32'
        
    cdef int nspikes, spike_total_length
    nspikes = spikes.shape[0]
    spike_total_length = spikes.shape[1]

    if align_sign == -1:
        # only align to the negative peaks
        spikes2 = spikes.copy()
        spikes2[spikes2>0] = 0
    else:
        spikes2 = spikes
    
    abs_spike = np.abs(spikes2)
    
    cdef int spike_length # length of each waveform
    spike_length = spikes.shape[1]//chan_per_electrode
    
    #Create a mask to restrict the search region
    # set the signal output the mask to be zero
    mask = np.zeros_like(abs_spike)
    search_region = [np.arange(spike_length/2-search_span,spike_length/2+search_span)
        +i*spike_length for i in range(chan_per_electrode) ]
    search_region = np.concatenate(search_region).astype(int)
    mask[:,search_region] = 1

    #only search the masked region
    masked_spike = abs_spike*mask 

    peak_idx = masked_spike.argmax(axis=1)


    # find out which channel it is 
    chan_idx = peak_idx//spike_length
    idx_rel_chan = peak_idx%spike_length #index of peak within that channel
    
    cdef long long [:] idx_rel_chan_memview
    idx_rel_chan_memview = idx_rel_chan
    
    cdef np.float32_t [:,:,:] spikes_channel_memview
    cdef np.float32_t [:,:,:] spikes_aligned_memview
    spikes_channel = spikes.reshape(-1,chan_per_electrode,spike_length) # n x time x channel
    spikes_aligned = np.zeros((spikes.shape[0],
                               chan_per_electrode, (pre_peak_span+post_peak_span)),
                             dtype=np.float32)
    
    # use memory view to speed up computation
    spikes_channel_memview = spikes_channel
    spikes_aligned_memview = spikes_aligned
    

    # align each channel
    # pad the aligned signal with zero if the peak is too skewed to one side
    
    cdef int i, idx
    
    for i in range(nspikes):
        idx = idx_rel_chan_memview[i]
        
        start_idx = max(idx-pre_peak_span,0) #first half of spike
        end_idx =  min(idx+post_peak_span, spike_length) #second half of spike
        
        #make sure the peak is aligned at the centre
        copy_len = (end_idx-start_idx)
        
        # take care of the case when copy_len is odd
        first_half_length = idx-start_idx
        second_half_length = end_idx-idx
        spikes_aligned_memview[i,:, (pre_peak_span-first_half_length):(pre_peak_span+second_half_length)] = spikes_channel_memview[i, :, start_idx:end_idx]
        
    return spikes_aligned.reshape(nspikes,-1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def template_matching_mse_cy(np.float32_t [:,:] templates, np.float32_t [:,:] spike_waveforms, 
                                    np.int_t [:] template_electrode_ids, np.int_t [:] electrode_ids):
    # Calculate the MSE in pure C for speedup
    
    n_templates = templates.shape[0]
    n = templates.shape[1]
    n_spikes = spike_waveforms.shape[0]
    
    # initialize array for results
    cluster_ids = np.zeros((n_spikes,),dtype=np.int)
    cdef int[:] cluster_ids_view = cluster_ids
    
    cdef float mse = 0
    cdef float min_mse
    cdef int best_match_idx = 0
    cdef int i, j, electrode_template_idx, spike_i, electrode_id
    
    # for each spike waveform, compare with the corresponding template in the same electrode only
    for spike_i in range(n_spikes):
        electrode_id = electrode_ids[spike_i]
        best_match_idx = -1
        electrode_template_idx = 0 # index of template in the current electrodee
        min_mse = -1 
        
        for i in range(n_templates):
            if electrode_id == template_electrode_ids[i]:
                
                mse = 0

                for j in range(n):
                    mse = mse+ (spike_waveforms[spike_i, j] - templates[i,j])*(spike_waveforms[spike_i, j] - templates[i,j])

                mse = mse/n #avoid division by zero error check

                if min_mse < 0 or mse< min_mse: # first run or smaller
                    min_mse = mse
                    best_match_idx = electrode_template_idx
                
                electrode_template_idx += 1
                
                    
        cluster_ids_view[spike_i] = (electrode_id+1)*100 + (best_match_idx+1)

    return cluster_ids



def template_match_all_electrodes_cy(df, templates, template_electrode_id, template_cluster_id, 
                pca_transformers = None, standard_scalers = None):
    ''' 
    Match input spike wavefrom to the closest template, matching is done on each tetrode independently
    '''
    
    # cluster label is in the form of 101,102 etc. The x//100 represents the tetrode
    # x%100 represent the cluster in that tetrode

    if pca_transformers is not None:
        pc_norms = np.zeros((len(df), pca_transformers[0].n_components ))
    
    spikes = np.stack(df.spike_waveform.to_numpy())
    electrode_ids = df.electrode_ids.astype(np.int).to_numpy()
    aligned_waveforms = align_spike_cy(spikes)
    
    # Template match each electrode 
    cdef int i, eid
    
    labels_template = template_matching_mse_cy(templates, aligned_waveforms, template_electrode_id, electrode_ids)
    
    if pca_transformers is not None:
        for i in range(len(electrode_ids)):
            eid = electrode_ids[i]

            #Also do the PCA transform for monitoring purpose

            if eid in pca_transformers and pca_transformers[eid] is not None:
                pca_transformer = pca_transformers[eid]
                pc = pca_transformer.transform(aligned_waveforms.reshape(1,-1))
                pc_norm = standard_scalers[eid].transform(pc)
                pc_norms[i,:] = pc_norm.ravel()

            else:
                pc_norms[i,:] = -1


    df['cluster_id'] = labels_template #make plotting program aware this is a categorical variable
    df['spike_waveform_aligned'] = aligned_waveforms.tolist()
    if pca_transformers is not None:
        df['pc_norm'] = pc_norms

    return df 