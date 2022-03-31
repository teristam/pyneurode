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