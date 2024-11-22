import os
import random
import numpy as np
import deepdish as dd
from itertools import permutations
from brainiak.eventseg.event import EventSegment

def get_ev_perm_table(params,sl):
    """
    Generate a permutation table for event patterns based on the specified number of events and permutations. 
    The table includes random permutations of event sequences, ensuring they are distinct from the original sequence.

    Parameters
    ----------
    params : dict
        Configuration parameters, including:
        - the range of event counts ('nEvents')
        - the number of permutations ('event_PERMS').
    sl : string
        Searchlight identifier, used to seed the random number generator.
    
    Returns
    ----------
    ev_perm_table : list of ndarray
        A list of arrays, each containing permutations of event sequences.  
    """

    # Create a deterministic seed for reproducibility based on the searchlight identifier
    seed = hash(sl) % (2**32)
    random.seed(seed)

    ev_perm_table = []
    for number in params['nEvents']:
        values = np.arange(number)
        all_perms = list(permutations(values))
        non_identical_perms = [perm for perm in all_perms if not np.array_equal(perm, values)]
        random.shuffle(non_identical_perms)

        # Select the first permutation (original order) and a subset of shuffled permutations
        selected_perms = [values] + non_identical_perms[:params['event_PERMS']]

        ev_perm_table.append(np.array(selected_perms))
    return ev_perm_table

def prepare_data(sl, version, roi=None):
    """
    Load data for a specified searchlight (sl) and version of the experiment (e.g., 'Intact' or 'Scrambled'). 
    The function processes data from a designated ROI (Region of Interest) if specified, and returns the data in a structured format for further analysis. 
    It separates the data for two versions of the scrambled movie when applicable.

    Parameters
    ----------
    sl : string
        Searchlight identifier used to locate the corresponding data file.
    version : string
        Version of the data to load (e.g., 'Intact' or 'Scrambled').
    roi : string, optional
        The region of interest to load data from. If not specified, the default searchlight directory is used.

    Returns
    ----------
    list of ndarray
        A list containing the processed data arrays. If the version is 'Scrambled', the data will be split into two arrays corresponding to each version of the movie.

    """

    # Directory containing .h5 files
    ####directory_path = './sl_data/'
    ####file_path = directory_path + sl + '.h5'

    if roi:
        directory_path = f'./ROIs/sl/{roi}/'
    else:
        directory_path = '/data/gbh/data/SL/'
    file_path = directory_path + sl + '.h5'

    # Create intact Data
    D_orig = dd.io.load(file_path)
    N_vox = D_orig[list(D_orig.keys())[0]][f'{version}'].shape[2]
    print("num of voxels:", N_vox)
    if N_vox == 0:
        print("zero voxels")
        return None
    N_subj = len(D_orig.keys())

    D = np.zeros((N_subj, 6, 60, N_vox))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s][f'{version}']

    """

    average_data = np.mean(D, axis=0)  # Shape: [viewing, timepoint, voxel]
    valid_voxels = np.any(average_data != 0, axis=1)
    filtered_data = []
    for view in range(D.shape[1]):
        view_data = D[:, view, :, valid_voxels[view]]  # Shape: [subject, timepoint, valid_voxels]
        filtered_data.append(view_data)
    D = np.stack(filtered_data, axis=1)
    print("new voxels:", D.shape[2])
    """

    if version != 'Intact':
        # Separate data for the two versions of the scrambled movie
        D_v1 = D[:N_subj//2]  # Data for the first half of the subjects
        D_v2 = D[N_subj//2:]  # Data for the second half of the subjects

        return [D_v1, D_v2]
    
    else:
        return [D]


def heldout_ll(hmm1, hmm2, group1, group2, params):
    """
    Calculate the held-out log-likelihood (LL) for two groups of subjects, comparing the performance of two Hidden Markov Models (HMMs) trained on different groups. 
    This function can also calculate LL on a per-subject basis or across the groups' averages, depending on the parameters.

    Parameters
    ----------
    hmm1 : EventSegment
        HMM model trained on the first group of subjects.
    hmm2 : EventSegment
        HMM model trained on the second group of subjects.
    group1 : ndarray
        Data for the first group of subjects with shape [timepoints, voxels].
    group2 : ndarray
        Data for the second group of subjects with shape [timepoints, voxels].
    params : dict
        Configuration parameters, including 'perSubject', which determines whether to compute LL per subject or on average.

    Returns
    ----------
    ll : ndarray
        Array containing the log-likelihood values for each subject or group, depending on the perSubject setting.

    """

    N_subj = group1.shape[0] + group2.shape[0]
    ll = np.empty((N_subj))
    
    if params['perSubject']==1:
        for s in range(group2.shape[0]):
            _, ll_s = hmm1.find_events(group2[s])
            ll[s] = ll_s
        for s in range(group1.shape[0]):
            _, ll_s = hmm2.find_events(group1[s])
            ll[s+group2.shape[0]] = ll_s
    else:
        _, ll12 = hmm1.find_events(group2.mean(0))
        _, ll21 = hmm2.find_events(group1.mean(0))
        ll = (ll12 + ll21)/2

    return ll


def get_timescales(group1, group2, params, ev_perm_table):

    """
    Compute the log-likelihood (LL) of event segmentation for two groups, using permutations to evaluate the significance of the results. 
    This function trains and tests HMM models across different permutations of the event patterns, returning the log-likelihood values across permutations.

    Parameters
    ----------
    group1 : ndarray
        Data for the first group of subjects with shape [subjects, timepoints, voxels].
    group2 : ndarray
        Data for the second group of subjects with shape [subjects, timepoints, voxels].
    params : dict
        Configuration parameters, including:
        - the range of event counts ('nEvents')
        - number of permutations ('event_PERMS')
        - viewings ('VIEWINGS')
    ev_perm_table : list
        A table of event permutations used to test the robustness of the event patterns.

    Returns
    ----------
    LLs : ndarray
    Array of log-likelihood values, with dimensions corresponding to permutations, viewings, and event_counts.

    """

    LLs = np.empty((params['event_PERMS']+1,
                    params['VIEWINGS'],
                    len(params['nEvents'])), dtype=object)

    ###############
    D = group1
    average_data = np.mean(D, axis=0)  # Shape: [viewing, timepoint, voxel]
    valid_voxels_group1 = np.all(np.any(average_data != 0, axis=1), axis=0)  # Shape: [voxel]
    D = group2
    average_data = np.mean(D, axis=0)  # Shape: [viewing, timepoint, voxel]
    valid_voxels_group2 = np.all(np.any(average_data != 0, axis=1), axis=0)  # Shape: [voxel]
    valid_voxels = valid_voxels_group1 & valid_voxels_group2  # Shape: [voxel]
    group1_filtered = group1[:, :, :, valid_voxels]  # Shape: [subject, viewing, timepoint, common_valid_voxels]
    group2_filtered = group2[:, :, :, valid_voxels]  # Shape: [subject, viewing, timepoint, common_valid_voxels]
    group1 = group1_filtered
    group2 = group2_filtered
    print(group1.shape, group2.shape)

    g1_data = group1.mean(0)
    g2_data = group2.mean(0)

    for viewing in range(params['VIEWINGS']):

        for ev_i, n_ev in enumerate(params['nEvents']):

            # Train and test event segmentation across groups
            hmm1 = EventSegment(n_ev).fit(g1_data[viewing])
            hmm2 = EventSegment(n_ev).fit(g2_data[viewing])

            # Save original event pattern
            orig_ev_pat_1 = hmm1.event_pat_.copy()
            orig_ev_pat_2 = hmm2.event_pat_.copy()

            max_n_ev_perms = min(np.math.factorial(n_ev), params['event_PERMS'] + 1)
            for ev_perm_i in range(max_n_ev_perms):

                # Get permutation of event patterns
                [perm_hmm1, perm_hmm2] = get_events_perm([hmm1, hmm2], [orig_ev_pat_1, orig_ev_pat_2], ev_perm_i, ev_perm_table[ev_i])
                # Train and test event segmentation across groups
                LLs[ev_perm_i, viewing, ev_i] = heldout_ll(perm_hmm1, perm_hmm2, group1[:, viewing], group2[:,viewing], params)

    return LLs

def get_events_perm(hmms, orig_ev_pats, ev_perm_i, rand_ev_perms):

    """
    Set the event patterns of HMMs based on a specific permutation of the original event patterns. 
    This function applies a given permutation of event patterns to the HMM models, modifying their internal event patterns for subsequent training and evaluation.

    Parameters
    ----------
    hmms : list of EventSegment
        List of HMM models (corresponding to the two groups of participants) to modify.
    orig_ev_pats : list of ndarray
        List of original event patterns for each HMM model.
    ev_perm_i : int
        Index of the current permutation.
    rand_ev_perms : ndarray
        Array of random permutations for the event patterns.    

    Returns
    ----------
    hmms : list of EventSegment Models 
        Modified list of HMM models corresponding to the two groups of participants, with updated event patterns.
    """

    if ev_perm_i > 0:
        for i, hmm in enumerate(hmms):
            hmm.set_event_patterns(orig_ev_pats[i][:, rand_ev_perms[ev_perm_i]])
    return hmms

def get_data_perm(data, perm_i):

    """
    Permute the clip viewing order for each subject. 
    This function ensures that the data is shuffled consistently for each subject and permutation, allowing for repeated randomization in analyses.

    Parameters
    ----------
    data : ndarray
        The data array to permute, with dimensions [subjects, clip_viewings, timepoints, voxels].
    perm_i : int
        Index of the current permutation of the clip viewing order .

    Returns
    ----------
    perm_data : ndarray
        The permuted data array, with dimensions [subjects, clip_viewings, timepoints, voxels].
    """

    perm_data = np.copy(data)
    if perm_i > 0:
        for subj in range(data.shape[0]):
            seed = subj + perm_i * 1000
            np.random.seed(seed)
            np.random.shuffle(perm_data[subj])
    return perm_data

