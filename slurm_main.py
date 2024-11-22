import os
import sys
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import get_ev_perm_table, prepare_data, get_data_perm, get_timescales

version = sys.argv[1]
sl = sys.argv[2]
ti = time.time()

if os.path.exists(f'./{version}/LLs/{sl}.npy'):
    print(f"SL {sl} already exists. Exiting.")
    sys.exit(0)

print(f"Version: {version}, SL: {sl}")

# initialize some variables
params = {
    'test_size': 0.5,
    'nEvents': np.arange(2, 11),
    'event_PERMS': 50,
    'views_PERMS': 50,
    'VIEWINGS': 6,
    'perSubject': 0,
    'subj_SPLITS': 5
}
ev_perm_table = get_ev_perm_table(params,sl)
if len(ev_perm_table) != len(params['nEvents']):
    raise ValueError("ev_perm_table length does not match nEvents length.")

####### Prepare data structures for all versions
data = prepare_data(sl, version)

if version != 'Intact':
    versions_timescales = {'v1': None, 'v2': None}
    versions_LLs = {'v1': None, 'v2': None}

for v_i, D in enumerate(tqdm(data)):
    
    ######## Initialize data structures containing timescales for all versions
    nSubj = D.shape[0]
    LLs = np.empty((params['views_PERMS']+1,
                    params['subj_SPLITS'],
                    params['event_PERMS']+1,
                    params['VIEWINGS'],
                    len(params['nEvents'])), dtype=object)

    ############### Enter permutations loop
    
    for views_perm in range(params['views_PERMS']+1):

        perm_D = get_data_perm(D, views_perm)
        
        for subj_split in range(params['subj_SPLITS']):

            ######### Train-test split
            g1_indices, g2_indices = train_test_split(np.arange(nSubj), test_size=params['test_size'], random_state=42+subj_split) 
            LLs[views_perm, subj_split] = get_timescales(perm_D[g1_indices], perm_D[g2_indices], params, ev_perm_table)
            
    if version != 'Intact':
        versions_LLs[f"v{v_i+1}"] = LLs

saved_lls = LLs if version == 'Intact' else versions_LLs
np.save(f'./{version}/TS/{sl}.npy', saved_lls)

tf = time.time()
print(f"SL: {sl}, Time: {(tf - ti) / 60}")
