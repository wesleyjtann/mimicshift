from mimic import utils
from mimic.advgen_mimic import *

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import pickle
import yaml


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

## Select dataset
dataset_dir = '3_testmimic_caida07' #'3_testmimic_cicFriday' #'3_testmimic_cicWed' #
log_num = 2

if dataset_dir == "3_testmimic_cicWed":
    ## cicWed
    '''Load config file'''
    dataset = 'hulk'
    with open('../'+dataset_dir+'/light_config_'+dataset+'.yaml', "r") as ymlfile:
        config = yaml.load(ymlfile)
    '''Load normal traffic data'''
    # df_normal = pd.read_csv("../"+ dataset_dir + "/ids_test2/artefact_" + dataset + "/N1.csv")
    ## new normal = atk data
    df_normal = pd.read_csv("../"+ dataset_dir + "/ids_test2/artefact/" + "/A1_full.csv")
else:
    ## cicFriday and caida07
    '''Load config file'''
    with open('../'+dataset_dir+'/light_config.yaml', "r") as ymlfile:
        config = yaml.load(ymlfile)
    '''Load normal traffic data'''
    # df_normal = pd.read_csv("../"+ dataset_dir + "/ids_test2/artefact/N1.csv")
    ## new normal = atk data
    df_normal = pd.read_csv("../"+ dataset_dir + "/ids_test2/artefact/" + "/A1_full.csv")



print("Data chunk size: " ,len(df_normal))
df_normal.head()


def getTokenizer(df) :
    ### Dictionary for Normal ###
    tokenizer = Tokenizer(filters='', split='<sep>', oov_token='<OTHERS>', lower=True, num_words=config['variablesHash']['inputHashRange']) 
    tokenizer.fit_on_texts(df['Input'].values)

    tokenizer.fit_on_texts(['<SOS>'])
    tokenizer.fit_on_texts(['<EOS>'])
    return tokenizer

def createGeneratorData(df, tokenizer, max_len) :
    #Prepare training for normal model
    x = []
    y = []

    for seq in df['Input']:
        x_windows, y_windows = prepare_sentence(seq, max_len, tokenizer)
        x += x_windows
        y += y_windows
    x = np.array(x)
    y = np.array(y)  # The word <PAD> does not constitute a class

#     x.shape = [len(x), max_len + 1, 1]
#     y.shape = [len(y), max_len + 1, 1]
    x.shape = [len(x), max_len, 1]
    y.shape = [len(y), max_len , 1]

    return x, y

def prepare_sentence(seq, maxlen, tokenizer):
    # Pads seq and slides windows
    seq = seq[:maxlen]
    seqX = seq #np.append(tokenizer.word_index['<sos>'], seq)
    seqY = seq #np.append(seq, tokenizer.word_index['<eos>'])

    x= pad_sequences([seqX],
        maxlen=maxlen, #maxlen+1,
        padding='post')[0]  # Pads before each sequence

    y= pad_sequences([seqY],
        maxlen=maxlen, #maxlen+1,
        padding='post')[0]  # Pads before each sequence

    return [x], [y]


max_len = config['SEQUENCELENGTH']
tokenizer_normal = getTokenizer(df_normal)
df_normal_embedded = df_normal.copy()
df_normal_embedded['Input'] = tokenizer_normal.texts_to_sequences(df_normal['Input'].values)

x_normal, y_normal = createGeneratorData(df_normal_embedded, tokenizer_normal, max_len)

def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat

def graph_from_scores(scores, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Parameters
    ----------
    scores: np.array of shape (N,N)
            The input transition scores.
    n_edges: int
             The desired number of edges in the target graph.

    Returns
    -------
    target_g: symmettic binary sparse matrix of shape (N,N)
              The assembled graph.

    """

    if  len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape) # initialize target graph
    scores_int = scores.toarray().copy() # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)   # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N): # Iterate the nodes in random order

        row = scores_int[n,:].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1


    diff = np.round((n_edges - target_g.sum())/2)
    if diff > 0:

        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g

def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric


num_nodes = len(np.unique(x_normal))

# Compute score matrix
# gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
gr = score_matrix_from_random_walks(np.array(x_normal).reshape([-1, max_len]), num_nodes)
gr = gr.tocsr() # shape=(N, N)                
print("gr: ", gr.shape)

# max number of edges
max_edges = (gr.shape[0] * (gr.shape[0]-1) )/2 #285390.0

## Assemble a graph from the score matrix
# n_edges = int((num_nodes)*2)
n_edges = int(max_edges/4)
_graph = graph_from_scores(gr, n_edges) # shape=(N, N)
print("_graph: ", _graph)

## Preparing data
Adjtraining = sp.csr_matrix(_graph, dtype='float64')
_A_obs = Adjtraining
_A_obs = _A_obs + _A_obs.T # 
_A_obs[_A_obs > 1] = 1 # Max value of 1 

""" Reduce input graph to a subgraph where only the nodes in largest n_components are kept. """ 
lcc = utils.largest_connected_components(_A_obs) # 
_A_obs = _A_obs[lcc,:][:,lcc] # 
_N = _A_obs.shape[0] # 
print("_N: ", _N)

""" Set the list of conditions """
# # randomly assigned conditions to num_classes
# num_classes = 3
# lcc_condlist = np.concatenate((np.arange(_N).reshape(_N,1), np.random.randint(num_classes, size=_N).reshape(_N,1)), axis=1) 
num_classes = 3
cond = np.genfromtxt ('../'+dataset_dir+'/packetlen_cond.csv', delimiter=",")
# cond = np.append(cond, np.zeros(_N-len(cond)))
lcc_condlist = np.concatenate((np.arange(_N).reshape(_N,1), cond.reshape(_N,1)), axis=1) 



#### Separate the edges into train, test, validation
val_share = 0.1
test_share = 0.05
seed = 2021 #  
"""
Split the edges of the adjacency matrix into train, validation and test edges and randomly samples equal amount of validation and test non-edges. 
"""
train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(_A_obs, val_share, test_share, seed, undirected=True, connected=True, asserts=False) 

## EGGen
train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
assert (train_graph.toarray() == train_graph.toarray().T).all()


#### Parameters
""" Adjustable parameters for training. """ 
# setting GPU id 
gpu_id = 1 # 0 # 
# setting the number of nodes
_N = _A_obs.shape[0]
# setting the length of random walks
rw_len = max_len #20 #
# setting the training data batch size
batch_size = 128 #
# getting the number of departments
n_conds=len(np.unique(lcc_condlist[:,1]))
print("n_conds: ", n_conds)
sample_batch = 128 #

walker = utils.RandomWalker(train_graph, lcc_condlist, rw_len, p=1, q=1, batch_size=batch_size, sample_batch=sample_batch)


# ## Create our generative model 
l2_gen=1e-7; l2_disc=5e-5 #1e-4 
# gencond_lay=[10]; gen_lay=[50]; disc_lay=[40] # try gen_lay=[40]; disc_lay=[30]
gencond_lay=[10]; gen_lay=[40]; disc_lay=[30] # try gen_lay=[40]; disc_lay=[30]
lr_gencond=0.01; lr_gen=0.0002; lr_disc=0.0002
gencond_iters=1; gen_iters=3; disc_iters=1
discWdown_size=128; genWdown_size=128 

eggen = EGGen(_N,
rw_len,
walk_generator=walker,
n_conds=n_conds,
condgenerator_layers=gencond_lay,
generator_layers=gen_lay,
discriminator_layers=disc_lay,
W_down_discriminator_size=discWdown_size,
W_down_generator_size=genWdown_size,
batch_size=batch_size,
sample_batch=sample_batch,
condition_dim=n_conds,
gencond_iters=gencond_iters,
gen_iters=gen_iters,
disc_iters=disc_iters,
wasserstein_penalty=10, 
l2_penalty_generator=l2_gen,
l2_penalty_discriminator=l2_disc,
lr_gencond=lr_gencond,
lr_gen=lr_gen,
lr_disc=lr_disc,
noise_dim=16, #
noise_type="Gaussian", #
temp_start=5.0, #
min_temperature=0.5,
temperature_decay=1-5e-5,
seed=15, #seed, #
use_gumbel=True,
legacy_generator=False,
gpu_id=gpu_id,
plot_show=False
)


# #### Define the stopping criterion
stopping_criterion = "val"
assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."
if stopping_criterion == "val": # use val criterion for early stopping
    stopping = None
elif stopping_criterion == "eo":  #use eo criterion for early stopping
    stopping = 0.5 # set the target edge overlap here


# #### Train the model
eval_every = plot_every = 500 #200 #2000
max_iters = 5000 #10000 #20000 
patience= 20 


# ## Training the model 
# train and save model to ./snapshots/
log_dict = eggen.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
                        eval_every=eval_every, plot_every=plot_every, max_patience=patience, max_iters=max_iters)



# #### Save the training log
## when changing the directory, remember to change directory in eggen.train() too
# save_directory = "./testing"
save_directory = "./snapshots_mimic" #"./snapshots_gencond"  #"./snapshots_gencond2" 
model_name = "mimicgen"


save_log = "{}/log{}_{}_maxiter{}_evalevery{}.pkl".format(save_directory, log_num, model_name, max_iters, eval_every)
f = open(save_log,"wb")
pickle.dump(log_dict,f)
f.close()


