import pandas as pd
import torch
from scipy.spatial import distance

from ..config import N_SOCS, N_SES, WV_SIZE

se_embed_mat, soc_embed_mat = None, None

def set_top_k_cost(soc_embed_path=None):
    global se_embed_mat, soc_embed_mat

    with open('datasets/embeds/side_effect_embed.csv') as se_embed:
        se_embeds = pd.read_csv(se_embed)
    se_embed_mat = torch.empty((N_SES, WV_SIZE), dtype=torch.float32)
    for ind, row in se_embeds.iterrows():
        curr_embedding = row["embedding"]
        curr_embedding = curr_embedding.strip("[]").split()
        curr_embedding = torch.tensor([float(i) for i in curr_embedding], dtype=torch.float32)
        se_embed_mat[ind] = curr_embedding

    if not soc_embed_path: return

    with open(soc_embed_path) as soc_embed:
        soc_embeds = pd.read_csv(soc_embed)
    soc_embed_mat = torch.empty((N_SOCS, WV_SIZE), dtype=torch.float32)
    for ind, row in soc_embeds.iterrows():
        curr_embedding = row["embedding"]
        curr_embedding = curr_embedding.strip("[]").split()
        curr_embedding = torch.tensor([float(i) for i in curr_embedding], dtype=torch.float32)
        soc_embed_mat[ind] = curr_embedding

def top_k_cost(pred, real, k, is_soc):
    """
    Calculates the top-k cost as outlined in the paper 'Learning with a Wasserstein Loss'.
    Ground metric used is L2 distance.

    Parameters:
    pred (List<float>): List of model's output predictions.
    real (List<float>): Corresponding ground truth labels.
    k (int): Hyperparameter. Top K predictions to compare.
    is_soc (bool): True if pred and real are SOC label predictions (of dimension 26), False otherwise
    """
    if len(pred[0]) != len(real[0]): 
        raise Exception("Error in top k cost. Dimensions of pred and real are not the same.")
    
    if is_soc:
        if soc_embed_mat is None: raise Exception("Error in top k cost. SOC embed mat was not set.")
        embed_mat = soc_embed_mat
    else:
        if se_embed_mat is None: raise Exception("Error in top k cost. SE embed mat was not set.")
        embed_mat = se_embed_mat
    
    loss, batch_size = 0, len(pred)
    for index, p in enumerate(pred):
        top_k = torch.topk(torch.tensor(p), k)
        curr_loss = 0
        for i in top_k.indices:
            pred_embedding = embed_mat[int(i)]
            min_dist = float('inf')
            for r_index, r in enumerate(real[index]):
                if float(r) <= 0.0: continue
                real_embedding = embed_mat[r_index]
                dist = distance.euclidean(pred_embedding, real_embedding)
                min_dist = min(min_dist, dist)
            curr_loss += min_dist
        curr_loss /= k
        loss += curr_loss
    
    return loss / batch_size
