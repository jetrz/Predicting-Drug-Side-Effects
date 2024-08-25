import pandas as pd
import numpy as np
from rdkit import Chem
import torch

from ..config import N_SOCS, N_SES, WV_SIZE

def get_data(soc_label_path=None, soc_embed_path=None):
    """Parses and returns the dataset.

    Parameters:
    soc_label_path (str): Path to soc label csv. If this is None, soc data returned will be empty.
    soc_embed_path (str): Path to soc embed csv. If this is None, soc data returned will be empty.
    """
    with open('datasets/labels/side_effect_labels.txt') as se, open('datasets/embeds/side_effect_embed.csv') as se_embed, open('datasets/compounds/ID.txt') as ids:
        se_rows = se.read().split('\n')
        se_rows.pop(); se_rows.pop(0)
        se_embeds = pd.read_csv(se_embed)
        drug_ids = ids.read().split('\n')
        drug_ids.pop()

    se_embed_mat = torch.empty((N_SES, WV_SIZE), dtype=torch.float32)
    for ind, row in se_embeds.iterrows():
        curr_embedding = row["embedding"]
        curr_embedding = curr_embedding.strip("[]").split()
        curr_embedding = torch.tensor([float(i) for i in curr_embedding], dtype=torch.float32)
        se_embed_mat[ind] = curr_embedding

    parse_soc = soc_label_path and soc_embed_path
    if parse_soc:
        with open(soc_label_path) as c, open(soc_embed_path) as c_embed:
            c_rows = pd.read_csv(c)
            soc_embeds = pd.read_csv(c_embed)
        
        soc_embed_mat = torch.empty((N_SOCS, WV_SIZE), dtype=torch.float32)
        for ind, row in soc_embeds.iterrows():
            curr_embedding = row["embedding"]
            curr_embedding = curr_embedding.strip("[]").split()
            curr_embedding = torch.tensor([float(i) for i in curr_embedding], dtype=torch.float32)
            soc_embed_mat[ind] = curr_embedding            

    data_list, data_labels, data_socs, side_effect_count = [], [], [], { i:0 for i in range(N_SES) }
    for ind, row in enumerate(se_rows):
        ses = [int(i) for i in row.split('\t')]
        if parse_soc: socs = [c_rows.iloc[ind, i] for i in range(2,N_SOCS+2)]
        suppl = Chem.SDMolSupplier('datasets/compounds/{}.sdf'.format(drug_ids[ind]))
        for mol in suppl:
            fps = Chem.RDKFingerprint(mol)
            fps = [*fps.ToBitString()]
            fps = [int(i) for i in fps]
            data_list.append(np.array(fps)); data_labels.append(ses)
            if parse_soc: data_socs.append(socs)

        for ind, val in enumerate(ses):
            if val: side_effect_count[ind] += 1
    
    if parse_soc:
        return data_list, data_labels, data_socs, se_embed_mat, soc_embed_mat, side_effect_count
    else:
        return data_list, data_labels, se_embed_mat, side_effect_count