from copy import deepcopy
from geomloss import SamplesLoss
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
import random, time, warnings, wandb, yaml

from ..config import N_SES, FP_SIZE
from ..models.mlp import MLP
from ..models.custom_dataset import CustomDataset
from ..utils.get_data import get_data
from ..utils.split_data import split_data
from ..utils.top_k_cost import set_top_k_cost, top_k_cost

def run_model(fold, hyperparams, n_epochs=400, display_epoch_info=True):  
    print("Running model...")
    model = MLP(layers=[(FP_SIZE, 1500), (1500, 1000), (1000, N_SES)], activation="sigmoid")
    
    warnings.filterwarnings("ignore")

    dist_formula = "SqDist(Normalize(X), Normalize(Y))"
    loss_fn_2 = torch.nn.BCELoss()
    wass_loss_2 = SamplesLoss(loss='sinkhorn',p=2,blur=.05,cost=dist_formula,backend="online")
    optimizer_2 = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr_1'])
    scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.1, patience=20, verbose=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_list, data_labels, se_embed_mat, side_effect_count = get_data()
    se_embed_mat = se_embed_mat.to(device)
    
    losses = { loss : [] for loss in hyperparams['losses_list'] }
    w_alpha_1 = hyperparams['w_alpha_1']
    def train(loader):
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
            x, y = data[0], data[1]
            out1 = model(x)  # Perform a single forward pass.
            loss2_1 = loss_fn_2(out1, y)
            input = out1 @ se_embed_mat
            label_embeds = y @ se_embed_mat
            loss2_2 = wass_loss_2(input, label_embeds)
            loss2 = w_alpha_1*loss2_1 + (1-w_alpha_1)*loss2_2
            loss2.backward()  # Derive gradients.
            optimizer_2.step()  # Update parameters based on gradients.
            optimizer_2.zero_grad()  # Clear gradients.
            losses['loss2'].append(loss2.item())
            losses['loss2_1'].append(loss2_1.item()*w_alpha_1)
            losses['loss2_2'].append(loss2_2.item()*(1-w_alpha_1))

    def test(loader):
        model.eval()

        curr_losses = {'loss2' : []}
        outs, labels = [], []
        for data in loader:  # Iterate in batches over the training/test dataset.
            x, y = data[0], data[1]
            out1 = model(x)

            with torch.no_grad():
                loss2_1 = loss_fn_2(out1, y)
                input = out1 @ se_embed_mat
                label_embeds = y @ se_embed_mat
                loss2_2 = wass_loss_2(input, label_embeds)
                loss2 = w_alpha_1*loss2_1 + (1-w_alpha_1)*loss2_2
                curr_losses['loss2'].append(loss2.item())

            output_copy = deepcopy(out1.detach())
            for i, o in enumerate(output_copy):
                outs.append(o.tolist())
                labels.append(y[i].tolist())

        labels_converted, outs_converted = deepcopy(labels), deepcopy(outs)
        for i in outs_converted:
            for j, val in enumerate(i):
                i[j] = int(val>=hyperparams['threshold'])
        labels_converted = sum(labels_converted, [])
        outs_converted = sum(outs_converted, [])

        metrics = {}
        tn, fp, fn, tp = confusion_matrix(labels_converted, outs_converted).ravel() 
        metrics['acc'], metrics['precision'], metrics['specificity'], metrics['sensitivity'] = (tp+tn)/(tp+tn+fp+fn), tp/(tp+fp), tn/(tn+fp), tp/(tp+fn)
        metrics['roc_macro'], metrics['roc_micro'], metrics['roc_weighted'], metrics['roc_classes'] = roc_auc_score(labels, outs, average='macro'), roc_auc_score(labels, outs, average='micro'), roc_auc_score(labels, outs, average='weighted'), roc_auc_score(labels, outs, average=None)
        metrics['prc_macro'], metrics['prc_micro'], metrics['prc_weighted'], metrics['prc_classes'] = average_precision_score(labels, outs, average='macro'), average_precision_score(labels, outs, average='micro'), average_precision_score(labels, outs, average='weighted'), average_precision_score(labels, outs, average=None)
        metrics['top_k_cost_2'] = top_k_cost(outs, labels, hyperparams['top_k_cost'], is_soc=False)
        
        scheduler_2.step(sum(curr_losses['loss2'])/len(curr_losses['loss2']))
        return metrics
    
    c = list(zip(data_list, data_labels))
    random.shuffle(c)
    data_list, data_labels = zip(*c)
    train_data, train_labels, _, test_data, test_labels, _ = split_data(data_list, data_labels, side_effect_count, fold=fold)
    test_dataset = CustomDataset(torch.Tensor(test_data).to(device), torch.Tensor(test_labels).to(device))
    valid_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    metric_list_1, metric_list_2, metric_list_3 = hyperparams['metric_list_1'], hyperparams['metric_list_2'], hyperparams['metric_list_3']
    metrics = { metric : [0 for _ in range(n_epochs)] for metric in metric_list_1 }
    for metric in metric_list_2:
        metrics[metric] = [0 for _ in range(N_SES)]
    for metric in metric_list_3:
        metrics[metric] = [0 for _ in range(n_epochs)]
    for epoch in range(n_epochs):
        curr_train_data, curr_train_labels = deepcopy(train_data), deepcopy(train_labels)
        c = list(zip(curr_train_data, curr_train_labels))
        random.shuffle(c)
        curr_train_data, curr_train_labels = zip(*c)
        train_dataset = CustomDataset(torch.Tensor(curr_train_data).to(device), torch.Tensor(curr_train_labels).to(device))
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        train(train_loader)
        curr_metrics = test(valid_loader)
        wandb.log(curr_metrics)
        
        for m in metric_list_1:
            metrics[m][epoch] += curr_metrics[m]
        for m in metric_list_2:
            for i in range(N_SES):
                metrics[m][i] = max(metrics[m][i], curr_metrics[m][i])
        for m in metric_list_3:
            metrics[m][epoch] += curr_metrics[m]
        
        if display_epoch_info and epoch % 10 == 0:
            resString = f'''
            Fold: {fold} | Epoch: {epoch} \n
            '''
            for m in metric_list_1:
                curr_value = metrics[m][epoch]
                resString += f"| Curr {m}: {curr_value:.4f} |"
            for m in metric_list_3:
                curr_value = metrics[m][epoch]
                resString += f"| Curr {m}: {curr_value:.4f} |"
            resString += " \n "
            for m in metric_list_1:
                max_value = max(metrics[m])
                resString += f"| Max {m}: {max_value:.4f} |"
            for m in metric_list_3:
                min_value = min(filter(None, metrics[m]))
                resString += f"| Min {m}: {min_value:.6f} |"
            print(resString)

    return metrics, losses

def run_mlp_baseline():
    start_time = time.time()

    with open('code/hyperparameters.yaml', 'r') as f:
        hyperparams_f = yaml.safe_load(f)
        hyperparams = hyperparams_f['all']
        for k, v in hyperparams_f['baseline'].items():
            hyperparams[k]= v

    kfold, n_epochs = hyperparams.get('kfold', 1), hyperparams.get('n_epochs', 150)
    set_top_k_cost()

    metric_list_1 = hyperparams['metric_list_1']
    metric_list_2 = hyperparams['metric_list_2']
    metric_list_3 = hyperparams['metric_list_3']
    metrics = {}
    for metric in metric_list_1:
        metrics[metric] = [0 for _ in range(n_epochs)]
        metrics['max_'+metric] = 0
    for metric in metric_list_2:
        metrics[metric] = [0 for _ in range(N_SES)]
    for metric in metric_list_3:
        metrics[metric] = [0 for _ in range(n_epochs)]
        metrics['min_'+metric] = 0

    losses_list = hyperparams['losses_list']
    losses = { loss : [] for loss in losses_list }

    wandb.init(**hyperparams_f['wandb'])

    for fold in range(kfold):
        random.seed(777*kfold)

        curr_metrics, curr_losses = run_model(fold=fold, hyperparams=hyperparams, n_epochs=n_epochs, display_epoch_info=True)
        for m in metric_list_1:
            for i in range(n_epochs):
                metrics[m][i] += curr_metrics[m][i]/kfold
        for m in metric_list_1:
            curr_max = max(curr_metrics[m])
            metrics['max_'+m] += curr_max/kfold
        for m in metric_list_2:
            for i in range(N_SES):
                metrics[m][i] += curr_metrics[m][i]/kfold
        for m in metric_list_3:
            for i in range(n_epochs):
                metrics[m][i] += curr_metrics[m][i]/kfold
        for m in metric_list_3:
            curr_min = min(curr_metrics[m])
            metrics['min_'+m] += curr_min/kfold
        for l in losses_list:
            losses[l].extend(curr_losses[l])

        res_string = f'''
        FOLD {fold} COMPLETED \n
        '''
        for m in metric_list_1:
            curr_max = max(curr_metrics[m])
            res_string += f"| Max {m}: {curr_max:.4f} |"
        for m in metric_list_3:
            curr_min = min(curr_metrics[m])
            res_string += f"| Min {m}: {curr_min:.4f} |"
        print(res_string)
        
    print(f'''
          RUN COMPLETED
          Max Acc: {metrics['max_acc']} | Max ROC Macro: {metrics['max_roc_macro']} | Max ROC Micro: {metrics['max_roc_micro']} | Max ROC Weighted: {metrics['max_roc_weighted']} | Max PRC Macro: {metrics['max_prc_macro']} | Max PRC Micro: {metrics['max_prc_micro']} | Max PRC Weighted: {metrics['max_prc_weighted']} | Min Top K Cost 2: {metrics['min_top_k_cost_2']} 
          ''')
    exec_time = time.time() - start_time
    print(f"Execution time: {exec_time:.2f} seconds")

    return metrics