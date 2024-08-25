from copy import deepcopy
from geomloss import SamplesLoss
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random, time, torch, warnings, wandb, yaml

from ..config import N_SES, N_SOCS, FP_SIZE, WV_SIZE
from ..models.mlp import MLP
from ..models.custom_dataset import CustomDataset
from ..utils.get_data import get_data
from ..utils.split_data import split_data
from ..utils.top_k_cost import set_top_k_cost, top_k_cost

def run_model(fold, hyperparams, n_epochs=100, display_epoch_info=True):  
    print("Running model...")
    warnings.filterwarnings("ignore")

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    label_mode = hyperparams.get('label_mode', "")
    if label_mode == 'weighted':
        activation = 'softmax'
    elif label_mode == 'full':
        activation = 'relu'
    else:
        activation = 'sigmoid'

    model = MLP(layers=[(FP_SIZE, 1500), (1500, 1000), (1000, N_SOCS)], activation=activation)
    model2 = MLP(layers=[(FP_SIZE+N_SOCS, 1500), (1500, 1000), (1000, N_SES)], activation="sigmoid")
    model = model.to(device); model2 = model2.to(device)

    if label_mode == 'weighted':
        loss_fn_1 = torch.nn.CrossEntropyLoss()
    elif label_mode == 'full':
        loss_fn_1 = torch.nn.MSELoss()
    else:
        loss_fn_1 = torch.nn.BCELoss()
    loss_fn_2 = torch.nn.BCELoss()

    dist_formula = "SqDist(Normalize(X), Normalize(Y))"
    wass_loss_1, wass_loss_2 = SamplesLoss(loss='sinkhorn',blur=.05,scaling=0.3,cost=dist_formula,backend="online"), SamplesLoss(loss='sinkhorn',blur=.05,scaling=0.3,cost=dist_formula,backend="online")
    
    optimizer_1 = torch.optim.NAdam(model.parameters(), lr=hyperparams['lr_1'])
    optimizer_2 = torch.optim.NAdam(model2.parameters(), lr=hyperparams['lr_2'])
    scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.1, patience=20, verbose=True)
    scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.1, patience=20, verbose=True)

    randomise_mode = hyperparams.get('randomise_mode', '')
    label_ext, randomise_ext = '', ''
    if label_mode:
        label_ext = '_' + label_mode
    if randomise_mode:
        randomise_ext = '_' + randomise_mode

    data_list, data_labels, data_socs, se_embed_mat, soc_embed_mat, side_effect_count = get_data(soc_label_path=f"datasets/labels/soc_labels{label_ext}{randomise_ext}.csv", soc_embed_path=f"datasets/embeds/soc_embed_micro{randomise_ext}.csv")
    soc_embed_mat = soc_embed_mat.to(device); se_embed_mat = se_embed_mat.to(device)
    
    losses = { loss : [] for loss in hyperparams['losses_list'] }
    w_alpha_1, w_alpha_2 = hyperparams['w_alpha_1'], hyperparams['w_alpha_2']
    def train(loader):
        model.train()
        model2.train()
        for data in loader:  # Iterate in batches over the training dataset.
            x, y, z = data[0], data[1], data[2]
            out1 = model(x)  # Perform a single forward pass.
            loss1_1 = loss_fn_1(out1, z)
            pred_embeds = out1 @ soc_embed_mat
            label_embeds = z @ soc_embed_mat
            loss1_2 = wass_loss_1(pred_embeds, label_embeds)
            loss1 = w_alpha_1*loss1_1 + (1-w_alpha_1)*loss1_2
            loss1.backward()  # Derive gradients.
            optimizer_1.step()  # Update parameters based on gradients.
            optimizer_1.zero_grad()  # Clear gradients.
            losses['loss1'].append(loss1.item())
            losses['loss1_1'].append(loss1_1.item()*w_alpha_1)
            losses['loss1_2'].append(loss1_2.item()*(1-w_alpha_1))

            out1_copy = deepcopy(out1.detach())
            x_copy = deepcopy(x.detach())
            cat_input = torch.cat((x_copy, out1_copy), dim=1)

            out2 = model2(cat_input)
            loss2_1 = loss_fn_2(out2, y)
            pred_embeds = out2 @ se_embed_mat
            label_embeds = y @ se_embed_mat
            loss2_2 = wass_loss_2(pred_embeds, label_embeds)
            loss2 = w_alpha_2*loss2_1 + (1-w_alpha_2)*loss2_2
            loss2.backward()
            optimizer_2.step()
            optimizer_2.zero_grad()
            losses['loss2'].append(loss2.item())
            losses['loss2_1'].append(loss2_1.item()*w_alpha_2)
            losses['loss2_2'].append(loss2_2.item()*(1-w_alpha_2))

    def test(loader):
        model.eval()
        model2.eval()

        curr_losses = {'loss1' : [], 'loss2' : []}
        outs, out1s, labels, socs = [], [], [], []
        for data in loader:  # Iterate in batches over the training/test dataset.
            x, y, z = data[0], data[1], data[2]
            out1 = model(x)

            with torch.no_grad():
                loss1_1 = loss_fn_1(out1, z)
                pred_embeds = out1 @ soc_embed_mat
                label_embeds = z @ soc_embed_mat
                loss1_2 = wass_loss_1(pred_embeds, label_embeds)
                loss1 = w_alpha_1*loss1_1 + (1-w_alpha_1)*loss1_2
                curr_losses['loss1'].append(loss1.item())

            out1_copy = deepcopy(out1.detach())
            x_copy = deepcopy(x.detach())
            cat_input = torch.cat((x_copy, out1_copy), dim=1)

            out2 = model2(cat_input)

            with torch.no_grad():
                loss2_1 = loss_fn_2(out2, y)
                pred_embeds = out2 @ se_embed_mat
                label_embeds = y @ se_embed_mat
                loss2_2 = wass_loss_2(pred_embeds, label_embeds)
                loss2 = w_alpha_2*loss2_1 + (1-w_alpha_2)*loss2_2
                curr_losses['loss2'].append(loss2.item())

            output_copy = deepcopy(out2.detach())
            for i, o in enumerate(output_copy):
                outs.append(o.tolist())
                out1s.append(out1[i].tolist())
                labels.append(y[i].tolist())
                socs.append(z[i].tolist())

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
        metrics['top_k_cost_1'], metrics['top_k_cost_2'] = top_k_cost(out1s, socs, hyperparams['top_k_cost'], is_soc=True), top_k_cost(outs, labels, hyperparams['top_k_cost'], is_soc=False)

        scheduler_1.step(sum(curr_losses['loss1'])/len(curr_losses['loss1']))
        scheduler_2.step(sum(curr_losses['loss2'])/len(curr_losses['loss2']))

        return metrics
    
    c = list(zip(data_list, data_labels, data_socs))
    random.shuffle(c)
    data_list, data_labels, data_socs = zip(*c)
    train_data, train_labels, train_socs, test_data, test_labels, test_socs = split_data(data_list, data_labels, side_effect_count, data_socs=data_socs, fold=fold)
    test_dataset = CustomDataset(torch.Tensor(test_data).to(device), torch.Tensor(test_labels).to(device), socs=torch.Tensor(test_socs).to(device))
    valid_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    metric_list_1 = hyperparams['metric_list_1']
    metric_list_2 = hyperparams['metric_list_2']
    metric_list_3 = hyperparams['metric_list_3']
    metrics = { metric : [0 for _ in range(n_epochs)] for metric in metric_list_1 }
    for metric in metric_list_2:
        metrics[metric] = [0 for _ in range(N_SES)]
    for metric in metric_list_3:
        metrics[metric] = [0 for _ in range(n_epochs)]

    for epoch in range(n_epochs):
        curr_train_data, curr_train_labels, curr_train_socs = deepcopy(train_data), deepcopy(train_labels), deepcopy(train_socs)
        c = list(zip(curr_train_data, curr_train_labels, curr_train_socs))
        random.shuffle(c)
        curr_train_data, curr_train_labels, curr_train_socs = zip(*c)

        train_dataset = CustomDataset(torch.Tensor(curr_train_data).to(device), torch.Tensor(curr_train_labels).to(device), socs=torch.Tensor(curr_train_socs).to(device))
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

def run_mlp_concat_w_soc_labels():
    start_time = time.time()

    with open('code/hyperparameters.yaml', 'r') as f:
        hyperparams_f = yaml.safe_load(f)
        hyperparams = hyperparams_f['all']
        for k, v in hyperparams_f['concat'].items():
            hyperparams[k]= v

    kfold, n_epochs = hyperparams.get('kfold', 1), hyperparams.get('n_epochs', 150)

    randomise_mode = hyperparams.get('randomise_mode', None)
    if randomise_mode:
        soc_embed_path = f'datasets/embeds/soc_embed_micro_{randomise_mode}.csv'
    else:
        soc_embed_path = 'datasets/embeds/soc_embed_micro.csv'
    set_top_k_cost(soc_embed_path=soc_embed_path)

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
          Max Acc: {metrics['max_acc']} | Max ROC Macro: {metrics['max_roc_macro']} | Max ROC Micro: {metrics['max_roc_micro']} | Max ROC Weighted: {metrics['max_roc_weighted']} | Max PRC Macro: {metrics['max_prc_macro']} | Max PRC Micro: {metrics['max_prc_micro']} | Max PRC Weighted: {metrics['max_prc_weighted']} | Min Top K Cost 1: {metrics['min_top_k_cost_1']} | Min Top K Cost 2: {metrics['min_top_k_cost_2']} 
          ''')
    exec_time = time.time() - start_time
    print(f"Execution time: {exec_time:.2f} seconds")

    return metrics