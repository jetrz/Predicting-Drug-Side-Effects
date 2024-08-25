from copy import deepcopy 
from ..config import N_SES

def split_data(data_list, data_labels, side_effect_count, fold=0, data_socs=None):
    """Splits the data into train and valid sets, ensuring that all labels appear in both.

    Parameters:
    data_list (List<List<int>>): Data list returned from get_data.
    data_labels (List<List<int>>): Data labels returned from get_data.
    data_socs (List<List<int>>): Optional. Data SOCs returned from get_data. If None, the returned train_socs and test_socs will also be empty.
    side_effect_count (List<int>): Each label's appearance count in the dataset.
    fold (int): Current fold.
    """
    train_dataset, train_labels, train_socs, test_dataset, test_labels, test_socs = [], [], [], [], [], []
    freq_criteria = deepcopy(side_effect_count)
    test_train_counts = {i:[0,0] for i in range(N_SES)}
    has_conflict = False
    
    for i in range(len(data_list)):
        c_data, c_labels = data_list[i], data_labels[i]
        c_socs = data_socs[i] if data_socs else None
        needed_in_test, needed_in_train, needed_in_test_veto, needed_in_train_veto = False, False, False, False
        true_indices = [i for i, x in enumerate(c_labels) if x == 1]
        for ind in true_indices:
            freq_criteria[ind] -= 1
            counts = test_train_counts[ind]
            if not counts[0]:
                if freq_criteria[ind] == 0: needed_in_test_veto = True
                needed_in_test = True
            if not counts[1]:
                if freq_criteria[ind] == 0: needed_in_train_veto = True
                needed_in_train = True
                
        if needed_in_test_veto and needed_in_train_veto: has_conflict=True
        
        put_in_test, put_in_train = False, False
        if needed_in_test_veto:
            test_dataset.append(c_data)
            test_labels.append(c_labels)
            test_socs.append(c_socs)
            put_in_test = True
        elif needed_in_train_veto:
            train_dataset.append(c_data)
            train_labels.append(c_labels)
            train_socs.append(c_socs)
            put_in_train = True
            
        if not put_in_test and not put_in_train:
            if needed_in_test and not needed_in_train:
                test_dataset.append(c_data)
                test_labels.append(c_labels)
                test_socs.append(c_socs)
                put_in_test = True
            elif needed_in_train and not needed_in_test:
                train_dataset.append(c_data)
                train_labels.append(c_labels)
                train_socs.append(c_socs)
                put_in_train = True
            else:
                if (i+fold) % 5 == 0:
                    test_dataset.append(c_data)
                    test_labels.append(c_labels)
                    test_socs.append(c_socs)
                    put_in_test = True
                else:
                    train_dataset.append(c_data)
                    train_labels.append(c_labels)
                    train_socs.append(c_socs)
                    put_in_train = True
                
        for ind in true_indices:
            if put_in_test: test_train_counts[ind][0] += 1
            if put_in_train: test_train_counts[ind][1] += 1
            
    if has_conflict: print("Conflict in splitting dataset: Data needed in both test and train")
    return train_dataset, train_labels, train_socs, test_dataset, test_labels, test_socs
