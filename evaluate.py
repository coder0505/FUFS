import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.utils.linear_assignment_ as la
import warnings
warnings.filterwarnings('ignore')
max_repeat = 50


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        if A[i][1] < n_class2:
            new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)


def evaluate(data, y_pred, y_true, protect_attribute, n_cluster, protect_values):
    # calculate accuracy
    y_permuted_predict = best_map(y_true,y_pred)
    acc = accuracy_score(y_true, y_permuted_predict)
    # calculate nmi
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # only for crime dataset
    #x = [data[:, protect_attribute] > 0.06]
    #x = np.squeeze(np.array(x))
    #x = [int(i) for i in x]
    #for k in range(data.shape[0]):
    #    data[k, protect_attribute] = x[k]

    #calculate balance
    bal_temp = []
    prop_temp = []
    for i in range(n_cluster):
        index = np.argwhere(y_pred == i)
        size = index.shape[0]
        index = np.squeeze(index)
        C_i = data[index, :]   #cluster i

        temp = []
        for p_value in protect_values:
            if size != 1:
                num = np.sum((C_i[:,protect_attribute])==p_value)
            else:
                num = np.sum((C_i[protect_attribute])==p_value)
            temp.append(num)
        inter = min(temp)
        inter2 = max(temp)
        bal_temp.append(inter/size)
        prop_temp.append(inter2/size)
    balance = min(bal_temp)
    proportion = sum(prop_temp)

    return acc,nmi,balance,proportion


def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
    return cosine_similarity(X, Y)


def func(idx,k,data,protect_attribute,true_label,n_cluster,protect_value):
    clf = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, precompute_distances=True, verbose=0,
                random_state=None, copy_x=True, n_jobs=1)

    #for sparse data Google+ and Toxicity
    #k_means_.euclidean_distances = euc_dist
    #clf = k_means_.KMeans(n_clusters=n_cluster, random_state=None)

    idx = np.squeeze(idx)
    idxes = idx[0:k]
    data_selected = data[:,idxes]

    acc_sel =  []
    nmi_sel = []
    balance_sel = []
    prop_sel = []

    #repeat 50 times and get the average values
    for repeat in range(max_repeat):
        clf.fit(data_selected)
        y_pred = clf.labels_
        acc,nmi,bal,prop = evaluate(data, y_pred, true_label, protect_attribute,n_cluster,protect_value)
        acc_sel.append(acc)
        nmi_sel.append(nmi)
        balance_sel.append(bal)
        prop_sel.append(prop)


    aver_acc = np.mean(acc_sel)
    aver_nmi = np.mean(nmi_sel)
    aver_bal = np.mean(balance_sel)
    aver_prop = np.mean(prop_sel)

    #print('-------feature number:',k,'---------')
    #print('acc:',aver_acc)
    #print('nmi:', aver_nmi)
    #print('balance:', aver_bal)
    #print('proportion:',aver_prop)
    return aver_acc, aver_nmi, aver_bal, aver_prop