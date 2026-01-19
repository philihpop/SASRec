import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, train_ratio=0.6, val_ratio=0.2):
    """
    Per-user split: 60% train, 20% val, 20% test
    Works cleanly with 5-core (min 5 interactions)
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    f.close()
    
    for user in User:
        nfeedback = len(User[user])
        
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            train_end = int(nfeedback * train_ratio)
            val_end = int(nfeedback * (train_ratio + val_ratio))
            
            user_train[user] = User[user][:train_end]
            user_valid[user] = User[user][train_end:val_end]
            user_test[user] = User[user][val_end:]
    
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, K_list=[5, 10, 20, 50]):
    """Full ranking evaluation (rank against ALL items)"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    metrics = {k: {'NDCG': 0.0, 'Recall': 0.0} for k in K_list}
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0] if len(valid[u]) > 0 else 0
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        # Rank against ALL items
        rated = set(train[u])
        if len(valid[u]) > 0:
            rated.add(valid[u][0])
        rated.add(0)  # padding
        
        # Full item set
        all_items = list(range(1, itemnum + 1))
        target_item = test[u][0]
        
        # Predict scores for all items
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], all_items]])
        predictions = predictions[0]  # Shape: [itemnum]
        
        # Mask out training/validation items (set to -inf so they rank last)
        for rated_item in rated:
            if rated_item > 0 and rated_item <= itemnum:
                predictions[rated_item - 1] = float('inf')  # -(-inf) = inf, ranks last
        
        # Get rank of target item
        rank = predictions.argsort().argsort()[target_item - 1].item()

        valid_user += 1

        # Compute metrics for each K
        for k in K_list:
            if rank < k:
                metrics[k]['NDCG'] += 1 / np.log2(rank + 2)
                metrics[k]['Recall'] += 1
                
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    results = {}
    for k in K_list:
        results[k] = {
            'NDCG': metrics[k]['NDCG'] / valid_user,
            'Recall': metrics[k]['Recall'] / valid_user
        }
    
    return results

# evaluate on val set
def evaluate_valid(model, dataset, args, K_list=[5, 10, 20, 50]):
    """Full ranking evaluation on validation set"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    metrics = {k: {'NDCG': 0.0, 'Recall': 0.0} for k in K_list}
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # Rank against ALL items
        rated = set(train[u])
        rated.add(0)  # padding
        
        # Full item set
        all_items = list(range(1, itemnum + 1))
        target_item = valid[u][0]
        
        # Predict scores for all items
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], all_items]])
        predictions = predictions[0]  # Shape: [itemnum]
        
        # Mask out training items (set to -inf so they rank last)
        for rated_item in rated:
            if rated_item > 0 and rated_item <= itemnum:
                predictions[rated_item - 1] = float('inf')  # -(-inf) = inf, ranks last
        
        # Get rank of target item
        rank = predictions.argsort().argsort()[target_item - 1].item()

        valid_user += 1

        # Compute metrics for each K
        for k in K_list:
            if rank < k:
                metrics[k]['NDCG'] += 1 / np.log2(rank + 2)
                metrics[k]['Recall'] += 1
                
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Normalize by number of users
    results = {}
    for k in K_list:
        results[k] = {
            'NDCG': metrics[k]['NDCG'] / valid_user,
            'Recall': metrics[k]['Recall'] / valid_user
        }
    
    return results
