
import time as timep

import sys

from collections import defaultdict

import yaml

import numpy as np
import torch as th


def convert_w2s(word2state, num_states):
    # invert word to state mapping
    # assumes the last word is a padding word
    state2word = [[] for _ in range(num_states)]
    num_wordsp1, states_per_word = word2state.shape
    num_words = num_wordsp1 - 1
    for word in range(num_words):
        for idx in range(states_per_word):
            state = word2state[word, idx]
            state2word[state].append(word)
    words_per_state = max(len(x) for x in state2word)
    for s in range(num_states):
        l = len(state2word[s])
        if l < words_per_state:
            state2word[s].extend([num_words] * (words_per_state - l))
    return th.tensor(state2word, dtype=word2state.dtype, device=word2state.device)

def perturb_kmax(potentials, noise_dist, k):
    num_states, num_words = potentials.shape
    perturbed_scores = potentials + noise_dist.sample(potentials.shape).squeeze(-1)
    # always topk on inner dim
    scores, idx = perturbed_scores.t().topk(k, dim=1)
    # return word2state
    return idx

def read_lm_clusters(V, path="clusters/lm-128/paths"):
    with open(path, "r") as f:
        word2cluster = {}
        word_counts = []
        cluster2word = defaultdict(list)
        cluster2id = {}
        id = 0
        for line in f:
            cluster, word, count = line.strip().split()
            if cluster not in cluster2id:
                cluster2id[cluster] = id
                id += 1
            cluster_id = cluster2id[cluster]
            word2cluster[V[word]] = cluster_id
            cluster2word[cluster_id].append(V[word])
            word_counts.append((V[word], int(count)))
        print(f"Read {id} clusters from {path}")
        return (
            word2cluster,
            sorted(word_counts, key=lambda x: x[1], reverse=True),
            dict(cluster2word),
        )

def assign_states_brown_cluster(
    num_states, word2cluster, V,
    states_per_word,
    states_per_word_d,
):
    # must have num_states = num_clusters * num_repeats 
    num_words = len(V)
    # assume this is less than num_states // states_per_word
    num_clusters = len(set(word2cluster.values()))
    #states_per_word = num_states // num_clusters
    w2c = np.ndarray(len(V), dtype=np.int64)
    for word in range(len(V)):
        w2c[word] = (word2cluster[word]
            if word in word2cluster
            else num_clusters-1
        )
    cluster2state = np.ndarray((num_clusters, states_per_word), dtype=np.int64)
    for c in range(0, num_clusters):
        cluster2state[c] = range(
            states_per_word * c,
            states_per_word * (c + 1),
        )
    word2state = cluster2state[w2c]
    # the dropped cluster to words after reindexing
    # assume states per word // 2
    c2sw_d = th.LongTensor([
        list(range(c * states_per_word_d, (c+1) * states_per_word_d))
        for c in range(num_clusters)
    ])
    return word2state, cluster2state, w2c, c2sw_d


