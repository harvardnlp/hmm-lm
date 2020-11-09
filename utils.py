
import sys
import random

import yaml

import numpy as np
import torch as th

import matplotlib.pyplot as plt


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


def set_seed(seed):
    """Sets random seed everywhere."""
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_config(path, device):
    with open(path, "r") as f:
        pack = Pack(yaml.load(f, Loader = yaml.Loader))
        pack.device = device
        return pack

def get_name(config):
    return "_".join([
        config.dataset,
        config.iterator,
        config.model,
        f"k{config.num_classes}",
        f"wps{config.words_per_state}",
        f"spw{config.states_per_word}",
        f"tspw{config.train_spw}",
        #f"ff{config.ffnn}",
        f"ed{config.emb_dim}",
        f"d{config.hidden_dim}",
        f"cd{config.char_dim}",
        f"dp{config.dropout}",
        f"tdp{config.transition_dropout}",
        f"cdp{config.column_dropout}",
        f"sdp{config.start_dropout}",
        f"dt{config.dropout_type}",
        f"wd{config.word_dropout}",
        config.bsz_fn,
        f"b{config.bsz}",
        config.optimizer,
        f"lr{config.lr}",
        f"c{config.clip}",
        f"tw{config.tw}",
        f"nas{config.noise_anneal_steps}",
        f"pw{config.posterior_weight}",
        f"as{config.assignment}",
        f"nb{config.num_clusters}",
        f"nc{config.num_common}",
        f"ncs{config.num_common_states}",
        f"spc{config.states_per_common}",
        f"n{config.ngrams}",
        f"r{config.reset_eos}",
        f"ns{config.no_shuffle_train}",
        f"fc{config.flat_clusters}",
        f"e{config.emit}",
        f"ed{'-'.join(str(x) for x in config.emit_dims) if config.emit_dims is not None else 'none'}",
        f"nh{config.num_highway}",
        f"s{config.state}",
    ])

def get_mask_lengths(text, V):
    mask = text != V.stoi["<pad>"]
    lengths = mask.sum(-1)
    n_tokens = mask.sum()
    return mask, lengths, n_tokens

def log_eye(K, dtype, device):
    x = th.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

def plot_counts(counts):
    num_c, num_w = counts.shape
    words = [
        13, 29, 67, 111, 131, 171, 373, 567, 700, 800,
        5617,5053,5601,5756,1482,7443,3747,8314,11,3722,7637,7916,3376,7551,
        5391,9072,230,9244,6869,441,1076,7093,1845,201,1386,6738,2840,4909,
    ]
    counts = counts[:, words]
    fig, axs = plt.subplots(1, 3)
    axs[0].spy(counts, precision=0.0001, markersize=1, aspect="auto")
    axs[1].spy(counts, precision=0.001, markersize=1, aspect="auto")
    axs[2].spy(counts, precision=0.01, markersize=1, aspect="auto")
    return plt
