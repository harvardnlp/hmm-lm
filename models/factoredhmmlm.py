
import time as timep
import os

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "hmm_runners/hmm.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import numpy as np

import torch as th
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torch_struct as ts

from .misc import ResidualLayer, LogDropout
from .stateemb import StateEmbedding

from utils import Pack
from assign import read_lm_clusters, assign_states_brown_cluster


class FactoredHmmLm(nn.Module):
    def __init__(self, V, config):
        super(FactoredHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.num_clusters = config.num_clusters

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word
        self.train_states_per_word = config.train_spw
        self.states_per_word_d = config.train_spw

        self.num_layers = config.num_layers

        self.timing = config.timing > 0
        self.chp_theta = config.chp_theta > 0

        self.reset_eos = "reset_eos" in config and config.reset_eos > 0
        self.flat_clusters = "flat_clusters" in config and config.flat_clusters > 0

        num_clusters = config.num_clusters if "num_clusters" in config else 128

        if "dataset" not in config:
            path = f"clusters/lm-{num_clusters}/paths"
        elif config.dataset == "ptb":
            lmstring = "lm" if not self.flat_clusters else "flm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wikitext2":
            #lmstring = "w2lm"
            lmstring = "w2flm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wikitext103":
            lmstring = "wlm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wsj":
            lmstring = "wsj"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        else:
            raise ValueError

        word2cluster, word_counts, cluster2word = read_lm_clusters(
            V,
            path=path,
        )
        self.word_counts = word_counts

        assert self.states_per_word * num_clusters <= self.C

        word2state = None
        if config.assignment == "brown":
            (
                word2state,
                cluster2state,
                word2cluster,
                c2sw_d,
            ) = assign_states_brown_cluster(
                self.C,
                word2cluster,
                V,
                self.states_per_word,
                self.states_per_word_d,
            )
        else:
            raise ValueError(f"No such assignment {config.assignment}")

        # need to save this with model
        self.register_buffer("word2state", th.from_numpy(word2state))
        self.register_buffer("cluster2state", th.from_numpy(cluster2state))
        self.register_buffer("word2cluster", th.from_numpy(word2cluster))
        self.register_buffer("c2sw_d", c2sw_d)
        self.register_buffer("word2state_d", self.c2sw_d[self.word2cluster])

        self.tvm_fb = "tvm_fb" in config and config.tvm_fb
        self.fb_train = foo.get_fb(self.train_states_per_word)
        self.fb_test = foo.get_fb(self.states_per_word)

        # p(z0)
        self.start_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )
        self.start_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # p(zt | zt-1)
        self.state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )
        self.trans_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
        )
        self.next_state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )

        # p(xt | zt)
        self.preterminal_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
        )
        self.terminal_proj = nn.Linear(config.hidden_dim, len(V))

        self.dropout = nn.Dropout(config.dropout)

        # tie embeddings key. use I separated pairs to specify
        # s: start
        # l: left
        # r: right
        # p: preterminal
        # o: output, can't be tied
        if "sl" in config.tw:
            self.state_emb.share(self.start_emb)
        if "lr" in config.tw:
            self.next_state_emb.share(self.state_emb)
        if "rp" in config.tw:
            self.preterminal_emb.share(self.next_state_emb)

        self.transition_dropout = LogDropout(config.transition_dropout)
        self.column_dropout = config.column_dropout > 0

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)


        self.ad = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word_d)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.vd = th.ones((len(self.V)) * self.states_per_word_d).to(self.device)

        self.keep_counts = config.keep_counts > 0
        if self.keep_counts:
            self.register_buffer(
                "counts",
                th.zeros(self.states_per_word, len(self.V)),
            )
            self.register_buffer(
                "state_counts",
                th.zeros(self.C, dtype=th.int),
            )

        self.register_buffer("zero", th.zeros(1))
        self.register_buffer("one", th.ones(1))


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    def start(self, states=None):
        start_emb = self.start_emb(states)
        return self.start_mlp(self.dropout(start_emb)).squeeze(-1).log_softmax(-1)

    def start_chp(self, states=None):
        start_emb = (self.start_emb[states]
            if states is not None
            else self.start_emb
        )
        return checkpoint(
            lambda x: self.start_mlp(self.dropout(x)).squeeze(-1).log_softmax(-1),
            start_emb
        )

    def transition_logits(self, states=None):
        state_emb = self.state_emb(states)
        next_state_emb = self.next_state_emb(states)
        x = self.trans_mlp(self.dropout(state_emb))
        return x @ next_state_emb.t()

    def mask_transition(self, logits):
        return logits.log_softmax(-1)

    def transition_chp(self, states=None):
        state_emb = self.state_emb(states)
        next_state_proj = self.next_state_emb(states)
        return checkpoint(
            lambda x, y: (self.trans_mlp(self.dropout(x)) @ y.t()).log_softmax(-1),
            state_emb, next_state_proj,
        )

    def emission_logits(self, states=None):
        preterminal_emb = self.preterminal_emb(states)
        h = self.terminal_mlp(self.dropout(preterminal_emb))
        logits = self.terminal_proj(h)
        return logits

    def mask_emission(self, logits, word2state):
        a = self.ad if self.training else self.a
        v = self.vd if self.training else self.v

        i = th.stack([word2state.view(-1), a])
        C = logits.shape[0]
        sparse = th.sparse.ByteTensor(i, v, th.Size([C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        log_probs = logits.masked_fill_(~mask, float("-inf")).log_softmax(-1)
        return log_probs

    def emission_chp(self, word2state, states=None):
        preterminal_emb = (self.preterminal_emb.weight[states]
            if states is not None
            else self.preterminal_emb.weight
        )
        return checkpoint(
            lambda x: self.mask_emission(
                self.terminal_mlp(self.dropout(x)),
                word2state,
            ),
            preterminal_emb
        )


    def clamp(
        self, text, start, transition, emission, word2state,
        reset = None,
    ):
        clamped_states = word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        if reset is not None:
            eos_mask = text[:,:-1] == self.V["<eos>"]
            # reset words following eos
            reset_states = word2state[text[:,1:][eos_mask]]
            log_potentials[eos_mask] = reset[reset_states][:,None]
        
        b_idx = th.arange(batch, device=self.device)
        init = (
            start[clamped_states[:,0]]
            if start.ndim == 1
            else start[b_idx[:,None], clamped_states[:,0]]
        )

        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]

        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        return log_potentials.transpose(-1, -2)

    def trans_to(self, from_states, to_states):
        state_emb = self.state_emb(from_states)
        next_state_proj = self.next_state_emb(to_states)
        x = self.trans_mlp(self.dropout(state_emb))
        return (x @ next_state_proj.t()).log_softmax(-1)

    def compute_parameters(self, word2state,
        states=None, word_mask=None,
        lpz=None, last_states=None,
    ):
        if self.chp_theta:
            transition = self.transition_chp(states)
        else:
            transition = self.mask_transition(self.transition_logits(states))

        if last_states is None:
            start = self.start(states)
        else:
            # compute start from last_state
            start = (
                lpz[:,:,None] + self.trans_to(last_states, states)
            ).logsumexp(1)
             
        emission = self.mask_emission(self.emission_logits(states), word2state)
        return start, transition, emission

    def log_potentials(
        self, text,
        states = None,
        lpz=None, last_states=None,
        word_mask=None,
    ):
        word2state = self.word2state_d if states is not None else self.word2state

        start, transition, emission = self.compute_parameters(
            word2state, states,
            word_mask,
            lpz, last_states,
        )
        # reset hidden state at EOS
        reset = self.start(states) if self.reset_eos else None

        return self.clamp(
            text, start, transition, emission, word2state,
            reset = reset,
        )

    def compute_loss(
        self,
        log_potentials, mask, lengths,
        keep_counts = False,
    ):
        N = lengths.shape[0]
        fb = self.fb_train if self.training else self.fb_test
        log_m, alphas = fb(log_potentials.clone(), mask=mask)

        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1)

    def score(
        self, text,
        lpz=None, last_states=None,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        if self.training:
            I = (th.distributions.Gumbel(self.zero, self.one)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.train_states_per_word, dim=-1)
                .indices
            )
            states = self.cluster2state.gather(1, I).view(-1)
        else:
            states = None
        if self.timing:
            startpot = timep.time()
        log_potentials = self.log_potentials(
            text,
            states,
            lpz, last_states,
        )
        if self.timing:
            print(f"log pot: {timep.time() - startpot}")
        fb = self.fb_train if self.training else self.fb_test
        with th.no_grad():
            log_m, alphas = fb(log_potentials.detach().clone(), mask=mask)
        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        last_words = text[idx, lengths-1]
        c2s = states.view(self.config.num_clusters, -1)
        end_states = c2s[self.word2cluster[last_words]]

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), end_states

