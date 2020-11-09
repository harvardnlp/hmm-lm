
import time as timep

import sys

import math
import time

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import torchtext
from datasets.lm import PennTreebank, WikiText2
from datasets.data import BucketIterator, BPTTIterator

from args import get_args

from utils import set_seed, get_config, get_name, get_mask_lengths
from utils import Pack
from utils import plot_counts

import wandb

valid_schedules = ["reducelronplateau"]

WANDB_STEP = -1

BEST_VALID = -math.inf
PREV_SAVE = None

def update_best_valid(
    valid_losses, valid_n, model, optimizer, scheduler, name,
):
    global WANDB_STEP
    global BEST_VALID
    global PREV_SAVE
    if valid_losses.evidence > BEST_VALID:
        # do not save on dryruns
        if not wandb.run._settings._offline:
            save_f = f"wandb_checkpoints/{name}/{WANDB_STEP}_{-valid_losses.evidence / valid_n:.2f}.pth"
            print(f"Saving model to {save_f}")
            Path(save_f).parent.mkdir(parents=True, exist_ok=True)
            th.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": model.config,
            }, save_f)
            if PREV_SAVE is not None:
                Path(PREV_SAVE).unlink()
            PREV_SAVE = save_f

        BEST_VALID = valid_losses.evidence
        wandb.run.summary["best_valid_ppl"] = math.exp(-BEST_VALID / valid_n)
        wandb.run.summary["best_valid_loss"] = BEST_VALID / valid_n


def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    str_list = [
        f"{prefix}: log_prob = {loss:.2f}",
        f"xent(word) = {-loss / n:.2f}",
        f"ppl = {math.exp(-loss / n):.2f}",
    ]
    if elbo is not None:
        str_list.append(f"elbo = {elbo / n:.2f}")
    total_time = None
    if start_time is not None:
        total_time = time.time() - start_time
        str_list.append(f"total_time = {total_time:.2f}s")
    print(" | ".join(str_list))
    return total_time

def count_params(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

def eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    lpz, last_states = None, None
    with th.no_grad():
        for i, batch in enumerate(iter):
            model.train(False)
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0
            mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
            if args.iterator != "bptt":
                lpz, last_states = None, None
            losses, lpz, _ = model.score(
                batch.text,
                lpz=lpz, last_states = last_states,
                mask=mask, lengths=lengths,
            )
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n

def cached_eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    with th.no_grad():
        model.train(False)
        lpz = None
        start, transition, emission = model.compute_parameters(model.word2state)
        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text = batch.text

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            N, T = text.shape

            if lpz is not None and args.iterator == "bptt":
                start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)

            log_potentials = model.clamp(text, start, transition, emission, word2state)
            losses, lpz = model.compute_loss(log_potentials, mask, lengths)

            if word2state is not None:
                idx = th.arange(N, device=model.device)
                last_words = text[idx, lengths-1]
                last_states = model.word2state[last_words]

            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n

def mixed_cached_eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    with th.no_grad():
        model.train(False)
        lpz = None

        start = model.start().cpu()
        emission = model.mask_emission(model.emission_logits(), model.word2state).cpu()

        # blocked transition
        num_blocks = 128
        block_size = model.C // num_blocks
        next_state_proj = (model.next_state_proj.weight
            if hasattr(model, "next_state_proj")
            else model.next_state_emb()
        )
        transition = th.empty(model.C, model.C, device=th.device("cpu"), dtype=emission.dtype)
        for s in range(0, model.C, block_size):
            states = range(s, s+block_size)
            x = model.trans_mlp(model.dropout(
                model.state_emb.weight[states]
                if hasattr(model.state_emb, "weight")
                else model.state_emb(th.LongTensor(states).to(model.device))
            ))
            y = (x @ next_state_proj.t()).log_softmax(-1)
            transition[states] = y.to(transition.device)

        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text = batch.text

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            N, T = text.shape

            if lpz is not None and args.iterator == "bptt":
                # hopefully this isn't too slow on cpu
                start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)

            log_potentials = model.clamp(
                text, start, transition, emission, word2state
            ).to(model.device)

            losses, lpz = model.compute_loss(log_potentials, mask, lengths)
            lpz = lpz.cpu()

            idx = th.arange(N, device=model.device)
            last_words = text[idx, lengths-1]
            last_states = model.word2state[last_words]

            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n

def train_loop(
    args, V, iter, model,
    parameters, optimizer, scheduler,
    valid_iter=None,
    verbose=False,
):
    global WANDB_STEP

    noise_scales = np.linspace(1, 0, args.noise_anneal_steps)
    total_ll = 0
    total_elbo = 0
    n = 0
    # check is performed at end of epoch outside loop as well
    checkpoint = len(iter) // (args.num_checks - 1)
    with th.enable_grad():
        lpz = None
        last_states = None
        for i, batch in enumerate(iter):
            model.train(True)
            WANDB_STEP += 1
            optimizer.zero_grad()

            text = batch.textp1 if "lstm" in args.model else batch.text
            if args.iterator == "bucket":
                lpz = None
                last_states = None

            # set noise scale
            if hasattr(model, "noise_scale"):
                noise_scale = noise_scales[
                    min(WANDB_STEP, args.noise_anneal_steps-1)
                ] if args.noise_anneal_steps > 0 else model.init_noise_scale
                model.noise_scale = noise_scale
                wandb.log({
                    "noise_scale": noise_scale,
                }, step=WANDB_STEP)

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            if model.timing:
                start_forward = timep.time()

            # check if iterator == bptt
            losses, lpz, last_states = model.score(
                text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)

            if model.timing:
                print(f"forward time: {timep.time() - start_forward}")
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens

            loss = -losses.loss / n_tokens
            if model.timing:
                start_backward = timep.time()
            loss.backward()
            if model.timing:
                print(f"backward time: {timep.time() - start_backward}")
            clip_grad_norm_(parameters, args.clip)
            if args.schedule not in valid_schedules:
                scheduler.step()
            optimizer.step()
            wandb.log({
                "running_training_loss": total_ll / n,
                "running_training_ppl": math.exp(min(-total_ll / n, 700)),
            }, step=WANDB_STEP)

            if verbose and i % args.report_every == args.report_every - 1:
                report(
                    Pack(evidence = total_ll, elbo = total_elbo),
                    n,
                    f"Train batch {i}",
                )

            if valid_iter is not None and i % checkpoint == checkpoint-1:
                v_start_time = time.time()
                if args.model == "mshmm" or args.model == "factoredhmm":
                    if args.num_classes > 2 ** 15:
                        eval_fn = mixed_cached_eval_loop
                    else:
                        eval_fn = cached_eval_loop
                elif args.model == "hmm":
                    eval_fn = cached_eval_loop
                else:
                    eval_fn = eval_loop
                valid_losses, valid_n  = eval_fn(
                    args, V, valid_iter, model,
                )
                report(valid_losses, valid_n, "Valid eval", v_start_time)
                wandb.log({
                    "valid_loss": valid_losses.evidence / valid_n,
                    "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
                }, step=WANDB_STEP)

                update_best_valid(
                    valid_losses, valid_n, model, optimizer, scheduler, args.name)

                wandb.log({
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=WANDB_STEP)
                scheduler.step(valid_losses.evidence)

                if args.log_counts > 0 and args.keep_counts > 0:
                    counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
                    c, v = counts.shape
                    cg2 = counts > 1e-2

                    wandb.log({
                        "avgcounts@1e-2": cg2.sum().item() / float(v),
                        "maxcounts@1e-2": cg2.sum(0).max().item(),
                        "mincounts@1e-2": cg2.sum(0).min().item(),
                        "maxcounts": counts.sum(0).max().item(),
                        "mincounts": counts.sum(0).min().item(),
                    }, step=WANDB_STEP)
                    del cg2
                    del counts

    return Pack(evidence = total_ll, elbo = total_elbo), n


def main():
    global WANDB_STEP
    args = get_args()
    print(args)

    set_seed(args.seed)

    device = th.device("cpu" if args.devid < 0 else f"cuda:{args.devid}")
    args.device = device
    aux_device = th.device("cpu" if args.aux_devid < 0 else f"cuda:{args.aux_devid}")
    args.aux_device = aux_device

    TEXT = torchtext.data.Field(batch_first = True)

    if args.dataset == "ptb":
        Dataset = PennTreebank
    elif args.dataset == "wikitext2":
        Dataset = WikiText2

    train, valid, test = Dataset.splits(
        TEXT,
        newline_eos = True,
    )

    TEXT.build_vocab(train)
    V = TEXT.vocab

    def batch_size_tokens(new, count, sofar):
        return max(len(new.text), sofar)
    def batch_size_sents(new, count, sofar):
        return count

    if args.iterator == "bucket":
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            sort_key = lambda x: len(x.text),
            batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,
        )
    elif args.iterator == "bptt":
        train_iter, valid_iter, test_iter = BPTTIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            bptt_len = args.bptt,
            sort = False,
        )
    else:
        raise ValueError(f"Invalid iterator {args.iterator}")

    if args.no_shuffle_train:
        train_iter.shuffle = False

    name = get_name(args)
    import tempfile
    wandb.init(project="hmm-lm", name=name, config=args, dir=tempfile.mkdtemp())
    args.name = name

    model = None
    from models.factoredhmmlm import FactoredHmmLm
    model = FactoredHmmLm(V, args)
    model.to(device)
    print(model)
    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable: {num_params:,}, {num_trainable_params:,}")
    wandb.run.summary["num_params"] = num_params

    if args.eval_only:
        model.load_state_dict(th.load(args.eval_only)["model"])
        v_start_time = time.time()
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        valid_losses, valid_n = eval_fn(
            args, V, valid_iter, model,
        )
        report(valid_losses, valid_n, f"Valid perf", v_start_time)

        t_start_time = time.time()
        test_losses, test_n = eval_fn(
            args, V, test_iter, model,
        )
        report(test_losses, test_n, f"Test perf", t_start_time)

        sys.exit()

    parameters = list(model.parameters())
    if args.optimizer == "adamw":
        optimizer = AdamW(
            parameters,
            lr = args.lr,
            betas = (args.beta1, args.beta2),
            weight_decay = args.wd,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            parameters,
            lr = args.lr,
        )
    if args.schedule == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor = 1. / args.decay,
            patience = args.patience,
            verbose = True,
            mode = "max",
        )
    elif args.schedule == "noam":
        warmup_steps = args.warmup_steps
        def get_lr(step):
            scale = warmup_steps ** 0.5 * min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return args.lr * scale
        scheduler = LambdaLR(
            optimizer,
            get_lr,
            last_epoch=-1,
            verbse = True,
        )
    else:
        raise ValueError("Invalid schedule options")

    for e in range(args.num_epochs):
        start_time = time.time()
        if args.log_counts > 0 and args.keep_counts > 0:
            # reset at START of epoch
            model.state_counts.fill_(0)
        train_losses, train_n = train_loop(
            args, V, train_iter, model,
            parameters, optimizer, scheduler,
            valid_iter = valid_iter if not args.overfit else None,
            verbose = True,
        )
        total_time = report(train_losses, train_n, f"Train epoch {e}", start_time)

        v_start_time = time.time()
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        valid_losses, valid_n  = eval_fn(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid epoch {e}", v_start_time)

        if args.schedule in valid_schedules:
            scheduler.step(
                valid_losses.evidence if not args.overfit else train_losses.evidence)

        update_best_valid(
            valid_losses, valid_n, model, optimizer, scheduler, args.name)

        wandb.log({
            "train_loss": train_losses.evidence / train_n,
            "train_ppl": math.exp(-train_losses.evidence / train_n),
            "epoch_time": total_time,
            "valid_loss": valid_losses.evidence / valid_n,
            "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
            "best_valid_loss": BEST_VALID / valid_n,
            "best_valid_ppl": math.exp(-BEST_VALID / valid_n),
            "epoch": e,
        }, step=WANDB_STEP)

        if args.log_counts > 0 and args.keep_counts > 0:
            counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
            c, v = counts.shape
            cg2 = counts > 1e-2

            # state counts
            # log these once per epoch, then set back to zero
            sc0 = (model.state_counts == 0).sum()
            sc1 = (model.state_counts == 1).sum()
            sc2 = (model.state_counts == 2).sum()
            sc3 = (model.state_counts == 3).sum()
            sc4 = (model.state_counts == 4).sum()
            sc5 = (model.state_counts >= 5).sum()

            wandb.log({
                "avgcounts@1e-2": cg2.sum().item() / float(v),
                "maxcounts@1e-2": cg2.sum(0).max().item(),
                "mincounts@1e-2": cg2.sum(0).min().item(),
                "maxcounts": counts.sum(0).max().item(),
                "mincounts": counts.sum(0).min().item(),

                "statecounts=0": sc0,
                "statecounts=1": sc1,
                "statecounts=2": sc2,
                "statecounts=3": sc3,
                "statecounts=4": sc4,
                "statecounts>=5": sc5,
            }, step=WANDB_STEP)
            del cg2
            del counts

    # won't use best model. Rerun with eval_only
    t_start_time = time.time()
    test_losses, test_n = eval_fn(
        args, V, test_iter, model,
    )
    report(test_losses, test_n, f"Test perf", t_start_time)

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
