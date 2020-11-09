import sys
import os
import time
import torch
import numpy as np

import tvm
from tvm import autotvm
import tvm.runtime
import tvm.te

sizes = [32, 64, 128, 256, 512, 1024, 2048]

@autotvm.template("hmm_runner")
def hmm_runner(dtype, nn):
    #n_num_step = 128
    #n_num_hidden = 1152
    #n_batch_size = 4
    #num_step = tvm.var("num_step")

    #nn = 1152
    #nn = 128
    #bb = 32
    #tt = 128
    n = tvm.runtime.convert(nn)
    m = n

    b = tvm.te.var("batch")
    t = tvm.te.var("num_step")
    l = n
    k = tvm.te.reduce_axis((0, l), name='k')
    k2 = tvm.te.reduce_axis((0, l), name='k2')
    #X = tvm.te.placeholder((t-1, b, n, m), name="X", dtype=dtype)
    X = tvm.te.placeholder((t, b, n, m), name="X", dtype=dtype)

    s_state = tvm.te.placeholder((t, b, n))
    s_init = tvm.te.compute((1, b, n), lambda a, b, c: 0.0)
    M = tvm.te.compute(
        (t, b, n),
        lambda t, bb, ii: tvm.te.max(s_state[t-1, bb, k] + X[t-1, bb, k, ii], axis=k),
        name="M")

    M2 = tvm.te.compute(
        (t, b, n),
        lambda t, bb, ii: tvm.te.sum(tvm.te.exp(s_state[t-1, bb, k2] + X[t-1, bb, k2, ii]
                                            - M[t, bb, ii]), axis=k2),
        name="M2")
    C = tvm.te.compute(
        (t, b, n),
        #lambda t, bb, ii: M[t, bb, ii] + M2[t, bb,ii],
        lambda t, bb, ii: tvm.te.log(M2[t, bb, ii]) + M[t, bb, ii],
        name='C')

    s_scan = tvm.te.scan(s_init, C, s_state, inputs=[X])

    s = tvm.te.create_schedule(s_scan.op)
    #tvm.lower(s, [X], simple_mode=True )

    cfg = autotvm.get_config()
    cfg.define_knob("y_t", [8])
    cfg.define_knob("x_t", [16])
    cfg.define_knob("sm", [24])
    cfg.add_flop(1)

    num_thread_y = 1# cfg["y_t"].val
    num_thread_x = 128
    num_sm = 1 #cfg["sm"].val

    PERSIST_KERNEL = False
    DETECT_GLOBAL_BARRIER = False
    detect_global_barrier = DETECT_GLOBAL_BARRIER

    s = tvm.te.create_schedule(s_scan.op)
    CL = C
    SS = s.cache_read(s_state, "shared", [M, M2])
    #SL = s.cache_read(SS, "local", [M])
    #SS2 = s.cache_read(s_state, "shared", [M2])
    #SL2 = s.cache_read(SS2, "local", [M2])

    WhhL = s.cache_read(X, "local", [M, M2])

    block_x = tvm.te.thread_axis((0, num_sm), "blockIdx.x")
    thread_x = tvm.te.thread_axis((0, num_thread_x), "threadIdx.x")
    block_y = tvm.te.thread_axis((0, b), "blockIdx.y")
    thread_y = tvm.te.thread_axis((0, 1), "threadIdx.y")

    batch = s[s_init].op.axis[1]
    h = s[s_init].op.axis[2]

    s[s_init].bind(batch, block_y)
    s[s_init].bind(h, thread_x)

    batch = s[CL].op.axis[1]
    h = s[CL].op.axis[2]

    s[CL].bind(batch, block_y)
    s[CL].bind(h, thread_x)
    s[SS].compute_at(s[CL], h)

    #s[SS].bind(s[SS].op.axis[1], block_y)
    s[SS].bind(s[SS].op.axis[2], thread_x)
    s[M2].compute_at(s[CL], h)
    s[M].compute_at(s[CL], h)

    s[WhhL].compute_at(s[CL], h)
    return s, [X, s_scan]

#task = autotvm.task.create(hmm, args=('float32',), target='cuda', target_host="llvm")

def log_eye(K, dtype, device):
    x = torch.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

def log_eye_cat(x):
    K = x.shape[-1]
    batch = x.shape[1]
    return torch.cat([
        x,
        log_eye(K, x.dtype, x.device).view(1, 1, K, K).expand(1, batch, K, K),
    ], dim=0)

from tvm.contrib.dlpack import to_pytorch_func

def get_fb(size):
    """
    with autotvm.apply_history_best(f'best_hmm_k{size}.log'):
        with tvm.target.create("cuda"):
            s_mult, arg_bufs = hmm_runner('float32', size)
            mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
            hmm_pytorch = to_pytorch_func(mod)
    """
    with tvm.target.create("cuda"):
        s_mult, arg_bufs = hmm_runner('float32', size)
        mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
        hmm_pytorch = to_pytorch_func(mod)

    # if the padding doesn't make a difference this must be an inclusive scan
    # x: batch x time x zt x zt-1
    #@profile
    def fb(x, mask=None):
        batch, time, size, _ = x.shape
        lex = log_eye(size, dtype=x.dtype, device=x.device)
        # need time x batch x zt-1, zt
        x = x.permute(1, 0, 3, 2)
        if mask is not None:
            mask = mask.t()
            x[~mask[1:]] = lex
            """
            x.masked_scatter_(
                ~mask[1:,:,None,None],
                # EXPAND HERE IS BAD?
                lex[None,None].expand(x.shape),
            )
            import pdb; pdb.set_trace()
            """

        """
        out_fb = torch.empty(time+1, batch * 2, size, device=x.device)
        out_fb.fill_(float("-inf"))
        hmm_pytorch(
            log_eye_cat(torch.cat([x, x.flip(0).transpose(-2, -1)], 1)),
            out_fb,
        )

        out_fb2 = torch.empty(time+1, batch * 2, size, device=x.device)
        out_fb2.fill_(float("-inf"))
        hmm_pytorch(
            log_eye_cat(x),
            out_fb2[:,:batch],
        )
        hmm_pytorch(
            log_eye_cat(x.flip(0).transpose(-2, -1)),
            out_fb2[:,batch:],
        )
        alphas = out_fb[:, :batch]
        betas = out_fb[:, batch:].flip(0)
        """

        out_fb = torch.empty(2, time+1, batch, size, device=x.device)
        out_fb.fill_(float("-inf"))
        inp = torch.empty(time+1, batch, size, size, device=x.device)
        inp[-1] = lex
        # forward
        inp[:-1] = x
        hmm_pytorch(inp, out_fb[0])
        # backward
        inp[range(time-1, -1, -1)] = x.transpose(-2, -1)
        hmm_pytorch(inp, out_fb[1])

        alphas = out_fb[0]
        betas = out_fb[1].flip(0) # pay the memory cost here
        # not sure if i can flip the argument to hmm_pytorch

        log_marginals = x
        log_marginals += alphas[:-1].view(time, batch, size, 1)
        log_marginals += betas[1:].view(time, batch, 1, size)
        log_marginals -= alphas[-1].logsumexp(-1).view(1, -1, 1, 1)
        if mask is not None:
            log_marginals.masked_fill_(~mask[1:,:,None,None], float("-inf"))
        log_marginals = log_marginals.permute(1, 0, 3, 2)
        return log_marginals, alphas
        #marginals = log_marginals.exp()
        # switch back marginals: batch x time x zt x zt-1
        #return marginals, alphas, betas, log_marginals

    return fb

if __name__ == "__main__":
    from tvm import autotvm
    import logging
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    for size in sizes:
        task = autotvm.task.create(hmm_runner, args=('float32', size),
                                   target='cuda', target_host="llvm")

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=5),
            runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=100,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(f'hmm_k{size}.log')])

        autotvm.record.pick_best(f"hmm_k{size}.log", f"best_hmm_k{size}.log")
