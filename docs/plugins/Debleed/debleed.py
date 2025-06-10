#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2024, Jason Lequyer and Laurence Pelletier.
# All rights reserved.
# Sinai Health System - Lunenfeld-Tanenbaum Research Institute
# 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada
#
# Torch-free port (NumPy only).

import os, sys, time
import numpy as np
from tifffile import imread, imwrite
from scipy.optimize import linear_sum_assignment
import concurrent.futures as cf
import multiprocessing as mp

def _solve_chunk(hj_vec: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Run _solve_one_plane on every index in hj_vec and return results."""
    out = []
    for hj in hj_vec:
        out.append(_solve_one_plane(int(hj)))   # pass scalar
    return out


# helper to yield index vectors of a chosen grainsize
def _chunks(n_planes, grainsize=512):
    for start in range(0, n_planes, grainsize):
        yield np.arange(start, min(start + grainsize, n_planes), dtype=np.int64)



# ------------------------------------------------------------

mp.set_start_method("fork", force=True)     # ⇦ one line

# ───────── helper that processes ONE colour plane ─────────
def _process_compare_channel(ow: int) -> np.ndarray:
    """
    Re-runs the existing per-channel logic and returns the indices
    (whurdif) that must be zeroed out in dom2d for that channel.
    Uses the global read-only arrays created in the parent process.
    """
    curcomp = RP(compare3d[:, ow : ow + 1, :, :])
    curcomp = unfold(curcomp)
    curcomp = curcomp.reshape(-1, patsize + 4, patsize + 4,
                              curcomp.shape[-1])[:, 2:-2, 2:-2, :]

    # local rfdom (avoid races)  ──────────────────────────────────
    rfdom_local = rfdom.copy()
    side = int(np.sqrt(rfdom_local.shape[1]))          # e.g. 18
    rfdom_local = rfdom_local.reshape(4, side, side, -1).astype(np.float32)
    rfdom_local[0] = curcomp[0, 2:,    1:-1, :]
    rfdom_local[1] = curcomp[0, :-2,   1:-1, :]
    rfdom_local[2] = curcomp[0, 1:-1,  2:,   :]
    rfdom_local[3] = curcomp[0, 1:-1, :-2,   :]

    rfdom_local = rfdom_local.reshape(4, -1, rfdom_local.shape[-1])

    rfdomstd = (_abs(rfdom_local[0] - rfdom_local[1]) +
                _abs(rfdom_local[0] - rfdom_local[2]) +
                _abs(rfdom_local[0] - rfdom_local[3]) +
                _abs(rfdom_local[1] - rfdom_local[2]) +
                _abs(rfdom_local[1] - rfdom_local[3]) +
                _abs(rfdom_local[2] - rfdom_local[3]))

    rfdommean = _unsqueeze(_mean(rfdomstd, axis=0), 0)
    rfdommean[rfdommean == 0] = 1
    rfdomstd /= rfdommean

    compdiff = np.sum(_abs(rf - rfdomstd), axis=0)

    curcomp = curcomp.reshape(1, -1, curcomp.shape[-1])
    trumeancomp = _unsqueeze(_mean(curcomp, axis=1), 1)
    trumeancomp[trumeancomp == 0] = FLOAT32_MIN

    whurdif = np.where((compdiff < basediff) &
                       (trumeanbase[0, 0, :] < trumeancomp[0, 0, :]))[0]
    return whurdif            # 1-D int array (can be empty)

def _solve_one_plane(hj: int) -> tuple[int, np.ndarray]:
    """
    Run the full optimisation for channel `hj`.
    Returns (hj, new_patch) so the caller can put it back.
    Uses all globals created above (base2d, rf, neighbours, etc.),
    so the arithmetic is identical to the original serial loop.
    """
    curpat = _clone(base2d[0, :, :, hj])                       # (20,20)
    curpat = _repeat(curpat, patsize + 4, patsize + 4, 1, 1)   # (20,20,20,20)

    curcost         = _zeros_like(curpat)
    curcost[top_idx] = _abs(curpat[toprit_idx] - curpat[rit_idx]) + FLOAT32_MIN
    curcost[bot_idx] = _abs(curpat[botlef_idx] - curpat[lef_idx]) + FLOAT32_MIN
    curcost[lef_idx] = _abs(curpat[toplef_idx] - curpat[top_idx]) + FLOAT32_MIN
    curcost[rit_idx] = _abs(curpat[botrit_idx] - curpat[bot_idx]) + FLOAT32_MIN

    # ─── ranking, jitter, and inflation (unchanged) ───
    totcost = np.stack([curcost[top_idx],
                        curcost[bot_idx],
                        curcost[lef_idx],
                        curcost[rit_idx]])
    sort     = np.argsort(totcost, axis=0)
    maxcost  = sort[3];  mincost = sort[0];  ndcost  = sort[1];  rdcost = sort[2]

    topmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 0)
    botmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 1)
    lefmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 2)
    ritmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 3)

    topmin = (mincost == 0).astype(np.float32)
    botmin = (mincost == 1).astype(np.float32)
    lefmin = (mincost == 2).astype(np.float32)
    ritmin = (mincost == 3).astype(np.float32)

    topnd = (ndcost == 0).astype(np.float32)
    botnd = (ndcost == 1).astype(np.float32)
    lefnd = (ndcost == 2).astype(np.float32)
    ritnd = (ndcost == 3).astype(np.float32)

    toprd = (rdcost == 0).astype(np.float32)
    botrd = (rdcost == 1).astype(np.float32)
    lefrd = (rdcost == 2).astype(np.float32)
    ritrd = (rdcost == 3).astype(np.float32)

    curcost[top_idx] = 0
    curcost[bot_idx] = 0
    curcost[lef_idx] = 0
    curcost[rit_idx] = 0

    curcost[top_idx] += topmax + topmin + topnd + toprd
    curcost[bot_idx] += botmax + botmin + botnd + botrd
    curcost[lef_idx] += lefmax + lefmin + lefnd + lefrd
    curcost[rit_idx] += ritmax + ritmin + ritnd + ritrd

    curcost[top_idx] += (np.random.rand(curcost[top_idx].shape[0]) / 10).astype(np.float32)
    curcost[bot_idx] += (np.random.rand(curcost[bot_idx].shape[0]) / 10).astype(np.float32)
    curcost[lef_idx] += (np.random.rand(curcost[lef_idx].shape[0]) / 10).astype(np.float32)
    curcost[rit_idx] += (np.random.rand(curcost[rit_idx].shape[0]) / 10).astype(np.float32)

    curcost[bord_idx] = FLOAT32_MAX_4RT

    rl1 = rf[0, :, :, hj] == 2
    rr1 = rf[1, :, :, hj] == 2
    rt1 = rf[2, :, :, hj] == 2
    rb1 = rf[3, :, :, hj] == 2

    if rl1.any():
        _inflate_cost(curcost, rf[0, :, :, hj] == 1, rl1, FLOAT32_MAX_SQRT)
    if rr1.any():
        _inflate_cost(curcost, rf[1, :, :, hj] == 1, rr1, FLOAT32_MAX_SQRT)
    if rt1.any():
        _inflate_cost(curcost, rf[2, :, :, hj] == 1, rt1, FLOAT32_MAX_SQRT)
    if rb1.any():
        _inflate_cost(curcost, rf[3, :, :, hj] == 1, rb1, FLOAT32_MAX_SQRT)

    curcost = curcost[1:-1, 1:-1, 1:-1, 1:-1]
    curpat  = curpat [1:-1, 1:-1, 1:-1, 1:-1]

    curcost = curcost.reshape(curcost.shape[0]*curcost.shape[1], -1)
    curpat  = curpat.reshape(curpat.shape[0]*curpat.shape[1],  -1)

    curcost[curcost == 0] = FLOAT32_MAX
    row_ind, col_ind = linear_sum_assignment(curcost.astype(np.float32))

    newpat = _clone(curpat[0, :])
    newpat[row_ind] = curpat[0, col_ind]
    return hj, newpat.reshape(patsize + 2, patsize + 2)
# ───────── end helper ─────────────────────

# ───────────────────── Helper shims for selected torch ops ──────────────────────
def _clone(a):                 # torch.clone
    return a.copy()

def _unsqueeze(a, dim):        # torch.unsqueeze
    return np.expand_dims(a, dim)

def _repeat(a, *reps):
    """
    NumPy equivalent of torch.Tensor.repeat.
    Supports more repeat factors than ndim (pads new dims at the front).
    """
    reps = tuple(int(r) for r in reps)
    if len(reps) > a.ndim:                       # need to add leading dims
        a = a.reshape((1,) * (len(reps) - a.ndim) + a.shape)
    elif len(reps) < a.ndim:                     # pad with 1s on the left
        reps = (1,) * (a.ndim - len(reps)) + reps
    return np.tile(a, reps)


def _ones_like(a, dtype=None):
    return np.ones_like(a, dtype=dtype or a.dtype)

def _zeros_like(a, dtype=None):
    return np.zeros_like(a, dtype=dtype or a.dtype)

def _cat(tensors, axis=0):
    return np.concatenate(tensors, axis=axis)

def _where(*args, **kwargs):
    return np.where(*args, **kwargs)

def _abs(x):
    return np.abs(x)

def _mean(x, axis=None):
    return np.mean(x, axis=axis)

# ReflectionPad2d analogue (pad = (l, r, t, b))
def _reflection_pad2d(arr, pad=(2, 2, 2, 2)):
    l, r, t, b = pad
    return np.pad(
        arr,
        pad_width=((0, 0), (0, 0), (t, b), (l, r)),
        mode="reflect",
    )

def _unfold4d(x, ksize):
    """
    Exact analogue of torch.nn.Unfold(stride=1, dilation=1, padding=0).

    Args
    ----
    x : ndarray, shape (B, C, H, W)
    ksize : tuple (kH, kW)

    Returns
    -------
    patches : ndarray, shape (B, C * kH * kW, L)
              where L = (H - kH + 1) * (W - kW + 1)
    """
    B, C, H, W   = x.shape
    kH, kW       = ksize
    # sliding_window_view → (B, C, out_h, out_w, kH, kW)
    win          = np.lib.stride_tricks.sliding_window_view(
                       x, (kH, kW), axis=(2, 3)
                   )
    out_h, out_w = win.shape[2:4]
    # bring kernel dims in front of spatial dims so the flatten order
    # matches torch (row-major over the kernel, row-major over spatial)
    win          = win.transpose(0, 1, 4, 5, 2, 3)
    patches      = win.reshape(B, C * kH * kW, out_h * out_w)
    return patches


def _fold4d(patches, output_size, ksize):
    """
    Exact analogue of torch.nn.Fold(stride=1, dilation=1, padding=0).

    Args
    ----
    patches : ndarray, shape (B, C * kH * kW, L)
    output_size : tuple (H, W)
    ksize : tuple (kH, kW)

    Returns
    -------
    out : ndarray, shape (B, C, H, W)
    """
    B, CK2, L    = patches.shape
    kH, kW       = ksize
    C            = CK2 // (kH * kW)
    H, W         = output_size
    out          = np.zeros((B, C, H, W), dtype=patches.dtype)
    counter      = np.zeros_like(out)

    # restore kernel dims and spatial index
    patches = patches.reshape(B, C, kH, kW, L)

    # row-major enumeration of sliding windows (identical to Unfold)
    idx = 0
    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            out[:, :, i : i + kH, j : j + kW]     += patches[..., idx]
            counter[:, :, i : i + kH, j : j + kW] += 1
            idx += 1

    counter[counter == 0] = 1
    return out / counter

# helper: broadcast a 2-D mask onto curcost’s first or last two axes
def _row_mask(mask2d):
    # shape (H, W, 1, 1) → acts on axes 0 & 1
    return mask2d[:, :, None, None]

def _col_mask(mask2d):
    # shape (1, 1, H, W) → acts on axes 2 & 3
    return mask2d[None, None, :, :]

def _inflate_cost(curcost4d, mask_a2d, mask_b2d, big):
    """
    Torch-exact: curcost4d[mask_a2d, mask_b2d] = big
    Pair-wise (zipped) assignment, NOT outer product.
    """
    h, w = mask_a2d.shape                 # here (18, 18)
    flat = curcost4d.reshape(h * w, h * w)

    # flatten masks and take the True positions *in order*
    rows = np.flatnonzero(mask_a2d.ravel())
    cols = np.flatnonzero(mask_b2d.ravel())

    # they are guaranteed the same length in the original algorithm
    n = min(len(rows), len(cols))
    if n:
        flat[rows[:n], cols[:n]] = big



# ─────────────────────────────── constants ──────────────────────────────────────
FLOAT32_MAX      = np.float32(np.finfo(np.float32).max)
FLOAT32_MAX_SQRT = np.sqrt(FLOAT32_MAX)
FLOAT32_MAX_4RT  = np.sqrt(np.sqrt(FLOAT32_MAX))
FLOAT32_MAX_8RT  = np.sqrt(np.sqrt(np.sqrt(FLOAT32_MAX)))
FLOAT32_MIN      = np.float32(np.finfo(np.float32).tiny)
patsize          = np.int32(16)

varkern = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

# ────────────────────────────────── main ────────────────────────────────────────
if __name__ == "__main__":

    functions  = [
        lambda x: x,
        lambda x: np.rot90(x),
        lambda x: np.rot90(np.rot90(x)),
        lambda x: np.rot90(np.rot90(np.rot90(x))),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: np.rot90(np.fliplr(x)),
        lambda x: np.fliplr(np.rot90(x)),
    ]
    ifunctions = [
        lambda x: x,
        lambda x: np.rot90(x, -1),
        lambda x: np.rot90(np.rot90(x, -1), -1),
        lambda x: np.rot90(np.rot90(np.rot90(x, -1), -1), -1),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: np.rot90(np.fliplr(x)),
        lambda x: np.fliplr(np.rot90(x)),
    ]

    file_name = sys.argv[1]
    oz        = int(sys.argv[2]) - 1     # 0-based channel index
    print(file_name)

    start_time = time.time()

    inp = imread(file_name)
    np.set_printoptions(threshold=sys.maxsize)

    if inp.shape[-1] == 3:               # RGB → channel-first
        inp = np.swapaxes(inp, -2, -1)
        inp = np.swapaxes(inp, -3, -2)

    ogshape = inp.shape                 # (Z,H,W)
    inp     = inp.reshape(-1, ogshape[-2], ogshape[-1])

    inpnorm = inp.astype(np.float32)    # working copy

    # ───────────── prep main tensors (NumPy arrays) ──────────────
    base2d    = _clone(inpnorm[oz : oz + 1, :, :])    # (1,H,W)
    compare3d = _clone(inpnorm)                       # (Z,H,W)

    base2d    = _unsqueeze(base2d, 0)                 # (1,1,H,W)
    compare3d = _unsqueeze(compare3d, 0)              # (1,Z,H,W)
    curcomp   = _clone(compare3d[:, oz : oz + 1, :, :])

    # Optional inclusion mask from CSV
    try:
        file_path = os.path.splitext(file_name)[0] + ".csv"
        matrix = np.genfromtxt(file_path, delimiter=",", skip_header=1)[:, 1:]
        matrix = np.nan_to_num(matrix, nan=1)
    
        cur_rwo = matrix[oz]
        included = np.where(np.isclose(cur_rwo, 1, atol=1e-6))[0]   # ← tolerant test
    
        if included.size:                                           # ← fallback
            compare3d = compare3d[:, included, :, :]
        else:
            print("Matrix row has no ‘1’s — using all channels.")
    except Exception as e:
        print("CSV not found or unreadable:", e, "\nUsing all channels.")


    dummy   = _ones_like(base2d)
    RP      = lambda x: _reflection_pad2d(x, (2, 2, 2, 2))
    unfold  = lambda x: _unfold4d(x, (patsize + 4, patsize + 4))
    fold = lambda x: _fold4d(x, (ogshape[-2], ogshape[-1]), (patsize, patsize))

    base2d = RP(base2d)
    dummy  = RP(dummy)

    # ─────────────── Prepare reflection masks (identical logic) ────────────────
    rflef = _zeros_like(base2d)
    rfrit = _zeros_like(base2d)
    rftop = _zeros_like(base2d)
    rfbot = _zeros_like(base2d)

    rflef[0, 0, 2:-2, 1]  = 2
    rflef[0, 0, 2:-2, 2]  = 1
    rfrit[0, 0, 2:-2, -2] = 2
    rfrit[0, 0, 2:-2, -3] = 1
    rftop[0, 0, 1, 2:-2]  = 2
    rftop[0, 0, 2, 2:-2]  = 1
    rfbot[0, 0, -2, 2:-2] = 2
    rfbot[0, 0, -3, 2:-2] = 1

    evens = np.zeros((patsize + 4, patsize + 4), dtype=np.float32)
    odds  = np.zeros_like(evens)

    for i in range(1, evens.shape[0] - 1):
        for j in range(1, evens.shape[1] - 1):
            if i in (1, evens.shape[0] - 2) or j in (1, evens.shape[1] - 2):
                (evens if (i + j) % 2 == 0 else odds)[i, j] = 1

    rf      = _cat([rflef, rfrit, rftop, rfbot], axis=0)
    base2d  = unfold(base2d)
    dummy   = unfold(dummy)
    rf      = unfold(rf)

    odds_idx  = _where(odds > 0)
    evens_idx = _where(evens > 0)

    base2d = base2d.reshape(-1, patsize + 4, patsize + 4, base2d.shape[-1])
    dummy  = dummy.reshape(-1, patsize + 4, patsize + 4, base2d.shape[-1])
    rf     = rf.reshape(-1, patsize + 4, patsize + 4, base2d.shape[-1])

    dom2d  = _clone(base2d)

    # Shift RF masks as in original (swap border/core areas)
    for idx in range(4):
        tmp = _clone(rf[idx, 2:-2, 1:-1, :]) if idx < 2 else _clone(rf[idx, 1:-1, 2:-2, :])
        rf[idx, :, :, :] *= 0
        if idx < 2:
            rf[idx, 2:-2, 1:-1, :] = tmp
        else:
            rf[idx, 1:-1, 2:-2, :] = tmp

    # Build neighbourhood lookup tensors (top, bot, etc.) – unchanged
    shp = (patsize + 4, patsize + 4, patsize + 4, patsize + 4)
    top = np.zeros(shp, dtype=np.float32)
    bot = np.zeros(shp, dtype=np.float32)
    lef = np.zeros(shp, dtype=np.float32)
    rit = np.zeros(shp, dtype=np.float32)
    toplef = np.zeros(shp, dtype=np.float32)
    toprit = np.zeros(shp, dtype=np.float32)
    botlef = np.zeros(shp, dtype=np.float32)
    botrit = np.zeros(shp, dtype=np.float32)
    cn     = np.zeros(shp, dtype=np.float32)
    bord   = np.zeros(shp, dtype=np.float32)

    for ka in range(1, patsize + 3):
        for kb in range(1, patsize + 3):
            top    [ka, kb, ka - 1, kb    ] = 1
            bot    [ka, kb, ka + 1, kb    ] = 1
            lef    [ka, kb, ka    , kb - 1] = 1
            rit    [ka, kb, ka    , kb + 1] = 1
            toplef [ka, kb, ka - 1, kb - 1] = 1
            toprit [ka, kb, ka - 1, kb + 1] = 1
            botlef [ka, kb, ka + 1, kb - 1] = 1
            botrit [ka, kb, ka + 1, kb + 1] = 1
            cn     [ka, kb, ka    , kb    ] = 1

            if ka in (1, patsize + 2) or kb in (1, patsize + 2):
                (bord if (ka + kb) % 2 else bord)[ka, kb][odds_idx if (ka + kb) % 2 == 0 else evens_idx] = 1
            if ka == 1:                    bord[ka, kb, ka + 1, kb] = 1
            if kb == 1:                    bord[ka, kb, ka, kb + 1] = 1
            if ka == patsize + 2:          bord[ka, kb, ka - 1, kb] = 1
            if kb == patsize + 2:          bord[ka, kb, ka, kb - 1] = 1

    # Convert neighbourhoods to index tuples once
    top_idx, bot_idx  = _where(top == 1), _where(bot == 1)
    lef_idx, rit_idx  = _where(lef == 1), _where(rit == 1)
    toplef_idx, toprit_idx = _where(toplef == 1), _where(toprit == 1)
    botlef_idx, botrit_idx = _where(botlef == 1), _where(botrit == 1)
    cn_idx   = _where(cn == 1)
    bord_idx = _where(bord == 1)

    # ───────────────────── main per-patch optimisation loop ─────────────────────
    _ix = np.ix_  # just a short alias; np.ix_(row_bool, col_bool)
    
    N_WORKERS = min( os.cpu_count() or 1, base2d.shape[-1] )   # after arrays built
    
    
    
    
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for res_list in pool.map(_solve_chunk, _chunks(base2d.shape[-1], 512)):
            for hj, newpat in res_list:
                dom2d[0, 1:-1, 1:-1, hj] = newpat

    # ──────────────────────────────────────────────────────



    # ──────────────────── Finish bleed-through suppression ──────────────────────
    base2d = base2d[:, 2:-2, 2:-2, :]
    dom2d  = dom2d[:, 2:-2, 2:-2, :]
    dummy  = dummy[:, 2:-2, 2:-2, :]
    rf     = rf   [:, 2:-2, 2:-2, :]

    rf      = rf[:, 1:-1, 1:-1, :]
    rfdom   = _clone(rf)

    rf[0] = base2d[0, 2:, 1:-1, :]
    rf[1] = base2d[0, :-2, 1:-1, :]
    rf[2] = base2d[0, 1:-1, 2:, :]
    rf[3] = base2d[0, 1:-1, :-2, :]

    rfdom[0] = dom2d[0, 2:, 1:-1, :]
    rfdom[1] = dom2d[0, :-2, 1:-1, :]
    rfdom[2] = dom2d[0, 1:-1, 2:, :]
    rfdom[3] = dom2d[0, 1:-1, :-2, :]

    rf = rf.reshape(rf.shape[0], -1, rf.shape[-1])
    rf = (_abs(rf[0] - rf[1]) + _abs(rf[0] - rf[2]) + _abs(rf[0] - rf[3]) +
          _abs(rf[1] - rf[2]) + _abs(rf[1] - rf[3]) + _abs(rf[2] - rf[3]))
    rfmean = _unsqueeze(_mean(rf, axis=0), 0)
    rfmean[rfmean == 0] = 1
    rf /= rfmean

    rfdom = rfdom.reshape(rfdom.shape[0], -1, rfdom.shape[-1])
    rfdomstd = (_abs(rfdom[0] - rfdom[1]) + _abs(rfdom[0] - rfdom[2]) +
                _abs(rfdom[0] - rfdom[3]) + _abs(rfdom[1] - rfdom[2]) +
                _abs(rfdom[1] - rfdom[3]) + _abs(rfdom[2] - rfdom[3]))
    rfdommean = _unsqueeze(_mean(rfdomstd, axis=0), 0)
    rfdommean[rfdommean == 0] = 1
    rfdomstd /= rfdommean

    basediff = _abs(rf - rfdomstd)
    basediff = np.sum(basediff, axis=0)

    # reshape base/patch tensors for later comparisons
    base2d = base2d.reshape(base2d.shape[0], -1, base2d.shape[-1])
    dom2d  = dom2d .reshape(dom2d .shape[0], -1, dom2d .shape[-1])
    dummy  = dummy .reshape(dummy .shape[0], -1, dummy .shape[-1])

    trumeanbase = _mean(base2d, axis=1)
    trumeanbase = trumeanbase.reshape(1, 1, trumeanbase.shape[-1])
    trumeanbase[trumeanbase == 0] = FLOAT32_MIN

    # Zero-out base2d/dom2d as in original
    base2d *= 0
    dom2d  *= 0

    # counts
    curcomp = RP(curcomp)
    curcomp = unfold(curcomp)
    curcomp = curcomp.reshape(-1, patsize + 4, patsize + 4, curcomp.shape[-1])
    curcomp = curcomp[:, 2:-2, 2:-2, :]
    curcomp = curcomp.reshape(1, -1, curcomp.shape[-1])
    dom2d   = _clone(curcomp)
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for whurdif in pool.map(_process_compare_channel,
                                range(compare3d.shape[1])):
            if whurdif.size:                      # skip empty returns
                dom2d[:, :, whurdif] *= 0


    # Fold back to image space
    dom2d  = fold(dom2d)
    dummy  = fold(dummy)
    dom2d  = dom2d / dummy
    dom2d  = np.clip(dom2d, np.min(inp[oz, :, :]), np.max(inp[oz, :, :]))
    target_dtype = inp[oz, :, :].dtype          # whatever dtype that slice has
    dom2d = dom2d.astype(target_dtype)   

    # Save result
    imwrite(file_name[:-4] + f"_Channel_{oz + 1}_debleed.tif",
            dom2d[0, 0], imagej=True)

    print(f"--- {time.time() - start_time:.2f} seconds ---")
