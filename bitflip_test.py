#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Dimitar Kretski
"""bitflip_test.py v2.11  (Dimitar Kretski)
How much do real bit-flips (single-event upsets) in STORED WEIGHTS hurt a neural network, in the number
format it would actually be stored in, and which cheap protections pay off?

No claim about any optimizer is made here.

Fault model
  * A fault flips ONE bit of the stored memory image. A flip is drawn by choosing an array of the image
    (proportional to its number of allowed bits), an element uniformly, and a bit position uniformly.
  * Stored formats are simulated at the REAL bit level (not "convert back to float32 and flip there"):
        fp32 : IEEE-754 binary32 words (uint32)
        fp16 : IEEE-754 binary16 words (uint16)
        bf16 : bfloat16 words (uint16 = top 16 bits of binary32, round-to-nearest-even)
        int8 : symmetric PER-TENSOR int8 weights (uint8 view) + float32 scales (the scales are also in the fault domain)
        int8pc: symmetric PER-CHANNEL int8: one float32 scale per output channel of every Conv2d/Linear weight
               (rows of the [out, ...] tensor); each 1-D bias tensor keeps ONE scale. Biases are int8 here,
               whereas real toolchains usually keep them int32 - a documented simplification.
  * k = number of flips per memory image. TMR keeps 3 images, so it is exposed to 3k flips (same per-bit rate).
  * STORAGE format only: weights are stored (and corrupted) in fp16/bf16/int8, then DECODED to float32 and run with
    float32 arithmetic. Hardware that really computes in fp16/bf16/int8 (overflow behaviour, accumulators) is not modelled.
  * Fault domain = every float tensor of the model: parameters AND float buffers (e.g. BatchNorm running mean/var).
    Each 1-D tensor (bias, BN weight/bias/stats) is one quantization group in the int8 formats.
  * Not modelled: faults in activations, compute units, control logic, instruction memory, optimizer state
    (train mode can flip Adam state, fp32 only), ECC behaviour of any specific platform.

Protections (extra model copies in brackets)
  none   no protection                                                              [no copy]
  clamp  decode, NaN/Inf -> 0, clip each tensor to 1.1 x its clean |max|             [no copy; per-tensor bounds + one clip pass]
  scrub  position-weighted checksum per tensor segment; mismatch -> restore from a golden copy   [1 extra copy]
         ASSUMPTION: golden copy and checksums are fault-free (ECC / rad-hard memory). Under that assumption,
         and unless two flips cancel in the checksum, the result is the clean model.
  scrub_scales  (int8 / int8pc only) golden copy + ONE position-weighted checksum over the float32 scales ONLY; weights
         are left unprotected. Same assumption as scrub (fault-free golden copy). Also reported combined with clamp.
  tmr    3 copies, bitwise majority vote. Fails only if >=2 copies are hit at the same bit.   [2 extra copies]
  rollback (train mode only) restore last healthy snapshot when loss is non-finite / spikes

Usage
  python bitflip_test.py --mode selftest
  python bitflip_test.py --fake-data --quick                            # code check only (random data)
  python bitflip_test.py --mode inference --seeds 3 --epochs 8 --formats fp32,fp16,bf16,int8 --bit-trials 25 --trials 200
  python bitflip_test.py --mode inference --formats fp32 --ks 100 --exclude-bits 30 --skip-bit-sweep   # mechanism test
  python bitflip_test.py --mode inference --formats int8,int8pc --ks 100 --trials 50 --bit-trials 25 --out bitflip_pc.jsonl   # per-tensor vs per-channel scales
  python bitflip_test.py --mode inference --arch resnet8 --formats fp32,int8,int8pc --ks 100 --trials 300 --bit-trials 25 --epochs 8 --out bitflip_resnet8.jsonl
  python bitflip_test.py --mode overhead --arch resnet8 --repeats 50     # time + memory cost of clamp / scrub / tmr
  python bitflip_test.py --analyze bitflip_inference.jsonl
  # YOUR OWN model (inference / overhead mode). Two ways to supply it:
  python bitflip_test.py --model-def mynet.py:build --weights mynet.pt --data-npz eval.npz --formats fp32,int8,int8pc --normalize-bits --ks 100
  python bitflip_test.py --model-file mynet_full.pt --data-npz eval.npz --formats int8,int8pc --normalize-bits --ks 100
      --model-def FILE.py:FUNC   your Python file; FUNC() must return the torch nn.Module (executes YOUR code - run only code you trust)
      --weights FILE.pt          state_dict for it (loaded with weights_only=True; "state_dict" key and "module." prefixes are handled)
      --model-file FILE.pt       alternative: a TorchScript file, or a fully pickled nn.Module (a pickle can execute code: trust the file)
      --data-npz FILE.npz        arrays x (N, ...) and y (N,) = your evaluation set, ALREADY preprocessed exactly as your model expects
  Notes for custom models: float32 compute; fault domain = all float parameters and float buffers; int8pc uses dim 0 of every
  weight tensor with >1 dims as the output-channel axis (true for Conv2d/Linear, NOT for ConvTranspose); already-quantized
  integer tensors inside the model are not part of the fault domain; one model = one run (no seed spread), so the intervals
  describe flip-to-flip noise only. Use --eval-n to limit evaluation time on big models.
  python bitflip_test.py --mode train --seeds 3 --epochs 6 --rates 0.005,0.02 --flip-state     # slow
"""
import argparse, copy, json, math, os, random, sys, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

np.seterr(all="ignore"); warnings.filterwarnings("ignore")
U32 = np.uint32
FMTS = ("fp32", "fp16", "bf16", "int8", "int8pc")
EXP_MSB = {"fp32": 30, "fp16": 14, "bf16": 14, "int8": 7, "int8pc": 7}
TNAMES = ["conv1.w", "conv1.b", "conv2.w", "conv2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b"]   # SmallCNN parameter order


# ----------------------------------------------------------------- basics
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
                                        nn.Dropout(0.25), nn.Linear(256, 10))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y))
        return torch.relu(y + (x if self.sc is None else self.sc(x)))


class ResNet8(nn.Module):
    """Small CIFAR ResNet (3 basic blocks, widths 16/32/64, ~78k parameters) WITH BatchNorm."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layers = nn.Sequential(BasicBlock(16, 16, 1), BasicBlock(16, 32, 2), BasicBlock(32, 64, 2))
        self.fc = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(self.layers(self.stem(x)), 1), 1))


MODELS = {"smallcnn": SmallCNN, "resnet8": ResNet8}


def get_loaders(fake=False, bs=128):
    from torch.utils.data import DataLoader, TensorDataset, Subset
    if fake:
        g = torch.Generator().manual_seed(0)
        def ds(n): return TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
        return (DataLoader(ds(1024), batch_size=bs, shuffle=True),
                DataLoader(ds(256), batch_size=256), DataLoader(ds(512), batch_size=256))
    import torchvision, torchvision.transforms as T
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    t_tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), T.Normalize(mean, std)])
    t_ev = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full_tr = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=t_tr)
    full_ev = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=t_ev)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=t_ev)
    idx = np.random.default_rng(0).permutation(len(full_tr)); val_idx, tr_idx = idx[:5000], idx[5000:]
    return (DataLoader(Subset(full_tr, tr_idx.tolist()), batch_size=bs, shuffle=True),
            DataLoader(Subset(full_ev, val_idx.tolist()), batch_size=256),
            DataLoader(test, batch_size=256))


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); c = n = 0
    for x, y in loader:
        c += model(x).argmax(1).eq(y).sum().item(); n += y.size(0)
    return 100.0 * c / n


def load_subset(loader, n, seed=0):
    """Fixed RANDOM subset of the test set (same subset for every run), not simply the first n images."""
    X = torch.cat([x for x, _ in loader]); Y = torch.cat([y for _, y in loader])
    idx = np.random.default_rng(seed).permutation(len(Y))[:n]
    return X[idx], Y[idx]


@torch.no_grad()
def acc_subset(model, X, Y, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += (model(X[i:i + bs]).argmax(1) == Y[i:i + bs]).sum().item()
    return 100.0 * c / len(X)


def state_tensors(model):
    """Fault domain: all parameters, then all float buffers (BatchNorm running stats). Same order as parameters() for
    models without buffers, so earlier results stay comparable."""
    return list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()]


def state_names(model):
    return [n for n, _ in model.named_parameters()] + [n for n, b in model.named_buffers() if b.is_floating_point()]


def get_vec(model):
    return np.concatenate([t.detach().cpu().numpy().astype(np.float32).reshape(-1) for t in state_tensors(model)]).copy()


@torch.no_grad()
def set_vec(model, vec):
    v = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32).copy()); o = 0
    for t in state_tensors(model):
        n = t.numel(); t.copy_(v[o:o + n].view_as(t)); o += n


def same_f32(a, b): return np.array_equal(a.view(U32), b.view(U32))      # bit-exact, NaN-safe


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


# ----------------------------------------------------------------- memory image in a real storage format
class Image:
    """A model's weights as stored words. arrs[0] = weights; (int8/int8pc only) arrs[1] = float32 scales (as uint32).
    chans (int8pc only): number of scales (= output channels) of each parameter tensor; soffs = their offsets."""
    def __init__(self, fmt, arrs, sizes, offs, chans=None):
        self.fmt, self.arrs, self.sizes, self.offs = fmt, arrs, list(sizes), offs
        self.chans = None if chans is None else [int(c) for c in chans]
        self.soffs = None if chans is None else np.concatenate([[0], np.cumsum(self.chans)]).astype(np.int64)

    @staticmethod
    def build(fmt, vec, sizes, chans=None):
        assert fmt in FMTS, fmt
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        v = np.ascontiguousarray(vec, dtype=np.float32)
        if fmt == "fp32":
            arrs = [v.copy().view(U32)]
        elif fmt == "fp16":
            arrs = [v.astype(np.float16).view(np.uint16).copy()]
        elif fmt == "bf16":
            u = v.view(U32).copy()
            r = u + U32(0x7FFF) + ((u >> U32(16)) & U32(1))
            arrs = [(r >> U32(16)).astype(np.uint16)]
        elif fmt == "int8":  # per-tensor symmetric
            scales = np.array([max(float(np.abs(v[o:o + s]).max()) / 127.0, 1e-12)
                               for o, s in zip(offs[:-1], sizes)], dtype=np.float32)
            q = np.empty(len(v), dtype=np.int8)
            for i, (o, s) in enumerate(zip(offs[:-1], sizes)):
                q[o:o + s] = np.clip(np.rint(v[o:o + s] / scales[i]), -127, 127).astype(np.int8)
            arrs = [q.view(np.uint8).copy(), scales.view(U32).copy()]
        else:  # int8pc: one symmetric scale per output channel (row of the [out, ...] tensor)
            assert chans is not None and len(chans) == len(sizes), "int8pc needs a channel count per parameter tensor"
            q = np.empty(len(v), dtype=np.int8); sc = []
            for (o, s), c in zip(zip(offs[:-1], sizes), chans):
                assert s % c == 0, (s, c)
                w = v[o:o + s].reshape(c, s // c)
                sci = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-12).astype(np.float32)
                q[o:o + s] = np.clip(np.rint(w / sci[:, None]), -127, 127).astype(np.int8).reshape(-1)
                sc.append(sci)
            arrs = [q.view(np.uint8).copy(), np.concatenate(sc).astype(np.float32).view(U32).copy()]
        return Image(fmt, arrs, sizes, offs, chans if fmt == "int8pc" else None)

    def copy(self): return Image(self.fmt, [a.copy() for a in self.arrs], self.sizes, self.offs, self.chans)

    def decode(self):
        a = self.arrs[0]
        if self.fmt == "fp32": return a.view(np.float32).copy()
        if self.fmt == "fp16": return a.view(np.float16).astype(np.float32)
        if self.fmt == "bf16": return (a.astype(U32) << U32(16)).view(np.float32).copy()
        q = a.view(np.int8).astype(np.float32); sc = self.arrs[1].view(np.float32)
        out = np.empty_like(q)
        if self.fmt == "int8pc":
            for i, (o, s) in enumerate(zip(self.offs[:-1], self.sizes)):
                c = self.chans[i]
                out[o:o + s] = (q[o:o + s].reshape(c, s // c) * sc[self.soffs[i]:self.soffs[i + 1], None]).reshape(-1)
            return out
        for i, (o, s) in enumerate(zip(self.offs[:-1], self.sizes)): out[o:o + s] = q[o:o + s] * sc[i]
        return out

    def equals(self, other): return all(np.array_equal(x, y) for x, y in zip(self.arrs, other.arrs))


def sample_flips(img, k, rng, allowed=None):
    """k flips: array ~ number of allowed bits, element uniform, bit uniform among allowed bits of that width."""
    cand = []
    for a in img.arrs:
        w = a.dtype.itemsize * 8
        cand.append(np.array([b for b in range(w) if allowed is None or b in allowed], dtype=np.int64))
    wt = np.array([len(a) * len(c) for a, c in zip(img.arrs, cand)], dtype=np.float64)
    if wt.sum() == 0 or k == 0: return []
    ai = rng.choice(len(img.arrs), size=k, p=wt / wt.sum())
    out = []
    for i in range(len(img.arrs)):
        c = int((ai == i).sum())
        if c == 0: continue
        el = rng.integers(0, len(img.arrs[i]), size=c)
        bt = cand[i][rng.integers(0, len(cand[i]), size=c)]
        out.append((i, el, bt))
    return out


def apply_flips(img, flips):
    for ai, el, bt in flips:
        arr = img.arrs[ai]
        np.bitwise_xor.at(arr, el, arr.dtype.type(1) << bt.astype(arr.dtype))


def flip_one(img, ai, el, bit):
    apply_flips(img, [(ai, np.array([el]), np.array([bit]))])


# ----------------------------------------------------------------- protections
class Guards:
    """clamp: operates on the decoded float32 weights. bounds from the clean decoded image."""
    def __init__(self, dec0, sizes, margin=1.1):
        self.offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64); self.sizes = list(sizes)
        self.bounds = [margin * float(np.abs(dec0[o:o + s]).max()) for o, s in zip(self.offs[:-1], sizes)]

    def clamp(self, v):
        out = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for (o, s), b in zip(zip(self.offs[:-1], self.sizes), self.bounds):
            np.clip(out[o:o + s], -b, b, out=out[o:o + s])
        return out


def _wsum(words):
    """Position-weighted checksum (mod 2^64). Unlike a plain sum, two flips of the same bit in opposite
    directions at different positions do NOT cancel."""
    w = words.astype(np.uint64)
    return int((w * (np.arange(len(w), dtype=np.uint64) + np.uint64(1))).sum(dtype=np.uint64))


class Scrub:
    def __init__(self, img0):
        self.golden = img0.copy()
        self.segs = []                                   # (array index, lo, hi)
        for o, s in zip(img0.offs[:-1], img0.sizes): self.segs.append((0, int(o), int(o + s)))
        if len(img0.arrs) > 1:
            for i in range(len(img0.arrs[1])): self.segs.append((1, i, i + 1))
        self.sums = [_wsum(img0.arrs[ai][lo:hi]) for ai, lo, hi in self.segs]

    def repair(self, img):
        out = img.copy(); n = 0
        for (ai, lo, hi), c in zip(self.segs, self.sums):
            if _wsum(img.arrs[ai][lo:hi]) != c:
                out.arrs[ai][lo:hi] = self.golden.arrs[ai][lo:hi]; n += 1
        return out, n


class ScrubScales:
    """Protects ONLY the float32 scale array of an int8 / int8pc image: golden copy + one vectorized checksum."""
    def __init__(self, img0):
        self.golden = img0.arrs[1].copy(); self.sum = _wsum(self.golden)

    def repair(self, img):
        out = img.copy(); n = 0
        if _wsum(img.arrs[1]) != self.sum: out.arrs[1][:] = self.golden; n = 1
        return out, n


def tmr_vote(c0, c1, c2):
    arrs = []
    for a, b, c in zip(c0.arrs, c1.arrs, c2.arrs): arrs.append((a & b) | (a & c) | (b & c))
    return Image(c0.fmt, arrs, c0.sizes, c0.offs, c0.chans)


# ----------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    sizes = [400, 600]
    v = rng.normal(0, 0.1, 1000).astype(np.float32); v[5] = np.float32(0.1)
    # ---- fp32 bit semantics
    img = Image.build("fp32", v, sizes); flip_one(img, 0, 5, 30)
    assert abs(float(img.decode()[5])) > 1e30, "bit-30 flip of 0.1 must be huge"
    img = Image.build("fp32", v, sizes); flip_one(img, 0, 5, 0)
    assert img.decode()[5] != v[5] and abs(float(img.decode()[5]) - 0.1) < 1e-6
    img = Image.build("fp32", v, sizes); flip_one(img, 0, 5, 31); assert img.decode()[5] == -v[5]
    img = Image.build("fp32", v, sizes); flip_one(img, 0, 5, 30); flip_one(img, 0, 5, 30)
    assert same_f32(img.decode(), v), "double flip restores"
    w = np.array([1.5], dtype=np.float32); im = Image.build("fp32", w, [1]); flip_one(im, 0, 0, 30)
    assert not np.isfinite(im.decode()[0]) or abs(float(im.decode()[0])) > 1e30   # exponent 127 -> 255 => Inf/NaN
    w = np.array([4.0], dtype=np.float32); im = Image.build("fp32", w, [1]); flip_one(im, 0, 0, 30)
    assert abs(float(im.decode()[0])) < 1e-30, "|w|>=2: bit 30 is 1 -> flipping it gives ~0"
    # ---- round trips in each format
    for fmt, tol in (("fp32", 0.0), ("fp16", 2e-3), ("bf16", 1e-2), ("int8", None)):
        im = Image.build(fmt, v, sizes); d = im.decode()
        if tol is None:
            sc = im.arrs[1].view(np.float32)
            assert abs(d[:400] - v[:400]).max() <= sc[0] * 0.5 + 1e-7 and abs(d[400:] - v[400:]).max() <= sc[1] * 0.5 + 1e-7
        else:
            assert np.allclose(d, v, rtol=tol, atol=1e-6), fmt
    # ---- fp16 / bf16 exponent MSB (bit 14) of 0.1 blows up
    for fmt in ("fp16", "bf16"):
        im = Image.build(fmt, v, sizes); flip_one(im, 0, 5, 14)
        assert abs(float(im.decode()[5])) > 1e3, fmt
    # ---- int8: MSB flip bounded by ~127*scale; scale flips are in the fault domain
    im = Image.build("int8", v, sizes); sc = im.arrs[1].view(np.float32).copy(); flip_one(im, 0, 5, 7)
    assert abs(float(im.decode()[5])) <= 1.01 * 127 * sc[0] + 1e-6
    im = Image.build("int8", v, sizes); flip_one(im, 1, 0, 30)
    assert abs(im.decode()[:400]).max() > 1e20, "scale exponent flip must blow up the whole tensor"
    # ---- int8pc: per-channel scales; a scale flip must damage exactly ONE channel
    im = Image.build("int8pc", v, [400, 600], [4, 1]); sc = im.arrs[1].view(np.float32); d = im.decode()
    assert len(sc) == 5 and im.soffs.tolist() == [0, 4, 5]
    for c in range(4):
        seg = slice(c * 100, (c + 1) * 100); assert abs(d[seg] - v[seg]).max() <= sc[c] * 0.5 + 1e-7
    assert abs(d[400:] - v[400:]).max() <= sc[4] * 0.5 + 1e-7
    flip_one(im, 1, 2, 30); d2 = im.decode()
    assert abs(d2[200:300]).max() > 1e20 and abs(d2[:200]).max() < 1 and abs(d2[300:]).max() < 1, "pc scale flip hits ONE channel only"
    assert Image.build("int8pc", v, [400, 600], [1, 1]).decode().shape == v.shape
    # ---- clamp neutralizes non-finite / out-of-range values
    CH = [4, 2]
    for fmt in FMTS:
        im0 = Image.build(fmt, v, sizes, CH); G = Guards(im0.decode(), sizes)
        im = im0.copy(); flip_one(im, 0, 7, EXP_MSB[fmt]); c = G.clamp(im.decode())
        assert np.all(np.isfinite(c)) and abs(c[:400]).max() <= G.bounds[0] + 1e-5
    # ---- scrub restores exactly; weighted checksum catches two same-bit opposite-direction flips
    for fmt in FMTS:
        im0 = Image.build(fmt, v, sizes, CH); S = Scrub(im0)
        im = im0.copy(); apply_flips(im, sample_flips(im, 20, np.random.default_rng(1)))
        rep, n = S.repair(im); assert rep.equals(im0), fmt
    a = np.zeros(1000, dtype=np.uint32); a[8] = np.uint32(1 << 5)               # bit5=0 at idx 3, bit5=1 at idx 8
    plain0 = int(a.sum(dtype=np.uint64)); w0 = _wsum(a)
    a2 = a.copy(); a2[3] ^= np.uint32(1 << 5); a2[8] ^= np.uint32(1 << 5)
    assert int(a2.sum(dtype=np.uint64)) == plain0, "plain sum is blind to this pair"
    assert _wsum(a2) != w0, "weighted checksum must detect it"
    # ---- scrub_scales restores a scale flip, ignores weight flips
    for fmt in ("int8", "int8pc"):
        im0 = Image.build(fmt, v, sizes, [4, 2]); SS = ScrubScales(im0)
        im = im0.copy(); flip_one(im, 1, 1, 30); rep_, n = SS.repair(im)
        assert n == 1 and rep_.equals(im0), fmt
        im = im0.copy(); flip_one(im, 0, 5, 3); rep_, n = SS.repair(im)
        assert n == 0 and not rep_.equals(im0), "weights are NOT protected by scrub_scales"
    # ---- TMR
    im0 = Image.build("fp32", v, sizes); b = im0.copy(); flip_one(b, 0, 123, 29)
    assert tmr_vote(im0, b, im0.copy()).equals(im0)
    d = im0.copy(); flip_one(d, 0, 123, 29)
    assert not tmr_vote(im0, b, d).equals(im0), "same bit in two copies defeats TMR (documented)"
    for fmt in ("fp16", "int8"):
        im0 = Image.build(fmt, v, sizes); b = im0.copy(); flip_one(b, 0, 9, 3)
        assert tmr_vote(im0, b, im0.copy()).equals(im0)
    # ---- restricted bit sampling
    im0 = Image.build("fp32", v, sizes)
    fl = sample_flips(im0, 500, np.random.default_rng(2), allowed=set(range(32)) - {30})
    assert all((bt != 30).all() for _, _, bt in fl)
    fl = sample_flips(im0, 500, np.random.default_rng(2), allowed={30})
    assert all((bt == 30).all() for _, _, bt in fl)
    # ---- model <-> vector round trip
    m = SmallCNN(); vec = get_vec(m); m2 = SmallCNN(); set_vec(m2, vec); assert same_f32(get_vec(m2), vec)
    # ---- ResNet8: BatchNorm running statistics are part of the fault domain and survive a round trip
    r1 = ResNet8(); r1.train(); r1(torch.randn(8, 3, 32, 32)); r1.eval()
    n_par = sum(p.numel() for p in r1.parameters()); n_buf = sum(b.numel() for b in r1.buffers() if b.is_floating_point())
    vr = get_vec(r1); assert len(vr) == n_par + n_buf and n_buf > 0
    r2 = ResNet8(); set_vec(r2, vr); r2.eval(); assert same_f32(get_vec(r2), vr)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad(): assert torch.allclose(r1(x), r2(x), atol=1e-6), "ResNet8 output must match after set_vec"
    assert len(state_names(r1)) == len(state_tensors(r1))
    print("selftest OK")


# ----------------------------------------------------------------- clean model
def train_clean(seed, epochs, loaders, log, arch="smallcnn"):
    tr, va, te = loaders
    set_seed(seed); model = MODELS[arch](); opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad(set_to_none=True); crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate(model, va)
        if v > best: best, best_state = v, copy.deepcopy(model.state_dict())
        log(f"      clean training epoch {ep+1}/{epochs} val={v:.2f}")
    model.load_state_dict(best_state)
    return model


def load_custom(a, log):
    """Build the user's own model (see Usage). Returns (model, name). Runs user-supplied code by design."""
    import importlib.util
    if a.model_def:
        path, _, fn = a.model_def.partition(":"); fn = fn or "build"
        spec = importlib.util.spec_from_file_location("user_model_def", path); mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod); model = getattr(mod, fn)()
        assert isinstance(model, nn.Module), f"{fn}() must return an nn.Module"
        if a.weights:
            sd = torch.load(a.weights, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict): sd = sd["state_dict"]
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            model.load_state_dict(sd)
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        try: model = torch.jit.load(a.model_file, map_location="cpu")
        except Exception: model = torch.load(a.model_file, map_location="cpu", weights_only=False)
        assert isinstance(model, nn.Module), "--model-file must hold a TorchScript module or a pickled nn.Module"
        name = os.path.splitext(os.path.basename(a.model_file))[0]
    model = model.float().eval()
    n_tens = len(state_tensors(model))
    log(f"    custom model '{name}': {sum(t.numel() for t in state_tensors(model))} stored float values in {n_tens} tensors")
    return model, name


def custom_loaders(a):
    from torch.utils.data import DataLoader, TensorDataset
    d = np.load(a.data_npz); assert "x" in d and "y" in d, "--data-npz needs arrays named x and y"
    x = torch.from_numpy(np.ascontiguousarray(d["x"]).astype(np.float32)); y = torch.from_numpy(np.ascontiguousarray(d["y"]).astype(np.int64))
    assert len(x) == len(y) and len(x) >= 100, "need at least 100 evaluation samples with matching x and y"
    return (None, None, DataLoader(TensorDataset(x, y), batch_size=256))


def get_clean_model(seed, a, loaders, log):
    if getattr(a, "custom", False): return copy.deepcopy(a._model)
    path = f"bitflip_model_s{seed}_e{a.epochs}{'' if a.arch == 'smallcnn' else '_' + a.arch}{'_fake' if a.fake_data else ''}.pt"
    if os.path.exists(path):
        m = MODELS[a.arch](); m.load_state_dict(torch.load(path)); log(f"      loaded cached clean model {path}"); return m
    m = train_clean(seed, a.epochs, loaders, log, a.arch)
    torch.save(m.state_dict(), path)
    return m


# ----------------------------------------------------------------- INFERENCE experiment
def bits_label(a):
    if a.only_bits: return "only:" + ",".join(map(str, sorted(a.only_bits)))
    if a.exclude_bits: return "excl:" + ",".join(map(str, sorted(a.exclude_bits)))
    return "all"


def allowed_bits(a):
    if a.only_bits: return set(a.only_bits)
    if a.exclude_bits: return set(range(64)) - set(a.exclude_bits)
    return None


def run_inference(seed, a, loaders, fout, log):
    model = get_clean_model(seed, a, loaders, log)
    vec32 = get_vec(model); st = state_tensors(model); tnames = state_names(model)
    sizes = [t.numel() for t in st]
    chans = [t.shape[0] if t.dim() > 1 else 1 for t in st]       # output channels (1 for every 1-D tensor)
    X, Y = load_subset(loaders[2], a.eval_n)
    clean_full = evaluate(model, loaders[2])
    log(f"    [{a.arch}] fp32 clean: full test {clean_full:.2f}%  stored values {len(vec32)} in {len(sizes)} tensors")
    allowed = allowed_bits(a); label = bits_label(a)

    def ev(vec):
        set_vec(model, vec); return acc_subset(model, X, Y)

    for fi, fmt in enumerate(a.formats):
        img0 = Image.build(fmt, vec32, sizes, chans); dec0 = img0.decode()
        clean_sub = ev(dec0)
        G = Guards(dec0, sizes); S = Scrub(img0); SS = ScrubScales(img0) if len(img0.arrs) > 1 else None
        rng = np.random.default_rng(10_000 + 97 * seed + fi)
        base = dict(mode="inference", seed=seed, fmt=fmt, bits=label, clean=clean_sub, arch=a.arch)
        if getattr(a, "custom", False): base.update(getattr(a, "_hashes", {}))
        if len(img0.arrs) > 1:      # lets --analyze compute the share of stored bits that are scales
            base.update(n_scales=int(len(img0.arrs[1])), stored_bits=int(sum(len(x) * x.dtype.itemsize * 8 for x in img0.arrs)))

        def put(**kw): fout.write(json.dumps({**base, **kw}) + "\n")

        log(f"    [{fmt}] clean accuracy on eval-subset ({len(X)}): {clean_sub:.2f}%  (fp32 full test {clean_full:.2f}%)")
        put(kind="clean_fmt", clean_fp32_full=clean_full, tnames=tnames)

        if not a.skip_bit_sweep:
            t0 = time.time(); width = img0.arrs[0].dtype.itemsize * 8
            for b in range(width):
                for _ in range(a.bit_trials):
                    im = img0.copy(); flip_one(im, 0, int(rng.integers(len(im.arrs[0]))), b)
                    put(kind="bit", bit=b, width=width, acc=ev(im.decode()))
            if len(img0.arrs) > 1:      # int8: the float32 quantization scales are also stored bits
                # single flip in a scale; protections evaluated on the SAME flip.
                # (tmr is omitted: one flip lands in one of three copies and is always out-voted.)
                for b in range(32):
                    for _ in range(a.bit_trials):
                        im = img0.copy(); sidx = int(rng.integers(len(im.arrs[1]))); flip_one(im, 1, sidx, b)
                        ten = int(np.searchsorted(img0.soffs, sidx, side="right") - 1) if img0.soffs is not None else sidx
                        dv = im.decode(); a_none = ev(dv)
                        dc = G.clamp(dv)
                        a_clamp = a_none if same_f32(dc, dv) else (clean_sub if same_f32(dc, dec0) else ev(dc))
                        rep, _n = S.repair(im)
                        a_scrub = clean_sub if rep.equals(img0) else ev(rep.decode())
                        for m, acc in (("none", a_none), ("clamp", a_clamp), ("scrub", a_scrub)):
                            put(kind="bit_scale", bit=b, width=32, method=m, acc=acc, sidx=sidx, ten=ten)
            fout.flush(); log(f"      bit sweep done ({time.time()-t0:.0f}s)")

        bits_fmt = sum(len(x) * x.dtype.itemsize * 8 for x in img0.arrs); bits_ref = 32 * len(vec32)
        for k_ref in a.ks:
            # --normalize-bits: k_ref is the number of flips an fp32 image would receive; other formats get the
            # same flips PER STORED BIT (so a smaller image gets proportionally fewer flips).
            k = max(1, int(round(k_ref * bits_fmt / bits_ref))) if a.normalize_bits else k_ref
            t0 = time.time()
            for t in range(a.trials):
                im = img0.copy(); apply_flips(im, sample_flips(im, k, rng, allowed))
                dv = im.decode(); a_none = ev(dv)
                dc = G.clamp(dv)
                a_clamp = a_none if same_f32(dc, dv) else (clean_sub if same_f32(dc, dec0) else ev(dc))
                rep, nrep = S.repair(im)
                a_scrub = clean_sub if rep.equals(img0) else ev(rep.decode())
                extra = []
                if SS is not None:
                    repS, _n = SS.repair(im); dS = repS.decode()
                    a_sS = a_none if same_f32(dS, dv) else (clean_sub if same_f32(dS, dec0) else ev(dS))
                    dSc = G.clamp(dS)
                    a_sSc = a_sS if same_f32(dSc, dS) else (clean_sub if same_f32(dSc, dec0) else ev(dSc))
                    extra = [("scrub_scales", a_sS), ("scrub_scales+clamp", a_sSc)]
                cp = [img0.copy() for _ in range(3)]
                for ai, el, bt in sample_flips(img0, 3 * k, rng, allowed):
                    cidx = rng.integers(0, 3, size=len(el))
                    for ci in range(3):
                        sel = cidx == ci
                        if sel.any(): apply_flips(cp[ci], [(ai, el[sel], bt[sel])])
                vt = tmr_vote(*cp)
                a_tmr = clean_sub if vt.equals(img0) else ev(vt.decode())
                for m, acc in [("none", a_none), ("clamp", a_clamp), ("scrub", a_scrub), ("tmr", a_tmr)] + extra:
                    put(kind="rand", k=k_ref, k_eff=k, trial=t, method=m, acc=acc)
            fout.flush(); log(f"      k={k_ref} (flips/image used: {k}): {a.trials} trials ({time.time()-t0:.0f}s)")
    fout.write(json.dumps(dict(mode="inference", kind="seed_done", seed=seed, epochs=a.epochs,
                               fmts=list(a.formats), bits=label, sweep=not a.skip_bit_sweep,
                               norm=bool(a.normalize_bits), ks=list(a.ks), arch=a.arch)) + "\n"); fout.flush()


# ----------------------------------------------------------------- OVERHEAD (cost of the protections)
def _time(fn, repeats, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[min(len(ts) - 1, int(0.9 * len(ts)))]          # median, p90 (ms)


def run_overhead(a, loaders, fout, log):
    """Time and memory cost of each protection, measured with this script's NumPy implementation on THIS machine.
    Not a prediction for any target hardware: a C / NEON / DMA implementation will differ, only ratios are indicative."""
    model = get_clean_model(42, a, loaders, log)
    vec32 = get_vec(model); st = state_tensors(model)
    sizes = [t.numel() for t in st]; chans = [t.shape[0] if t.dim() > 1 else 1 for t in st]
    set_vec(model, vec32); model.eval()
    if getattr(a, "custom", False):
        xs = loaders[2].dataset.tensors[0]; x1 = xs[:1]; x32 = xs[:32]
    else:
        x1 = torch.randn(1, 3, 32, 32); x32 = torch.randn(32, 3, 32, 32)
    with torch.no_grad():
        f1 = _time(lambda: model(x1), a.repeats)[0]; f32 = _time(lambda: model(x32), a.repeats)[0]
    rng = np.random.default_rng(1); rows = []
    for fmt in a.formats:
        img0 = Image.build(fmt, vec32, sizes, chans); dec0 = img0.decode()
        G = Guards(dec0, sizes); S = Scrub(img0)
        bad = img0.copy(); apply_flips(bad, sample_flips(bad, 1, rng)); c1, c2 = img0.copy(), img0.copy()
        nbytes = int(sum(x.nbytes for x in img0.arrs))
        ops = [("decode (image -> float32)", lambda: img0.decode(), 0),
               ("clamp pass (after decode)", lambda: G.clamp(dec0), 4 * len(sizes)),
               ("scrub: checksum pass, no fault", lambda: [_wsum(img0.arrs[ai][lo:hi]) for ai, lo, hi in S.segs], nbytes + 8 * len(S.segs)),
               ("scrub: check + repair, 1 flip (incl. image copy)", lambda: S.repair(bad), nbytes + 8 * len(S.segs)),
               ("tmr: bitwise vote of 3 copies", lambda: tmr_vote(img0, c1, c2), 2 * nbytes)]
        if len(img0.arrs) > 1:
            SS = ScrubScales(img0); sc_bad = img0.arrs[1].copy(); sc_bad[min(3, len(sc_bad) - 1)] ^= np.uint32(1 << 30)

            def _fix():
                t = sc_bad.copy()
                if _wsum(t) != SS.sum: t[:] = SS.golden
            ops += [("scrub-scales: checksum pass (scales only)", lambda: _wsum(img0.arrs[1]), SS.golden.nbytes + 8),
                    ("scrub-scales: check + restore, 1 scale flip", _fix, SS.golden.nbytes + 8)]
        for name, fn, extra in ops:
            med, p90 = _time(fn, a.repeats)
            rows.append(dict(mode="overhead", arch=a.arch, fmt=fmt, op=name, median_ms=med, p90_ms=p90,
                             extra_bytes=int(extra), image_bytes=nbytes, fwd_b1_ms=f1, fwd_b32_ms=f32, repeats=a.repeats))
    for r in rows: fout.write(json.dumps(r) + "\n")
    fout.flush(); analyze_overhead(rows)


def analyze_overhead(rows):
    r0 = rows[0]
    print(f"\nPROTECTION COST   model {r0.get('arch', 'smallcnn')}   repeats {r0['repeats']}   "
          f"forward pass: batch 1 = {r0['fwd_b1_ms']:.3f} ms, batch 32 = {r0['fwd_b32_ms']:.3f} ms (this machine, PyTorch CPU)")
    print("NumPy implementation of this script on THIS machine; not a prediction for target hardware (ratios are indicative only).")
    for fmt in dict.fromkeys(r["fmt"] for r in rows):
        rr = [r for r in rows if r["fmt"] == fmt]
        print(f"\n[{fmt}]  stored image: {rr[0]['image_bytes']/1024:.1f} KB")
        print(f"  {'operation':<52} {'median ms':>10} {'p90 ms':>9} {'x fwd(b=1)':>11} {'extra memory':>14}")
        for r in rr:
            ex = r["extra_bytes"]; exs = "-" if ex == 0 else (f"{ex/1024:.1f} KB ({100*ex/r['image_bytes']:.1f}%)")
            print(f"  {r['op']:<52} {r['median_ms']:>10.3f} {r['p90_ms']:>9.3f} {r['median_ms']/r['fwd_b1_ms']:>10.1f}x {exs:>14}")
    print("\nNotes: the checksum pass and the clamp pass are recurring costs; repair is a rare event. decode is needed by every\n"
          "format except fp32 anyway. clamp needs only per-tensor bounds (4 bytes per tensor) and no extra copy.")


# ----------------------------------------------------------------- TRAIN experiment (fp32 only)
def pool_of(model, opt, flip_state):
    ts = [p.data for p in model.parameters()]
    if flip_state:
        for p in model.parameters():
            st = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st: ts.append(st[key])
    return ts


def tensor_wsum(t): return _wsum(t.numpy().reshape(-1).view(U32))


def run_train(guard, rate, seed, epochs, loaders, flip_state, ckpt_every, log):
    tr, va, te = loaders
    set_seed(seed); rng = np.random.default_rng(777 + seed)
    model = SmallCNN(); opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    step = flips = rollbacks = repairs = 0; ema = None
    snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
    val_hist, test_hist = [], []; t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            step += 1
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); lv = loss.item()
            if guard == "rollback" and step > 20 and (not math.isfinite(lv) or lv > 3 * ema + 1.0):
                model.load_state_dict(snap[0]); opt.load_state_dict(copy.deepcopy(snap[1])); rollbacks += 1
                continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if math.isfinite(lv): ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if guard == "rollback" and step % ckpt_every == 0 and math.isfinite(lv) and \
               all(torch.isfinite(p).all() for p in model.parameters()):
                snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
            if rate <= 0: continue
            pool = pool_of(model, opt, flip_state)
            if guard == "scrub":
                shadow = [t.clone() for t in pool]; cs = [tensor_wsum(t) for t in pool]
            if guard == "clamp":
                bnd = [1.5 * float(p.data.abs().max()) + 1e-3 for p in model.parameters()]
            sz = np.array([t.numel() for t in pool]); offs = np.concatenate([[0], np.cumsum(sz)])
            for _ in range(int(rng.poisson(rate))):
                g = int(rng.integers(offs[-1])); ti = int(np.searchsorted(offs, g, side="right") - 1)
                li = g - int(offs[ti]); b = int(rng.integers(32))
                pool[ti].numpy().reshape(-1).view(U32)[li] ^= U32(1 << b); flips += 1
            if guard == "scrub":
                for ti, t in enumerate(pool):
                    if tensor_wsum(t) != cs[ti]: t.copy_(shadow[ti]); repairs += 1
            if guard == "clamp":
                for p, b in zip(model.parameters(), bnd):
                    p.data.copy_(torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-b, b))
        val_hist.append(evaluate(model, va)); test_hist.append(evaluate(model, te))
        log(f"      ep{ep+1}/{epochs} val={val_hist[-1]:.2f} test={test_hist[-1]:.2f} flips={flips} rb={rollbacks} rep={repairs}")
    best = int(np.argmax(val_hist))
    return dict(mode="train", guard=guard, rate=rate, seed=seed, epochs=epochs, flip_state=flip_state,
                test_last=test_hist[-1], test_at_best_val=test_hist[best], flips=flips,
                rollbacks=rollbacks, repairs=repairs, minutes=(time.time() - t0) / 60)


# ----------------------------------------------------------------- analysis
def role(fmt, b):
    if fmt in ("int8", "int8pc"): return "MSB(sign)" if b == 7 else "magnitude"
    spec = {"fp32": (31, 30, 23, 16), "fp16": (15, 14, 10, 5), "bf16": (15, 14, 7, 3)}[fmt]
    s, em, elo, mh = spec
    return "sign" if b == s else "exp-MSB" if b == em else "exponent" if b >= elo else "mant-high" if b >= mh else "mant-low"


def analyze(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows: print("empty file"); return
    if rows[0].get("mode") == "train": return analyze_train(rows)
    if rows[0].get("mode") == "overhead": return analyze_overhead(rows)
    for r in rows:
        r.setdefault("fmt", "fp32"); r.setdefault("bits", "all"); r.setdefault("arch", "smallcnn")
    groups = sorted({(r["arch"], r["fmt"], r["bits"]) for r in rows if r.get("kind") in ("bit", "rand")})
    for arch, fmt, bits in groups:
        analyze_inference([r for r in rows if r["arch"] == arch and r["fmt"] == fmt and r["bits"] == bits], fmt, bits, arch)


def analyze_inference(rows, fmt, bits, arch="smallcnn"):
    print("\n" + "=" * 100)
    print(f"MODEL {arch}   FORMAT {fmt}   bit selection: {bits}")
    print("=" * 100)
    clean = {}
    for r in rows:
        if "clean" in r: clean[r["seed"]] = r["clean"]
    print("Seeds / clean accuracy of THIS format on the eval-subset: " + ", ".join(f"{s}:{c:.2f}" for s, c in sorted(clean.items())))
    bit = [r for r in rows if r.get("kind") == "bit"]
    if bit:
        print("\n--- single flip by bit position (drop = clean - accuracy after flip), pooled over seeds ---")
        print(f"{'bit':>4} {'role':<10} {'n':>5} {'mean drop':>10} {'P(drop>5pt)':>12} {'P(collapse<=15%)':>17}")
        for b in sorted({r["bit"] for r in bit}, reverse=True):
            rr = [r for r in bit if r["bit"] == b]; d = np.array([r["clean"] - r["acc"] for r in rr])
            print(f"{b:>4} {role(fmt, b):<10} {len(d):>5} {d.mean():>10.2f} {np.mean(d > 5):>12.3f} {np.mean([r['acc'] <= 15 for r in rr]):>17.3f}")
        em = EXP_MSB[fmt]
        print(f"\n--- exponent-MSB (bit {em}) by seed: is the vulnerability repeated in every model? ---")
        for s in sorted(clean):
            rr = [r for r in bit if r["bit"] == em and r["seed"] == s]
            if rr:
                d = np.array([r["clean"] - r["acc"] for r in rr])
                print(f"  seed {s}: n={len(d)}  mean drop {d.mean():6.2f}  P(drop>5pt) {np.mean(d > 5):.3f}")
    sbit = [r for r in rows if r.get("kind") == "bit_scale"]
    if sbit:
        pc = fmt == "int8pc"
        print("\n--- " + fmt + ": single flip in a float32 quantization SCALE, by bit position ("
              + ("one scale rescales ONE output channel" if pc else "one scale rescales a whole tensor") + ") ---")
        ns = next((r["n_scales"] for r in sbit if "n_scales" in r), None)
        sb = next((r["stored_bits"] for r in sbit if "stored_bits" in r), None)
        phit = 32 * ns / sb if ns and sb else None
        if phit is not None:
            print(f"    {ns} scales = {32*ns} of {sb} stored bits ({100*phit:.3f}%): a random flip hits a scale with probability "
                  f"{phit:.2e} (~{100*(1-(1-phit)**100):.2f}% over 100 flips).")
        else:
            print("    Only a few scales exist, so a random flip rarely hits one, but a hit can be severe.")
        print("    scrub restores from a fault-free golden copy (by construction); tmr omitted (a single flip is always out-voted).")
        print(f"{'bit':>4} {'role':<10} {'n':>4} | {'none: drop':>10} {'P>5':>6} {'collapse':>8} | {'clamp: drop':>11} {'P>5':>6} {'collapse':>8} | {'scrub P>5':>9}")
        def stat(rr):
            if not rr: return (float("nan"),) * 3
            d = np.array([r["clean"] - r["acc"] for r in rr])
            return d.mean(), np.mean(d > 5), np.mean([r["acc"] <= 15 for r in rr])
        for b in sorted({r["bit"] for r in sbit}, reverse=True):
            rb = [r for r in sbit if r["bit"] == b]
            sel = lambda m: [r for r in rb if r.get("method", "none") == m]
            n0, c0, s0 = stat(sel("none")), stat(sel("clamp")), stat(sel("scrub"))
            print(f"{b:>4} {role('fp32', b):<10} {len(sel('none')):>4} | {n0[0]:>10.2f} {n0[1]:>6.3f} {n0[2]:>8.3f} | "
                  f"{c0[0]:>11.2f} {c0[1]:>6.3f} {c0[2]:>8.3f} | {s0[1]:>9.3f}")
        if phit is not None:
            for m in ("none", "clamp"):
                pb = float(np.mean([stat([r for r in sbit if r["bit"] == b and r.get("method", "none") == m])[1]
                                    for b in sorted({r["bit"] for r in sbit})]))
                pf = phit * pb
                print(f"    summary [{m}]: P(drop>5) given a scale hit, uniform over the 32 bit positions = {pb:.3f}; "
                      f"per random flip anywhere = {pf:.2e}; over 100 flips ~ {100*(1-(1-pf)**100):.2f}%  "
                      f"(analytic combination: uniform flips, effects independent)")
            print(f"    cost of protecting ONLY the scales with a golden copy / extra copies: {4*ns} bytes per copy "
                  f"= {100*32*ns/sb:.3f}% of the {sb//8}-byte image")
        if any("ten" in r for r in sbit):
            hi = [r for r in sbit if r["bit"] >= 23 and r.get("method", "none") == "none" and "ten" in r]
            tn = next((r["tnames"] for r in rows if r.get("tnames")), TNAMES)
            print("    by stored tensor, 'none', pooled over scale bits 23-31 (sign + exponent):")
            for t in sorted({r["ten"] for r in hi}):
                rr = [r for r in hi if r["ten"] == t]; d = np.array([r["clean"] - r["acc"] for r in rr])
                nm = tn[t] if t < len(tn) else str(t)
                print(f"      {nm:<26} n={len(d):>4}  mean drop {d.mean():6.2f}  P(drop>5) {np.mean(d > 5):.3f}")
    rnd = [r for r in rows if r.get("kind") == "rand"]
    if rnd:
        keff = {k: np.mean([r.get("k_eff", r["k"]) for r in rnd if r["k"] == k]) for k in {r["k"] for r in rnd}}
        print("\n--- random flips, pooled over seeds (k = reference flips per image; tmr is exposed to 3x the flips) ---")
        print("    flips/image actually used: " + ", ".join(f"k={k}->{keff[k]:.0f}" for k in sorted(keff)))
        print(f"{'k':>4} {'method':<19} {'extra copies':>12} {'n':>5} {'mean acc':>9} {'mean drop':>10} {'P(drop>1)':>10} {'P(drop>5)':>10} {'95% CI P(drop>5)':>18} {'P(collapse)':>12}")
        mem = {"none": "0", "clamp": "0 (+clip)", "scrub": "1", "tmr": "2", "scrub_scales": "scales only", "scrub_scales+clamp": "scales only"}
        for k in sorted({r["k"] for r in rnd}):
            for m in ("none", "clamp", "scrub", "tmr", "scrub_scales", "scrub_scales+clamp"):
                rr = [r for r in rnd if r["k"] == k and r["method"] == m]
                if not rr: continue
                d = np.array([r["clean"] - r["acc"] for r in rr]); n = len(d)
                big = int((d > 5).sum()); lo, hi = wilson(big, n)
                print(f"{k:>4} {m:<19} {mem[m]:>12} {n:>5} {np.mean([r['acc'] for r in rr]):>9.2f} {d.mean():>10.2f} "
                      f"{np.mean(d > 1):>10.3f} {big/n:>10.3f} [{lo:6.3f},{hi:6.3f}] {np.mean([r['acc'] <= 15 for r in rr]):>12.3f}")
            print()
        print("--- paired by trial (every method sees the SAME flips): events 'drop > 5pt' that a method removes / adds; exact two-sided sign test ---")
        for k in sorted({r["k"] for r in rnd}):
            by = {}
            for r in rnd:
                if r["k"] == k: by.setdefault(r["method"], {})[(r["seed"], r["trial"])] = (r["clean"] - r["acc"]) > 5
            for m in ("clamp", "scrub_scales"):
                if m not in by or "none" not in by: continue
                keys = by["none"].keys() & by[m].keys()
                b = sum(1 for q in keys if by["none"][q] and not by[m][q]); c = sum(1 for q in keys if not by["none"][q] and by[m][q])
                nd = b + c
                pv = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, min(b, c) + 1)) / 2 ** nd)
                print(f"  k={k}  none -> {m:<13} removed {b:>3}, added {c:>3} of {len(keys)} trials   exact p = {pv:.3f}"
                      f"{'' if pv < 0.05 else '   (not distinguishable from chance at this n)'}")
        print()
        ms = [m for m in ("none", "clamp", "scrub_scales") if any(r["method"] == m for r in rnd)]
        print("--- by seed (one trained model each): P(drop>5pt), methods " + " / ".join(ms) + " ---")
        for k in sorted({r["k"] for r in rnd}):
            parts = []; per = {m: [] for m in ms}
            for sd in sorted(clean):
                vals = []
                for m in ms:
                    rr = [r for r in rnd if r["k"] == k and r["method"] == m and r["seed"] == sd]
                    v = float(np.mean([(r["clean"] - r["acc"]) > 5 for r in rr])) if rr else float("nan")
                    vals.append(v); per[m].append(v)
                parts.append(f"seed {sd}: " + "/".join(f"{v:.3f}" for v in vals))
            print(f"  k={k:<4} " + "   ".join(parts))
            print(f"         across {len(clean)} models: " + "   ".join(
                f"{m} mean {np.nanmean(per[m]):.3f} (min {np.nanmin(per[m]):.3f}, max {np.nanmax(per[m]):.3f})" for m in ms))
            if len(clean) >= 3:
                for m in ms[1:]:
                    d = np.array(per["none"]) - np.array(per[m]); n = len(d)
                    if np.isnan(d).any(): continue
                    try:
                        from scipy import stats
                        tc = float(stats.t.ppf(0.975, n - 1))
                    except Exception:
                        tc = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145}.get(n - 1, 2.0)
                    h = tc * d.std(ddof=1) / math.sqrt(n)
                    print(f"         paired per-model difference (none - {m}): mean {d.mean():+.3f}, 95% t-interval [{d.mean()-h:+.3f}, {d.mean()+h:+.3f}], "
                          f"{m} lower in {int((d > 0).sum())}/{n} models  (unit: one trained model; same flips within a model)")
    print("\nNotes: scrub/tmr results hold only under their stated assumptions (fault-free golden copy; no coincident flips).\n"
          "Trials are drawn from a few trained models, so CIs understate the real uncertainty.")


def analyze_train(rows):
    from scipy import stats
    print(f"\nRuns: {len(rows)}   flip_state={rows[0].get('flip_state')}")
    keys = sorted({(r["guard"], r["rate"]) for r in rows}, key=lambda x: (x[1], x[0]))
    base = {r["seed"]: r for r in rows if r["rate"] == 0 and r["guard"] == "none"}
    print(f"{'rate':>7} {'guard':<9} {'n':>3} {'test last':>10} {'sd':>5} {'at best-val':>12} {'diverged':>9} {'flips':>7} {'rollbk':>7} {'repairs':>8} {'paired drop vs clean (last)':>30}")
    for (g, rate) in keys:
        rr = [r for r in rows if r["guard"] == g and r["rate"] == rate]
        a = np.array([r["test_last"] for r in rr]); b = np.array([r["test_at_best_val"] for r in rr])
        div = np.mean([r["test_last"] < 30 for r in rr]); pd = ""
        com = [r["seed"] for r in rr if r["seed"] in base]
        if com and rate > 0:
            d = np.array([base[s]["test_last"] - next(r["test_last"] for r in rr if r["seed"] == s) for s in com])
            if len(d) >= 3:
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
                pd = f"{d.mean():+.2f} [{d.mean()-h:+.2f},{d.mean()+h:+.2f}]"
            else: pd = f"{d.mean():+.2f} (n<3)"
        print(f"{rate:>7} {g:<9} {len(rr):>3} {a.mean():>10.2f} {a.std(ddof=1) if len(a)>1 else float('nan'):>5.2f} {b.mean():>12.2f} "
              f"{div:>9.2f} {np.mean([r['flips'] for r in rr]):>7.1f} {np.mean([r['rollbacks'] for r in rr]):>7.1f} "
              f"{np.mean([r['repairs'] for r in rr]):>8.1f} {pd:>30}")
    print("\nNote: 'at best-val' already includes a free protection (keeping the best checkpoint). Compare guards on 'test last'.\n"
          "'diverged' = last-epoch test accuracy < 30%.")


# ----------------------------------------------------------------- main
def parse_bits(s):
    if not s: return []
    out = []
    for part in s.split(","):
        if "-" in part: lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="inference", choices=["inference", "train", "selftest", "overhead"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--arch", default="smallcnn", choices=sorted(MODELS), help="model architecture (inference mode)")
    ap.add_argument("--formats", default=None, help="comma list from fp32,fp16,bf16,int8,int8pc (inference mode); default fp32")
    ap.add_argument("--trials", type=int, default=300, help="random-flip trials per k (inference)")
    ap.add_argument("--bit-trials", type=int, default=40, help="single flips per bit position (inference)")
    ap.add_argument("--skip-bit-sweep", action="store_true")
    ap.add_argument("--normalize-bits", action="store_true",
                    help="scale k per format so every format gets the same number of flips PER STORED BIT "
                         "(k is then the fp32-equivalent count; small k rounds up to at least 1)")
    ap.add_argument("--ks", default="1,10,100")
    ap.add_argument("--only-bits", default="", help="random flips restricted to these bit POSITIONS (same index in every array; "
                    "meaningful for fp32, where bit 30 is the exponent MSB), e.g. 30 or 23-30")
    ap.add_argument("--exclude-bits", default="", help="random flips avoid these bit positions, e.g. 30 (fp32 mechanism test)")
    ap.add_argument("--eval-n", type=int, default=2000, help="test images used to evaluate each corrupted model")
    ap.add_argument("--rates", default="0.005,0.02", help="flips per optimizer step (train mode)")
    ap.add_argument("--guards", default="none,clamp,rollback,scrub")
    ap.add_argument("--flip-state", action="store_true", help="train mode: flips also hit Adam exp_avg / exp_avg_sq")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--model-def", default=None, metavar="FILE.py:FUNC", help="your own model: Python file and factory function returning an nn.Module")
    ap.add_argument("--weights", default=None, metavar="FILE.pt", help="state_dict for --model-def")
    ap.add_argument("--model-file", default=None, metavar="FILE.pt", help="your own model as TorchScript or a pickled nn.Module")
    ap.add_argument("--data-npz", default=None, metavar="FILE.npz", help="evaluation data for your model: arrays x and y, already preprocessed")
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", metavar="FILE")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20, help="timing repeats per operation (overhead mode)")
    a = ap.parse_args()
    if a.analyze: analyze(a.analyze); return
    if a.mode == "selftest": selftest(); return
    if a.quick:
        a.seeds, a.epochs, a.trials, a.bit_trials, a.ks, a.eval_n = 1, 1, 6, 2, "1,10", 256
        if a.formats is None: a.formats = "fp32,fp16,bf16,int8"
    if a.formats is None: a.formats = "fp32,fp16,bf16,int8,int8pc" if a.mode == "overhead" else "fp32"
    a.ks = [int(x) for x in str(a.ks).split(",")]
    a.formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    for f in a.formats: assert f in FMTS, f"unknown format {f}"
    a.only_bits = parse_bits(a.only_bits); a.exclude_bits = parse_bits(a.exclude_bits)
    assert not (a.only_bits and a.exclude_bits), "use only one of --only-bits / --exclude-bits"
    torch.set_num_threads(a.threads)
    seeds = [42, 123, 777, 1337, 2024][:a.seeds]
    a.custom = bool(a.model_def or a.model_file)
    if a.custom:
        assert not (a.model_def and a.model_file), "use either --model-def or --model-file"
        assert a.data_npz, "a custom model needs --data-npz (your evaluation data)"
        assert a.mode in ("inference", "overhead"), "custom models: inference or overhead mode only"
        assert not a.weights or a.model_def, "--weights belongs to --model-def"
        a._model, nm = load_custom(a, print); seeds = [0]
        import hashlib
        def sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""): h.update(blk)
            return h.hexdigest()
        a._hashes = dict(model_def_sha256=sha(a.model_def.partition(":")[0]) if a.model_def else None,
                         weights_sha256=sha(a.weights) if a.weights else None,
                         model_file_sha256=sha(a.model_file) if a.model_file else None, data_sha256=sha(a.data_npz))
        # the identity of model + data is part of the experiment key, so a new file with the same name never resumes an old run
        a.arch = "custom:" + nm + "@" + hashlib.sha256(json.dumps(a._hashes, sort_keys=True).encode()).hexdigest()[:8]
        print(f"    artifact hashes: {a._hashes}")
        if "int8pc" in a.formats:
            bad = [type(m).__name__ for m in a._model.modules() if "ConvTranspose" in type(m).__name__ or "ConvTranspose" in str(getattr(m, "original_name", ""))]
            if bad: raise SystemExit("int8pc assumes output channels on dim 0; this model has ConvTranspose layers. Use fp32/fp16/bf16/int8 for it.")
        loaders = custom_loaders(a)
        with torch.no_grad():        # sanity checks before any fault test, so a wrong setup does not produce a fake audit
            xs, ys = loaders[2].dataset.tensors
            try: out = a._model(xs[:8])
            except Exception as e: raise SystemExit(f"model does not run on your data (check shape/preprocessing): {e}")
            assert out.dim() == 2 and out.shape[0] == 8, f"expected logits of shape (N, classes), got {tuple(out.shape)}"
            assert int(ys.max()) < out.shape[1] and int(ys.min()) >= 0, f"labels outside 0..{out.shape[1]-1}"
            assert torch.isfinite(out).all(), "model output has NaN/Inf on clean data"
        clean_acc = evaluate(a._model, loaders[2]); chance = 100.0 / out.shape[1]
        print(f"    sanity: output (N, {out.shape[1]}), clean accuracy on your data {clean_acc:.2f}% (chance {chance:.1f}%)")
        if clean_acc < 1.5 * chance:
            raise SystemExit("clean accuracy is near chance: wrong preprocessing, label order or weights? Fix this first, a fault test on it is meaningless.")
    else:
        assert not (a.weights or a.data_npz), "--weights / --data-npz are only used with --model-def or --model-file"
        loaders = get_loaders(a.fake_data)
    log = lambda s: print(s, flush=True)
    out = a.out or {"inference": "bitflip_inference.jsonl", "overhead": "bitflip_overhead.jsonl"}.get(a.mode, "bitflip_train.jsonl")
    if a.mode == "train": assert a.arch == "smallcnn", "train mode supports smallcnn only"
    if a.mode == "inference":
        done = set()
        if os.path.exists(out):
            for l in open(out):
                if l.strip():
                    r = json.loads(l)
                    if r.get("kind") == "seed_done":
                        done.add((r["seed"], r["epochs"], tuple(r.get("fmts", ["fp32"])), r.get("bits", "all"),
                                  r.get("sweep", True), r.get("norm", False), tuple(r.get("ks", [])), r.get("arch", "smallcnn")))
        with open(out, "a") as f:
            for i, s in enumerate(seeds):
                key = (s, a.epochs, tuple(a.formats), bits_label(a), not a.skip_bit_sweep,
                       bool(a.normalize_bits), tuple(a.ks), a.arch)
                if key in done: log(f"[{i+1}/{len(seeds)}] seed {s}: already done"); continue
                log(f"[{i+1}/{len(seeds)}] seed {s}")
                run_inference(s, a, loaders, f, log)
    elif a.mode == "overhead":
        with open(out, "a") as f: run_overhead(a, loaders, f, log)
    else:
        rates = [0.0] + [float(x) for x in a.rates.split(",")]
        guards = a.guards.split(",")
        jobs = [("none", 0.0, s) for s in seeds] + [(g, r, s) for s in seeds for r in rates[1:] for g in guards]
        done = set()
        if os.path.exists(out):
            for l in open(out):
                if l.strip():
                    r = json.loads(l); done.add((r["guard"], r["rate"], r["seed"], r["epochs"]))
        for i, (g, r, s) in enumerate(jobs):
            if (g, r, s, a.epochs) in done: log(f"[{i+1}/{len(jobs)}] {g} rate={r} seed={s}: already done"); continue
            log(f"[{i+1}/{len(jobs)}] guard={g} rate={r} seed={s}")
            rec = run_train(g, r, s, a.epochs, loaders, a.flip_state, a.ckpt_every, log)
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
            log(f"      -> test last {rec['test_last']:.2f}  flips {rec['flips']}  rollbacks {rec['rollbacks']}  repairs {rec['repairs']}  {rec['minutes']:.1f} min")
    log(f"\nDone. Analyze with:  python bitflip_test.py --analyze {out}")


if __name__ == "__main__":
    main()
