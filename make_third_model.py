#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Dimitar Kretski
"""make_third_model.py - trains a DIFFERENT architecture (torchvision MobileNetV2, width 0.5, depthwise convs + BatchNorm)
on CIFAR-10 and writes the three files that bitflip_test.py needs for its custom-model path:
    mnv2_def.py        build() returns the architecture
    mnv2.pt            state_dict (best validation epoch)
    mnv2_eval.npz      x, y = 5000 test images, already normalized
Then run:
    python bitflip_test.py --mode selftest
    python bitflip_test.py --mode inference --model-def mnv2_def.py:build --weights mnv2.pt --data-npz mnv2_eval.npz ^
        --formats int8,int8pc --normalize-bits --skip-bit-sweep --ks 100 --trials 300 --eval-n 2000 --out mnv2_audit.jsonl
    python bitflip_test.py --analyze mnv2_audit.jsonl
This is still a model YOU trained (a different architecture, not a third party's); it checks that the tool works on a
different layer mix. A model from someone else is the stronger test.
Usage: python make_third_model.py [--epochs 6] [--fake-data]"""
import argparse, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim

DEF = '''import torch.nn as nn, torchvision
def build():
    m = torchvision.models.mobilenet_v2(weights=None, num_classes=10, width_mult=0.5)
    m.features[0][0].stride = (1, 1)          # 32x32 inputs
    m.features[2].conv[1][0].stride = (1, 1)  # keep some resolution
    return m
'''

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--threads", type=int, default=8); a = ap.parse_args(); torch.set_num_threads(a.threads)
    open("mnv2_def.py", "w").write(DEF)
    import importlib.util
    spec = importlib.util.spec_from_file_location("d", "mnv2_def.py"); d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    torch.manual_seed(0); np.random.seed(0); model = d.build()
    from torch.utils.data import DataLoader, TensorDataset, Subset
    if a.fake_data:
        g = torch.Generator().manual_seed(0)
        tr_x, tr_y = torch.randn(512, 3, 32, 32, generator=g), torch.randint(0, 10, (512,), generator=g)
        te_x, te_y = torch.randn(300, 3, 32, 32, generator=g), torch.randint(0, 10, (300,), generator=g)
        va_x, va_y = te_x, te_y
        tr = DataLoader(TensorDataset(tr_x, tr_y), batch_size=128, shuffle=True)
    else:
        import torchvision, torchvision.transforms as T
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        t_tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), T.Normalize(mean, std)])
        t_ev = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        full_tr = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=t_tr)
        full_ev = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=t_ev)
        test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=t_ev)
        idx = np.random.default_rng(0).permutation(len(full_tr)); val_idx, tr_idx = idx[:5000], idx[5000:]
        tr = DataLoader(Subset(full_tr, tr_idx.tolist()), batch_size=128, shuffle=True)
        va = [full_ev[i] for i in val_idx[:2000]]; va_x = torch.stack([x for x, _ in va]); va_y = torch.tensor([y for _, y in va])
        tidx = np.random.default_rng(1).permutation(len(test))[:5000]
        te = [test[i] for i in tidx]; te_x = torch.stack([x for x, _ in te]); te_y = torch.tensor([y for _, y in te])
    opt = optim.Adam(model.parameters(), lr=2e-3); crit = nn.CrossEntropyLoss(); best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad(set_to_none=True); crit(model(x), y).backward(); opt.step()
        model.eval()
        with torch.no_grad(): acc = 100.0 * (torch.cat([model(va_x[i:i+500]).argmax(1) for i in range(0, len(va_x), 500)]) == va_y).float().mean().item()
        print(f"epoch {ep+1}/{a.epochs} val {acc:.2f}", flush=True)
        if acc > best: best, best_state = acc, copy.deepcopy(model.state_dict())
    torch.save(best_state, "mnv2.pt")
    np.savez("mnv2_eval.npz", x=te_x.numpy().astype(np.float32), y=te_y.numpy().astype(np.int64))
    print(f"wrote mnv2_def.py, mnv2.pt, mnv2_eval.npz  (best val {best:.2f}%)")

if __name__ == "__main__":
    main()
