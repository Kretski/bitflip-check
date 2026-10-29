# bitflip-check

A small, reproducible check of what a few flipped bits in a neural network's **stored weights** do to its accuracy, in the number format the model would actually be stored in, and how much cheap protections help (and cost).

**Status: research prototype.** Three small models (ResNet8 with five trainings, SmallCNN, one MobileNetV2), simulated storage faults, no hardware. Not a safety tool; no safety or certification claim is made. The techniques are known (see *Related work*); this repo is a small, open, reproducible study and a script you can point at your own model.

## What it does

`bitflip_test.py` takes a trained model, stores its weights in a chosen format, flips random bits in the stored image, decodes back to float32 and measures accuracy.

- **Formats:** `fp32`, `fp16`, `bf16`, `int8` (one float32 scale per tensor), `int8pc` (one float32 scale per output channel). The float32 **scales** of the int8 formats are part of the fault domain.
- **Fault domain:** every float tensor of the model: parameters and float buffers (e.g. BatchNorm running mean/var).
- **Fault model:** one uniformly random bit of one stored element per flip. `k` = flips per memory image; `--normalize-bits` gives every format the same number of flips **per stored bit** (the fp32 image gets k flips, a 16-bit image k/2, an 8-bit image about k/4).
- **Protections compared:**
  - `clamp`: NaN/Inf -> 0 and clip each tensor to 1.1x its clean |max|. No extra copy. (This is the known "range restriction" idea.)
  - `scrub`: position-weighted checksum per tensor plus a golden copy; restore on mismatch. *Assumes the golden copy and checksums are fault-free.*
  - `scrub_scales`: the same, but only for the float32 scales of the int8 formats (about 0.2% to 2% extra memory).
  - `tmr`: three copies, bitwise majority vote (exposed to 3k flips). *Fails only on coincident flips of the same bit.*
- **Cost:** `--mode overhead` times each protection and reports extra memory.

## Quick start

```
pip install torch torchvision numpy
python bitflip_test.py --mode selftest
python bitflip_test.py --mode inference --arch resnet8 --formats fp32,fp16,bf16,int8,int8pc \
    --normalize-bits --ks 100 --trials 300 --bit-trials 25 --seeds 3 --epochs 8 --out bitflip_resnet8.jsonl
python bitflip_test.py --analyze bitflip_resnet8.jsonl
python bitflip_test.py --mode overhead --arch resnet8 --repeats 50
```

### Your own model

```
python bitflip_test.py --model-def mynet.py:build --weights mynet.pt --data-npz eval.npz \
    --formats fp32,int8,int8pc --normalize-bits --ks 100 --trials 300 --bit-trials 25
python bitflip_test.py --model-file mynet_full.pt --data-npz eval.npz --formats int8,int8pc --normalize-bits --ks 100
```

- `--model-def FILE.py:FUNC`: your file; `FUNC()` returns the `torch.nn.Module`; `--weights` is its `state_dict` (loaded with `weights_only=True`). `--model-file` takes a TorchScript file or a fully pickled module instead. Both routes run your code or a pickle, so use only files you trust.
- `--data-npz FILE.npz`: arrays `x` and `y`, preprocessed exactly as your model expects (at least 100 samples). Use `--eval-n` to limit the evaluation cost on big models.
- Fault domain: all float parameters and float buffers, float32 compute. Integer tensors already inside the model are not covered. For `int8pc` the output-channel axis is dim 0 of every weight tensor with more than one dimension (right for Conv2d and Linear). Models containing ConvTranspose layers are refused for `int8pc` with an explanatory message (use `fp32`, `fp16`, `bf16` or `int8` for them). Models with lazily initialised layers (`LazyLinear` and similar) are not supported: run one forward pass and save the initialised weights first.
- One model is one run: the intervals then describe flip-to-flip noise only, not variation between trainings.
- `--mode overhead` works the same way and uses your data for the timing input.

Models: `--arch smallcnn` (default, 1.07 M parameters) or `--arch resnet8` (78 k parameters, with BatchNorm). Both are trained from scratch on CIFAR-10 (8 epochs, 3 seeds). Clean accuracy: SmallCNN about 70%, ResNet8 about 66-68%.

## Results

All numbers are P(accuracy drops by more than 5 points) with 95% Wilson intervals unless stated, pooled over 3 trained models (n = 900 random-flip trials per cell), except the ResNet8 int8 random-flip table in section 2 (5 trained models, n = 1500). Drops are measured against the same format's clean accuracy on the same fixed 2000-image test subset. The intervals understate real uncertainty because only three models are trained. Raw trial data are in the `.jsonl` files.

### 1. Float formats: one bit dominates

Single flip per bit position, ResNet8 (75 trials per position, pooled over 3 seeds):

| Format | Most damaging bit | Mean drop | P(drop > 5) | Next worst bit |
|---|---|---|---|---|
| fp32 | 30 (exponent MSB) | 27.7 pts | 0.51 | bit 29: 1.9 pts |
| fp16 | 14 (exponent MSB) | 21.4 pts | 0.39 | sign: 1.6 pts |
| bf16 | 14 (exponent MSB) | 32.0 pts | 0.59 | bit 13: 0.8 pts |

Random flips at equal flips per stored bit (k = 100 fp32-equivalent):

| Model | Format | No protection | `clamp` |
|---|---|---|---|
| SmallCNN | fp32 | 66.1% [63.0, 69.1] | 0 / 900 |
| SmallCNN | fp16 | 62.4% [59.2, 65.5] | 0 / 900 |
| SmallCNN | bf16 | 65.0% [61.8, 68.0] | 0 / 900 |
| ResNet8 | fp32 | 84.2% [81.7, 86.5] | 13.3% [11.3, 15.7] |
| ResNet8 | fp16 | 83.4% [80.9, 85.7] | 5.1% [3.9, 6.8] |
| ResNet8 | bf16 | 84.6% [82.0, 86.8] | 12.8% [10.8, 15.1] |

- The three float formats are about equally vulnerable per stored bit.
- `clamp` removed the damage on SmallCNN but **not** on ResNet8, where 5-13% of trials still lose more than 5 points. It does not generalize as "protection solved". Why fp16 does better than fp32/bf16 here is not explained by these data.

### 2. int8: weights are fairly robust, the float32 scales are not

Single flips in int8 weights (ResNet8): mean drop at most 1.0 pt (1 of 75 trials above 5 pts at the sign bit for `int8`, 0 of 600 for `int8pc`). SmallCNN: 0 of 600 single flips above 5 pts for both. This is an observation on these models, not a proof of zero risk.

A single flip in a float32 **scale**:

| Model | Scheme | Scales | Share of stored bits | Bit 30 mean drop | P(drop > 5 \| scale hit), uniform over 32 bits |
|---|---|---|---|---|---|
| SmallCNN | int8 per-tensor | 8 | 0.003% | 58.8 pts | 0.165 |
| SmallCNN | int8 per-channel | 366 | 0.137% | 26.8 pts | 0.047 |
| ResNet8 | int8 per-tensor | 47 | 0.238% | 51.5 pts | 0.231 |
| ResNet8 | int8 per-channel | 383 | 1.909% | 51.9 pts | 0.116 |

Per-channel scales make each hit milder, but there are many more scales, so a random flip is much more likely to hit one. Measured with random flips (ResNet8, 25 flips per image, equal flips per stored bit; **5 trained models, n = 1500 per cell**):

| Scheme | No protection | `clamp` | `scrub_scales` | `scrub` / `tmr` |
|---|---|---|---|---|
| int8 per-tensor | 5.4% [4.4, 6.7] | 5.2% [4.2, 6.4] | 4.2% [3.3, 5.3] | 0 (by construction) |
| int8 per-channel | 9.1% [7.8, 10.7] | 7.4% [6.2, 8.8] | 3.8% [2.9, 4.9] | 0 (by construction) |

Per trained model, without protection: int8 per-channel 6.3% to 10.3%, int8 per-tensor 3.3% to 6.3% (a rough model-level 95% interval, t-distribution over 5 models: about 7-11% and 4-7%).

- Paired by trained model (same flips within a model), `scrub_scales` was below no protection in **5 of 5 models** for both schemes. Mean difference: int8 per-channel 5.3 points (t-interval over 5 models [4.0, 6.6]; `clamp` alone 1.7 [0.6, 2.9]); int8 per-tensor 1.2 points [0.6, 1.8] (`clamp` alone on per-tensor: 0.2 [-0.0, 0.4], lower in 3 of 5 models, i.e. no clear effect). The unit here is the trained model, so this interval is wider in spirit than the pooled Wilson intervals, but it is still one architecture and five models. `--analyze` prints these intervals.
- Per-channel had the higher measured failure rate on ResNet8; protecting only the scales removed the difference (3.8% vs 4.2%). On SmallCNN the analytic estimate is also higher for per-channel, but no failures were observed at 25 flips.
- What remains after `scrub_scales` (about 4%) comes from accumulated flips in weights and small tensors (BatchNorm statistics, biases). On a BatchNorm-folded deployment this residual may differ; **not tested**.
- SmallCNN: 0 of 900 for every method and both int8 schemes at 25 flips. Not informative about protections.
- `scrub`/`tmr` show 0 because they restore the clean model *under their stated assumptions*; this is not a measured fault-detection rate.

### 2b. A different architecture: MobileNetV2 (one training)

Torchvision MobileNetV2, width 0.5 (719 034 stored float values in 262 tensors, depthwise convolutions, BatchNorm), trained 6 epochs on CIFAR-10 by `make_third_model.py` (74.4% on its 5000 evaluation images). Run through the custom-model path (`--model-def`), 300 random-flip trials per cell, 25 (int8) or 26 (int8 per-channel) flips per image, evaluation on 2000 images:

| Scheme | No protection | `clamp` | `scrub_scales` | Trials with a drop above 5 pts removed by `scrub_scales` (paired, same flips) |
|---|---|---|---|---|
| int8 per-tensor | 3.7% [2.1, 6.4] | 3.3% | 3.0% [1.6, 5.6] | 2 removed, 0 added (exact sign test p = 0.50) |
| int8 per-channel | 4.3% [2.5, 7.3] | 2.7% | 2.3% [1.1, 4.7] | 6 removed, 0 added (p = 0.031) |

- This is one trained model. The p-values describe the flip-to-flip noise of this one network only, and 4 comparisons are shown, so p = 0.031 would not survive a multiple-comparison correction. Treat it as "same direction as ResNet8", not as a second confirmed result.
- `scrub_scales` only undoes scale flips, so "0 added" is expected by construction; the test asks whether the removed cases are more than a handful.
- The baseline risk is lower than for ResNet8 (4.3% vs 9.1% for per-channel), and per-channel was not worse than per-tensor here. These data do not explain why.
- With `scrub_scales`, the share of runs that collapse to near-chance accuracy is almost unchanged (int8: 3.0% to 3.0%; per-channel: 2.7% to 2.3%). The most severe failures on this model do not come from the scales; which tensors cause them was not investigated.
- Cost (NumPy, one laptop, forward pass batch 1 = 9.1 ms): `scrub_scales` 0.010 ms (int8) and 0.035 ms (per-channel), extra memory 0.1% and 5.0% of the stored image. A full `scrub` pass costs 7.6 ms (int8) and 70 ms (per-channel); the latter is dominated by one checksum segment per scale in this implementation.

### 3. Cost of the protections

Measured on the author's laptop (CPU, NumPy implementation, median of 50 repeats). Forward pass, batch 1: ResNet8 0.95 ms; SmallCNN 0.5-0.8 ms (it varied between runs).

| ResNet8 | `clamp` | `scrub` (checksum pass) | `scrub_scales` (checksum pass) | `tmr` (vote) | Extra memory |
|---|---|---|---|---|---|
| fp32 | 0.47 ms | 0.58 ms | n/a | 0.28 ms | clamp 0.1%, scrub 100%, tmr 200% |
| int8 | 0.44 ms | 0.87 ms | 0.010 ms | 0.27 ms | scrub_scales 0.2% |
| int8 per-channel | 0.52 ms | 3.1 ms (see note) | 0.009 ms | 0.25 ms | scrub_scales 1.9% |

| SmallCNN | `clamp` | `scrub` | `scrub_scales` | `tmr` |
|---|---|---|---|---|
| fp32 | 6.1 ms | 9.6 ms | n/a | 7.4 ms |
| int8 | 6.6 ms | 10.2 ms | 0.007 ms | 5.2 ms |
| int8 per-channel | 6.3 ms | 13.0 ms | 0.007 ms | 4.3 ms |

Notes:
- Protection time scales with the number of stored weights, not with compute. SmallCNN has one large dense layer, so its full-image protections cost 6-18x a forward pass; ResNet8's cost 0.3-0.9x.
- `scrub` for int8 per-channel is slow because this implementation keeps one checksum segment per scale (hundreds of small NumPy calls). It is an artifact of the implementation, not of the method. `scrub_scales` is one vectorized pass.
- `tmr` here is bitwise voting of three stored weight copies (memory TMR), **not** triple execution of the model.
- Protections do not have to run on every inference. Overhead = check time x check rate; a suitable rate depends on the fault rate of the environment, which this tool does not model.
- Times are for this NumPy code on one laptop. They are not a prediction for any target hardware.

## Limits (please read)

- Small CIFAR-10 models only: ResNet8 (five trainings), SmallCNN (three), MobileNetV2 (one). Confidence intervals understate uncertainty; results may not carry over to other architectures, datasets or sizes.
- Storage format only: weights are corrupted in fp16/bf16/int8 and decoded to float32 for compute. No faults in activations, accumulators, compute units, control logic or instruction memory. No optimizer state in inference mode.
- `k` is flips per memory image, not a physical fault rate. No ECC behaviour of any specific platform is modelled. Hardware with ECC corrects single-bit flips before they reach the model.
- `scrub`, `scrub_scales` and `tmr` hold only under their assumptions (fault-free golden copy and checksums; no coincident flips of the same bit).
- int8 biases and BatchNorm statistics are stored as int8 with one scale per tensor here; real toolchains usually fold BatchNorm and keep biases as int32. This changes the fault domain.
- The analytic combination "P(drop > 5 \| scale hit) x P(hit)" assumes independent single-flip effects and underestimates the measured random-flip rates, because several moderate hits accumulate.
- No claim about hardware, radiation environments, safety or certification.

## Related work

This is not a new method. Similar ideas and tools exist:

- [Ranger](https://arxiv.org/abs/2003.13874v3): range restriction to mask critical faults (the same idea as `clamp`).
- [PyTorchFI](https://research.nvidia.com/publication/2020-06_pytorchfi-runtime-perturbation-tool-dnns): open-source runtime fault injector for PyTorch.
- [MRFI](https://ar5iv.labs.arxiv.org/html/2306.11758): open-source multi-resolution fault injection (bit flips in quantized weights and activations).
- [Mind the Scaling Factors](https://past.date-conference.com/proceedings-archive/2022/html/0758.html) (DATE 2022): bit flips in the INT8 multiplier operands of an NVDLA accelerator; shows that larger quantization scaling factors amplify the effect of a flip. It does not flip bits in the stored scale words themselves.
- [Your Scale Factors are My Weapon](https://cvpr.thecvf.com/virtual/2025/poster/33058) (CVPR 2025): a targeted bit-flip *attack* on the scale factors of quantized Vision Transformers. That scales are a sensitive target is therefore known; this repo measures it under random faults and tests cheap protection of the scales.

## Reproducibility

- Fixed seeds (42, 123, 777; plus 1337 and 2024 for the 5-model int8 runs), deterministic training on CPU; clean models are cached as `bitflip_model_*.pt`.
- `--mode selftest` checks bit semantics, round trips for every format, restoration by `scrub`, voting by `tmr`, and that a per-channel scale flip damages exactly one channel.
- Raw per-trial data: `bitflip_*.jsonl`. Analysis: `--analyze FILE`.

## License / author

Dimitar Kretski, independent researcher, Varna, Bulgaria.

Licensed under the Apache License 2.0 (see `LICENSE`). Copyright 2026 Dimitar Kretski. The software is provided "as is", without warranty; see the license for the terms.
