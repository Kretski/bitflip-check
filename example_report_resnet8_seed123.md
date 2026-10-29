# bitflip-check report

- Result file: `audit.jsonl`
- Generated: 2026-10-29 08:08 UTC with `make_report.py`; test run by `bitflip_test.py` v2.11
- Status: research-grade measurement on simulated storage faults. Not a safety assessment, certification or guarantee.

## Model: `custom:bitflip_test`

No artifact hashes recorded in this file (hashes are written by bitflip_test.py v2.10 or later for your own models).

### Format `int8`

- Clean accuracy of this format on the evaluation subset: 63.95% (float32 model on the full evaluation set: 66.29%)
- Flips per memory image: reference k=100 -> 25 used; trained models in this file: 1

Random flips, k = 100 (accuracy drop measured against this format's clean accuracy):

| Method | Trials | Mean drop (pts) | P(drop > 5 pts) | 95% CI | P(collapse, acc <= 15%) |
|---|---|---|---|---|---|
| no protection (baseline) | 300 | 2.46 | 4.7% | [2.8, 7.7] | 2.3% |
| clamp (range restriction) | 300 | 2.34 | 4.3% | [2.5, 7.3] | 2.3% |
| scrub_scales (scales only) | 300 | 1.80 | 3.0% | [1.6, 5.6] | 1.7% |
| scrub_scales + clamp | 300 | 1.80 | 3.0% | [1.6, 5.6] | 1.7% |
| scrub (checksum + golden copy) * | 300 | 0.00 | 0.0% | [0.0, 1.3] | 0.0% |
| tmr (3 copies, bitwise vote) * | 300 | 0.00 | 0.0% | [0.0, 1.3] | 0.0% |

\* `scrub` and `tmr` restore the clean model by construction, under their assumptions (fault-free golden copy; no coincident flips of the same bit in `tmr`). A 0% there is not a measured fault-detection rate.

Paired comparison on the same flips (trials with a drop above 5 pts that a protection removes or adds; exact two-sided sign test):

| Protection | Removed | Added | Trials | p |
|---|---|---|---|---|
| clamp (range restriction) | 1 | 0 | 300 | 1.000 |
| scrub_scales (scales only) | 5 | 0 | 300 | 0.062 |

This file contains 1 trained model(s); the interval and p-value describe flip-to-flip noise of that model only, not variation between trainings.

### Format `int8pc`

- Clean accuracy of this format on the evaluation subset: 63.70% (float32 model on the full evaluation set: 66.29%)
- Flips per memory image: reference k=100 -> 25 used; trained models in this file: 1

Random flips, k = 100 (accuracy drop measured against this format's clean accuracy):

| Method | Trials | Mean drop (pts) | P(drop > 5 pts) | 95% CI | P(collapse, acc <= 15%) |
|---|---|---|---|---|---|
| no protection (baseline) | 300 | 2.37 | 5.7% | [3.6, 8.9] | 3.3% |
| clamp (range restriction) | 300 | 1.54 | 5.0% | [3.1, 8.1] | 1.3% |
| scrub_scales (scales only) | 300 | 1.09 | 2.3% | [1.1, 4.7] | 1.3% |
| scrub_scales + clamp | 300 | 1.09 | 2.3% | [1.1, 4.7] | 1.3% |
| scrub (checksum + golden copy) * | 300 | 0.00 | 0.0% | [0.0, 1.3] | 0.0% |
| tmr (3 copies, bitwise vote) * | 300 | 0.00 | 0.0% | [0.0, 1.3] | 0.0% |

\* `scrub` and `tmr` restore the clean model by construction, under their assumptions (fault-free golden copy; no coincident flips of the same bit in `tmr`). A 0% there is not a measured fault-detection rate.

Paired comparison on the same flips (trials with a drop above 5 pts that a protection removes or adds; exact two-sided sign test):

| Protection | Removed | Added | Trials | p |
|---|---|---|---|---|
| clamp (range restriction) | 2 | 0 | 300 | 0.500 |
| scrub_scales (scales only) | 10 | 0 | 300 | 0.002 |

This file contains 1 trained model(s); the interval and p-value describe flip-to-flip noise of that model only, not variation between trainings.

### Reproduce

```
python bitflip_test.py --mode inference --model-def FILE.py:FUNC --weights FILE.pt --data-npz FILE.npz --formats int8,int8pc --normalize-bits --skip-bit-sweep --ks 100 --trials 300 --out repeat.jsonl
python bitflip_test.py --analyze repeat.jsonl
```

Replace the model placeholders with the same files (the SHA-256 values above identify them). Flip positions are random: a repeat gives different flips, so individual percentages differ by the noise shown in the intervals.

## Limits of this measurement

- Storage faults only: stored weights and float buffers are corrupted in the stated format and decoded to float32 for compute. No faults in activations, accumulators, compute units, control logic or instruction memory.
- `k` is the number of flips per memory image, not a physical fault rate. No ECC behaviour or radiation environment of any specific platform is modelled; hardware with ECC corrects single-bit flips before they reach the model.
- With `--normalize-bits`, every format receives the same number of flips per stored bit (a smaller image receives fewer flips).
- Intervals describe the randomness of the flips for the model(s) in this file. They do not cover variation between different trainings, architectures or datasets.
- `scrub` and `tmr` hold only under their stated assumptions.
- Timings come from a NumPy implementation on one machine.
- This is not a safety, reliability or certification statement.
