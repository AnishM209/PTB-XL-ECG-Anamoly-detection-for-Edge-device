# Validation of pipeline B

| Script | What it tested |
|---|---|
| `hw_filter_test.py` (+ `.sbatch`) | the PYNQ script's original filters (causal notch Q30, band-pass 0.5 to 40 Hz) split into parts, clean and noisy |
| `chain_test.py` | lives in `../../shared/evaluation/` (also used by pipeline A): chains A to D3 at 500 Hz, clean, DC offset, noisy; writes `chain_thresholds.json` |
| `device_chain.py` (+ `.sbatch`) | `validate`: PTB-XL taken to 250 Hz, chain with Q30 and Q5, thresholds on fold 9, writes `config_device_q30.json` and `config_device_q5.json`. `recording <file>`: per-window predictions, heart rate and plot for a PYNQ recording (`--q`, `--invert`) |
| `mains_test.py` (+ `.sbatch`) | synthetic 50 Hz (0 to 17 mV, frequency 50 +- 0.05 Hz) through Q30, Q15, Q10, Q5; residual mains, AUROC, sensitivity and specificity |

Run from one folder with the shared scripts on the path, as described in the top-level README, e.g.

```
sbatch device_chain.sbatch
python device_chain.py recording ~/recordings/still_01.npz --q 5
sbatch mains_test.sbatch
```

Each script prints SELF-CHECK lines; a FAIL means the numbers are not comparable and the run should be checked.
