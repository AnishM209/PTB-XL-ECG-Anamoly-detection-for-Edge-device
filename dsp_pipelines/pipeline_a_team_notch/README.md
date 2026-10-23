# Pipeline A: notch and low-pass, then the team pipeline (variant TC)

```
ADS1299 (250 SPS) -> notch 50 Hz, Q5 (IIR) -> low-pass 40 Hz, Butterworth order 2 (IIR) -> upsample x2 to 500 Hz
  -> median filter 0.6 s -> LPF 150 Hz -> SWT (Haar, 7 levels) -> Christov R peaks on D4 -> QRS removal (D1..D6)
  -> per-band thresholds -> inverted thresholding -> inverse SWT (artefact) -> preprocessed minus artefact -> models
```

The team pipeline is a motion-artefact remover (First Phase of Berwal et al.). The notch and 40 Hz low-pass
in front are our addition: real recordings carry 0.3 to 42 mV of mains, the team pipeline's 150 Hz low-pass
passes 50 Hz, and without a notch the binary model called 9 of 10 windows of a healthy, still subject abnormal.
The notch must come before the SWT and the R-peak detector.

## Results in short

| Test | Result |
|---|---|
| PTB-XL fold 10, device setting (250 Hz) | binary dAUROC -0.0098 (pipeline B: -0.0076), AF +0.0005 |
| NSTDB noise (team pipeline at 500 Hz) | never better than noisy raw in 78 comparisons; worse under electrode motion (-0.013 to -0.022) |
| Still recording C | 10/10 windows Normal, no AF, heart rate 63 to 71 bpm |
| Movement recording move_01 | 10/10 windows Abnormal, false AF in 3 windows |
| Heart rate under arm swings and twists | steady about 100 bpm (best of all detectors tested) |

Conclusion: works as designed and is usable on still signals, but adds no classification robustness. Its
Christov detector is the best heart-rate source. Full details: `../docs/ecg_dsp_pipelines.html`.

## Folders

| Folder | Contents |
|---|---|
| `team_dsp/` | the team's pipeline code (with the R-peak detector reset fix) |
| `validation/` | scripts that tested the team pipeline: device setting, noise stress test, earlier batch tests |
| `patches/` | the detector reset patch |
| `results/` | summary outputs of the tests |
