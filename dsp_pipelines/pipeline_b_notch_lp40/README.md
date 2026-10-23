# Pipeline B: notch Q5 and 40 Hz low-pass (device chain D3)

```
ADS1299 (250 SPS) -> notch 50 Hz, Q5 (IIR) -> low-pass 40 Hz, Butterworth order 2 (IIR) -> upsample x2 to 500 Hz
  -> 10 s buffer -> each model window minus its own mean -> (x - mu) / sd -> models
```

All filters are causal with their state kept between chunks, as on a running device. There is deliberately no
high-pass: a causal 0.5 Hz high-pass cost the binary model 0.014 to 0.016 AUROC (undershoot after the T wave),
so the electrode DC offset is removed per window instead.

Why Q5 and not Q30: the mains stress test showed that with Q30, 1 mV of mains already halves AF sensitivity
(0.87 to 0.43), while Q5 stays near clean performance up to 2 mV. Q5 costs 0.0076 binary AUROC on clean data.

## Results in short

| Test | Result |
|---|---|
| Chain test (500 Hz) | equal to raw on clean data, safe with a 30 mV DC offset, best or equal under noise |
| PTB-XL fold 10 at 250 Hz | binary dAUROC -0.0076*, AF -0.0006; LP40 helps under muscle noise |
| Mains stress test | Q5 robust to 2 mV of mains; Q30 fails from about 1 mV |
| Still recording C | 10/10 windows Normal (p 0.09 to 0.35, threshold 0.575), no AF |
| Movement recording move_01 | all windows Abnormal, false AF while walking: needs the quality gate |

Thresholds: `../shared/models/config_device_q5.json` (binary 0.5750, AF 0.3286). This is the default model
input of the live demo. Full details: `../docs/ecg_dsp_pipelines.html`.

## Folders

| Folder | Contents |
|---|---|
| `validation/` | device-chain validation, mains stress test, filter tests |
| `results/` | summary outputs of the tests |
