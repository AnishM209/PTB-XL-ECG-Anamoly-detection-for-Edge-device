# Acquisition (PYNQ-Z2 + ADS1299)

| File | Purpose |
|---|---|
| `ads1299_uart_record.ipynb` | runs on the PYNQ (Jupyter). Configures the ADS1299 (250 SPS, gain 24, bias drive), reads channel 1, saves the first 60 s unfiltered to `recordings/<RECORD_NAME>.npz` with timestamps and prints a quality summary (rate, gaps, 50 Hz amplitude, DC, drift, clipping), and streams the signal over `/dev/ttyPS0` at 115200 baud, one integer in microvolts per line |
| `live_plot.py` | laptop: live plot of the UART stream with the model-path filter applied, and the 50 Hz amplitude as a live meter |

Settings in the notebook:

- `STREAM_RAW = True` streams the unfiltered signal, which `live_demo.py` needs (it applies its own filters).
  `False` streams the notebook's own filtered signal (notch Q30 + band-pass 0.5 to 40 Hz) for a plain plotter.
- `RECORD_NAME` names the recording; change it before each recording so files are not overwritten.

Wiring: V2 site to IN1P (positive), V1 site to SRB1 (negative), bias electrode on BIAS_OUT. Keep the
orientation fixed and labelled: swapped leads make the binary model call a normal heart abnormal.

For low mains interference: laptop on battery, good bias contact, fresh gel electrodes, short leads, subject not
touching earthed objects (sitting barefoot on the floor made the signal unusable in our tests).
