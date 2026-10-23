# Live demo

`live_demo.py` shows, updated live: the raw signal from the board, the clean signal after DSP filtering with
detected beats, the heart rate, the two model verdicts with their probabilities, signal quality, and a note when
the electrodes look swapped (inverted T wave). Press I to flip the polarity.

```
python live_demo.py --models ../shared/models --replay <recording.npz>                     # rehearsal
python live_demo.py COM37 --scale 1000 --models ../shared/models                           # pipeline B
python live_demo.py COM37 --scale 1000 --models ../shared/models \
       --pipeline team --team-dir ../pipeline_a_team_notch/team_dsp                        # pipeline A
```

| Option | Meaning |
|---|---|
| `--pipeline d3` (default) | model input = pipeline B, thresholds `config_device_q5.json` |
| `--pipeline team` | model input = pipeline A, thresholds `config_team_TC_q5.json`, heart rate from the team's Christov detector |
| `--scale 1000` | stream units per mV (the board sends microvolts) |
| `--replay file.npz` | play a recording from the PYNQ notebook as if live |
| `--invert` | start with the polarity flipped |
| `--save file.npz` | keep everything received |

The display path (notch + 0.5 to 40 Hz band-pass) is only for the screen. The quality gate withholds the verdict
when the 50 Hz amplitude is above 3 mV, the baseline drifts, movement artefact is present, or no beats are
found. With no data for 1.5 s it shows a flat line and "No signal".

`live_demo_mn.py` is a test version with the notch moved after the team's median filter
(median, then notch + LP40, then LPF 150 Hz, SWT ...). Its thresholds have not been recalibrated for that
order, so it is for testing only.
