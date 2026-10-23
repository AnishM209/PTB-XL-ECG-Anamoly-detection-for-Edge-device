# Results of pipeline B tests

Summary outputs copied from `~/dsp_test/batch` on Prajna, and plots of the real recordings.

| File | From |
|---|---|
| `device_chain_summary.txt` | `device_chain.py validate` |
| `mains_test_summary.txt`, `mains_test.png` | `mains_test.py` (note: in the lower plot panels the line colours do not match the legend; the text file is authoritative) |
| `hw_filter_summary.txt`, `hw_filter_summary.csv` | `hw_filter_test.py`: AUROC, sensitivity, specificity and dAUROC per task, condition and filter variant (raw, notch, notch + HP 0.5 Hz, notch + LP 40 Hz, hw) |
| `hw_filter_examples.png` | `hw_filter_test.py`: example clean fold-10 records, raw against notch only against notch + 0.5-40 Hz (PYNQ script) |
| `still_01_pred_q5.png`, `still_01_pred_q30.png`, `move_01_pred_q5.png`, `move_01_pred_q30.png` | `device_chain.py recording` |

Recording plots show one team member's ECG (healthy subject, consent given for this use).
