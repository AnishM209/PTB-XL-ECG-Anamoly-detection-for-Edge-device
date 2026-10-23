"""
Pipeline settings shared by every file -- change them HERE, and the block files, the full-signal
pipeline, the bank (hardware) pipeline, the streaming reference and clean_dataset.py all follow.

MEDIAN_WINDOW_S
    Length of the baseline median filter (x - medfilt(x)), in seconds.
    0.6 s keeps the ST segment and T wave (T 97-99 % of a standard diagnostic baseline removal, ST level
    within 0.01 mV) which the anomaly classifier needs. 0.3 s (the paper's value) shrank the T wave to
    52-76 % and shifted the ST level by up to 0.1 mV, so normal beats looked abnormal to the CNN.
    The R-peak detection is identical for 0.3 / 0.45 / 0.6 s (tested). Trade-off: a longer window removes
    less slow motion artefact by itself -- validate on real motion noise (e.g. MIT-BIH NSTDB).
    On the FPGA: window = round(MEDIAN_WINDOW_S * fs) | 1 samples (301 at 500 Hz) and the signal must be
    delayed by half a window (150 samples = 0.3 s) before the median is subtracted (centred median).

A7_INCLUDED
    False (default): A7 (0 - ~2 Hz) is thresholded like the detail bands -- only its outliers are treated
                     as artefact, so the slow ST/T content of normal beats is kept.
    True           : all of A7 is treated as artefact and removed (acts like a ~2 Hz high-pass; distorts ST/T).
"""
FS = 500
MEDIAN_WINDOW_S = 0.6
A7_INCLUDED = False
