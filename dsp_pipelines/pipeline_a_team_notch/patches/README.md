# Patches

`detector_reset.patch`: fix for a lock-out of the Christov R-peak detector. A large start-up transient set its
amplitude threshold so high, and its noise term so negative, that it never found a beat again (53 of 21,799
PTB-XL records). The fix restarts the detector state when no beat has been found for 2 s. Applied in the
`team_dsp/` copy; the same issue exists in the full-signal detector `christov_r_peak_detector.py`.
