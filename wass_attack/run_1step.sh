#!/bin/bash
python attack.py ECG200 fcn True 0.05 0.1 linf
python attack.py ECG200 cnn True 0.05 0.1 linf
python attack.py ECGFiveDays fcn True 0.1 0.1 linf
python attack.py ECGFiveDays cnn True 0.1 0.1 linf
