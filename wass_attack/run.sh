#!/bin/bash

python attack.py ECG200 fcn True 0.1 0.1>>./tmp.log
python attack.py ECG200 fcn False 0.1 0.1 >>./tmp.log
python attack.py ECG200 cnn True 0.1 0.1>>./tmp.log
python attack.py ECG200 cnn False 0.1 0.1 >>./tmp.log
python attack.py ECG200 resnet True 0.1 0.1 >>./tmp.log
python attack.py ECG200 resnet False 0.1 0.1 >>./tmp.log
python attack.py ECG200 mlp True 0.1 0.1>>./tmp.log
python attack.py ECG200 mlp False 0.1 0.1 >>./tmp.log
python attack.py ECG200 inception True 0.1 0.1>>./tmp.log
python attack.py ECG200 inception False 0.1 0.1 >>./tmp.log

#python attack.py ECG5000 fcn True >>./tmp.log
#python attack.py ECG5000 fcn False 1 5 >>./tmp.log
#python attack.py ECG5000 cnn True >>./tmp.log
#python attack.py ECG5000 cnn False 1 5 >>./tmp.log
#python attack.py ECG5000 resnet True >>./tmp.log
#python attack.py ECG5000 resnet False 1 5 >>./tmp.log
#python attack.py ECG5000 mlp True >>./tmp.log
#python attack.py ECG5000 mlp False 1 5 >>./tmp.log
#python attack.py ECG5000 inception True >>./tmp.log
#python attack.py ECG5000 inception False 1 5 >>./tmp.log


python attack.py ECGFiveDays fcn True 0.1 0.1>>./tmp.log
python attack.py ECGFiveDays fcn False 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays cnn True 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays cnn False 0.1 0.1  >>./tmp.log
python attack.py ECGFiveDays resnet True 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays resnet False 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays mlp True 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays mlp False 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays inception True 0.1 0.1 >>./tmp.log
python attack.py ECGFiveDays inception False 0.1 0.1 >>./tmp.log

#python attack.py NonInvasiveFatalECG_Thorax1 fcn True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 fcn False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 cnn True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 cnn False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 resnet True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 resnet False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 mlp True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 mlp False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 inception True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax1 inception False 1 5 >>./tmp.log


#python attack.py NonInvasiveFatalECG_Thorax2 fcn True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 fcn False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 cnn True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 cnn False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 resnet True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 resnet False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 mlp True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 mlp False 1 5 >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 inception True >>./tmp.log
#python attack.py NonInvasiveFatalECG_Thorax2 inception False 1 5 >>./tmp.log
