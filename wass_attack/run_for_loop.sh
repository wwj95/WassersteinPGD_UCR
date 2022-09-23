#!/bin/bash
for dataset in  ECG200			
do
	for model_type in cnn
	do
		for wass_clip in 0.01 0.02 0.03 0.04 0.05 0.08 0.1 0.2 0.3 0.4 0.5
		do
			for norm_clip  in 0.1 0.2
			do
				output_file="./log2/${dataset}_${model_type}.log"
				echo "######################################################################" >> ${output_file}
				echo "Log for python attack.py ${dataset} ${model_type} True ${wass_clip} ${norm_clip} " >> ${output_file}
				python attack.py ${dataset} ${model_type} True ${wass_clip} ${norm_clip} linf >> ${output_file}
				echo "Log for python attack.py ${dataset} ${model_type} False ${wass_clip} ${norm_clip} " >> ${output_file}
                                python attack.py ${dataset} ${model_type} False ${wass_clip} ${norm_clip} linf 1>> ${output_file}
			done
		done
	done
done
