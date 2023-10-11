exp="default"
gpu_num="8"

model="deaotl"
	
stage="pre_ytb_dav"

## Evaluation ##

dataset="lvos"
split="val"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

split="test"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}