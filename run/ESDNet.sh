# week sup

CUDA_VISIBLE_DEVICES=0,1 python main.py \
--arch ESDNet-L \
--traindata_path '/fhdmi_class/train/' \
--testdata_path '/FHDMi/test/' \
--dataset fhdmi \
--batchsize 32 \
--lr 2e-4 \
--patch_size 192 \
--tensorboard \
--max_epoch 150 \
--operation train \
--name "ESDNet-L-fhdmi" \
--generation_model_path 'path_for_generation_model_with192patchsize'
