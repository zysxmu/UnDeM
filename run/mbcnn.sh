
# fhdmi week sup
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--arch MBCNN \
--traindata_path '/fhdmi_class/train/' \
--testdata_path '/FHDMi/test/' \
--dataset fhdmi \
--batchsize 32 \
--lr 1e-4 \
--patch_size 192 \
--tensorboard \
--max_epoch 150 \
--operation train \
--name "mbcnn-fhdmi" \
--generation_model_path 'path_for_generation_model_with192patchsize'
