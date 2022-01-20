#GPU_ID="7"
#GPU_ID="3"

# felix asus
#TRAIN_DIR_SG="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
#VAL_DIR_SG="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
#TRAIN_DIR_FT="${HOME}/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/"
#VAL_DIR_FT="${HOME}/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/"

# colab
TRAIN_DIR_SG="../data/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
VAL_DIR_SG="../data/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
TRAIN_DIR_FT="../data/Dataset_Multiview_denoised_cropped_110_110_1024_1024_denoised_cropped_down_512px_good_manually_selected_5-2_stacks/"
VAL_DIR_FT="../data/Dataset_Multiview_denoised_cropped_110_110_1024_1024_denoised_cropped_down_512px_good_manually_selected_5-2_stacks/"


# UFORMER

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_1-1D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 1 \
  --out_channel 1
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_1-1D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 1 \
  --out_channel 1
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-1D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 5 \
  --out_channel 1
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-1D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 5 \
  --out_channel 1
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-3D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 5 \
  --out_channel 3
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-3D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 5 \
  --out_channel 3
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-5D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 5 \
  --out_channel 5
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_5-5D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 5 \
  --out_channel 5
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-1D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 10 \
  --out_channel 1
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-1D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 10 \
  --out_channel 1
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-3D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 10 \
  --out_channel 3
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-3D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 10 \
  --out_channel 3
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-5D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 10 \
  --out_channel 5
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-5D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 10 \
  --out_channel 5
  
#=======================================================================================================================

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-10D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir ${TRAIN_DIR_SG} \
  --val_dir ${VAL_DIR_SG} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 10 \
  --out_channel 10
  
# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-10D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/Uformerdeblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_10-10D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 10 \
  --out_channel 10
  
#=======================================================================================================================
