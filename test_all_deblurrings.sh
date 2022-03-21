'/mnt/data/data/data_Xareni/subset_hr_res/256px_uint16/train/imgs'

# UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64

python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --input_dir /mnt/data/data/data_Xareni/subset_hr_res/256px_uint16/train/imgs/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
   --input_dir /mnt/data/data/data_Xareni/data/features_sox/ \
   --results_dir /mnt/data/data/data_Xareni/data/features_sox/ \
   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
   --embed_dim 16 \
   --train_ps 64 \
   --save_images \
   --arch UNet \
   --in_channel 1 \
   --out_channel 1


########################################################################################################################
# DEBLURRINGS FOR EVERY EXPERIMENT AND manually_selected_5-2/stacks/blurry DATA
########################################################################################################################

## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 1 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 1 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 3 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 3 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 3 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 3 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 5
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_5-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 5 \
#   --out_channel 5
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 1
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-3D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 3
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 5
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-5D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 5
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 10
#
## UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64
#
#python3 ./test_in_any_resolution_stack.py \
#   --input_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/blurry/ \
#   --gt_dir /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/clean/ \
#   --results_dir /home/felix/Scratch/nus/20b_deblurring_uformer_nD/results/ \
#   --weights /home/felix/Scratch/nus/20b_deblurring_uformer_nD/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_10-10D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64/models/model_best_PSNR_SSIM.pth \
#   --embed_dim 16 \
#   --train_ps 64 \
#   --save_images \
#   --arch UNet \
#   --in_channel 10 \
#   --out_channel 10
