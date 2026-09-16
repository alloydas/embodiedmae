# tmux first
tmux new -s "train"

#alloc
salloc -N 1 -n 16 --mem=128G --gres=gpu:l40s:1 --time=4-00:00:00
salloc -N 1 -n 32 --mem=274G --gres=gpu:l40s:2 --time=4-00:00:00
salloc -N 1 -n 32 --mem=228G --gres=gpu:l40s:4 --time=4-00:00:00
salloc -N 1 --ntasks=4 --cpus-per-task=16 --mem=512G --gres=gpu:l40s:4 --time=4-00:00:00
salloc -N 1 -n 32 --mem=228G --gres=gpu:rtx_6000:4 --time=4-00:00:00

#load conda
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /work/mech-ai-scratch/yongyun/envs/myenv

#evaluation
python3 evaluate_zero_shot_official.py  --pretrained_source embodied_mae  --pretrained_path embodiedmae_adapted.pth
python3 evaluate_zero_shot_official.py --pretrained_source embodied_mae  --pretrained_path /work/mech-ai-scratch/alloy/embodiedmae/Dataset/soghumdata_10_small/train/checkpoint_epoch_17100.pth 

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python3 evaluate_zero_shot_official.py --pretrained_source embodied_mae  --pretrained_path /work/mech-ai-scratch/alloy/plant_point_cloud/outputs_sorghum_small_dataset_pretrained/checkpoints/checkpoint_epoch_17100.pth
python3 evaluate_zero_shot_official.py --pretrained_source embodied_mae --pretrained_path /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_old_done_8000/best_model.pth

python3 eval_model.py --checkpoint /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_old_done_8000/best_model.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/old_data/
python3 eval_model.py --checkpoint /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_new_data_8000/checkpoints/checkpoint_epoch_1000.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

# mask_ratio 0.5
python3 eval_model.py --checkpoint /work/mech-ai-scratch/yongyun/plant_point_cloud/outputs_sorghum_new_data_new_loss/best_model.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

# mask_ratio 0.75
python3 eval_model.py --checkpoint /work/mech-ai-scratch/alloy/plant_point_cloud/outputs_sorghum_new_data_new_loss_structure/best_model.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

# Alloy - best pth - mask_ratio 0.15
python3 eval_model_data_population.py --checkpoint /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_new_data_8000-4_pc_loss_10_with_angle/checkpoints/checkpoint_epoch_2400.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

#
python3 eval_model_data_population_all_data.py --checkpoint /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_new_data_8000-4_pc_loss_10_with_angle/checkpoints/checkpoint_epoch_2400.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

#train
python3 train_sorghum.py 

# eval loss

python3 eval_model_loss.py --checkpoint /work/mech-ai-scratch/alloy/embodiedmae/outputs/outputs_sorghum_long_new_data_8000-4_pc_loss_10_with_angle/checkpoints/checkpoint_epoch_2400.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

# eval with my loss
python3 eval_model_loss.py --checkpoint /work/mech-ai-scratch/yongyun/plant_point_cloud/outputs_sorghum_new_data_new_loss_with_coverage_loss/checkpoints/checkpoint_epoch_600.pth --data_root /work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/

python3 eval_model_loss.py --checkpoint /work/mech-ai-scratch/yongyun/plant_point_cloud/outputs_sorghum_new_data_50mask_qal_loss_fastprf/best_model.pth --data_root /work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/test


#train_for_each_loss
tmux new -s "s1"
python3 train_sorghum_for_loss.py

# eval one sample
python eval_one_sample.py   --chamfer_checkpoint ./outputs_sorghum_new_data_50mask_chamfer_fastprf/best_model.pth   --qal_checkpoint ./outputs_sorghum_new_data_50mask_qal_loss_fastprf/best_model.pth   --emd_checkpoint ./outputs_sorghum_new_data_50mask_sinkhorn_fastprf_fullval_num_samples_2048/best_model.pth   --data_root /work/mech-ai-scratch/alloy/plant_point_cloud/Dataset/new_data   --output_dir figs/pc_data   --num_points 8196   --target_suffix _05

