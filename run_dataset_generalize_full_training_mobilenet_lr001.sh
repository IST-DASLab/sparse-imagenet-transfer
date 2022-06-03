declare -a gpu=$1
declare -a manual_seed=(21)
declare -a checkpoints=(dense acdc_30)
declare -a transfer_dset=$2
declare -a dset_path=$3

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#checkpoints[@]};++i)); 
do
python main.py \
	--dset=${transfer_dset} \
	--dset_path=${dset_path} \
	--config_path="./configs/mobilenet_full_net_transfer_small_lr.yaml" \
	--workers=4 \
	--epochs=150 \
	--batch_size=64 \
	--from_checkpoint_path="../upstream_checkpoints_mobilenet/${checkpoints[i]}.pth" \
	--gpus=$1 \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_transfer" \
	--transfer_config_path="configs/mobilenet_transfer_fullnetwork_transfer.yaml" \
	--exp_name="${checkpoints[i]}-${transfer_dsets}-small-lr" \
	--use_wandb \
	--wandb_name="${checkpoints[i]}-${transfer_dsets}-small-lr" \
	--wandb_group="${transfer_dsets}" \
        --wandb_project "sparse-mobilenet-transfer-fullnetwork" 
done
done
