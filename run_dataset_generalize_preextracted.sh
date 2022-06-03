declare -a gpu=$1
declare -a manual_seed=(21)
declare -a checkpoints=(lth_80)
declare -a transfer_dset=$2
declare -a dset_path=$3

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#checkpoints[@]};++i)); 
do
python finetune_pre_extracted_features.py \
	--dataset=${transfer_dset} \
	--dataset_path=${dset_path} \
	--training_config_path="./configs/imagenet_transfer_preextract.yaml" \
        --from_checkpoint_path="../upstream_checkpoints/${checkpoints[i]}.pth" \
	--epochs=150 \
	--batch_size=64 \
	--gpus=${gpu} \
        --seed=${manual_seed[j]}
done
done
