export GPUS=2
export BATCH_SIZE_PER_GPU=64
export MAX_EPOCHS=100
export LATENT_SIZE=512
export TRAIN_DATA=32000
export VAL_DATA=32000
export LEARNING_RATE='0.02'
export NUM_SAMPLE=16
export LOG_NAME='batch-size'
export VERSION='128x250imgs'
export LOG_SAMPLE_EVERY=10

python $1 \
--gpus ${GPUS} \
--batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
--max_epochs ${MAX_EPOCHS} \
--latent_size ${LATENT_SIZE} \
--train_size ${TRAIN_DATA} \
--val_size ${VAL_DATA} \
--lr ${LEARNING_RATE} \
--num_sample ${NUM_SAMPLE} \
--log_name ${LOG_NAME} \
--version ${VERSION} \
--log_sample_every ${LOG_SAMPLE_EVERY}
