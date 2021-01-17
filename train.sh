GPUS='0,1,2,3';
BATCH_SIZE_PER_GPU=1024;
MAX_EPOCHS=5;
LATENT_SIZE=512;
TRAIN_DATA=32000;
VAL_DATA=1024;
TEST_DATA=1024;
LOSSES='ssim,lpips';
LEARNING_RATE='0.001';
NUM_SAMPLE=16;
LOG_NAME='test'; # lpips-tri-neq
VERSION='test-big-bs'; # mse-a1-fr1-after1-2
LOG_SAMPLE_EVERY=2;
ROOT_DIR='~/data/FFHQ';
LATENT_PATH='FFHQ_MDS_feat-cropface_70000.npy';
TARGET_DIR='images256x256-cropface';
N_WORKERS=4;
LOG_EVERY_N_STEPS=50;
FLUSH_LOGS_EVERY_N_STEPS=500;
 \
python $1 \
--root_dir ${ROOT_DIR} \
--latent_path ${LATENT_PATH} \
--target_dir ${TARGET_DIR} \
--num_workers ${N_WORKERS} \
--gpus ${GPUS} \
--bs_per_gpu ${BATCH_SIZE_PER_GPU} \
--max_epochs ${MAX_EPOCHS} \
--latent_dim ${LATENT_SIZE} \
--train_size ${TRAIN_DATA} \
--val_size ${VAL_DATA} \
--test_size ${TEST_DATA} \
--losses ${LOSSES} \
--lr ${LEARNING_RATE} \
--num_sample ${NUM_SAMPLE} \
--log_name ${LOG_NAME} \
--version ${VERSION} \
--log_sample_every ${LOG_SAMPLE_EVERY} \
--log_every_n_steps ${LOG_EVERY_N_STEPS} \
--flush_logs_every_n_steps ${FLUSH_LOGS_EVERY_N_STEPS}


# --sync_batchnorm
# --resume_from_checkpoint

