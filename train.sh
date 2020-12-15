GPUS='0,1,2,3';
BATCH_SIZE_PER_GPU=8;
MAX_EPOCHS=300;
LATENT_SIZE=1024;
TRAIN_DATA=28000;
VAL_DATA=1000;
TEST_DATA=1000;
NORM_LAYER_TYPE='spectral_norm';
LOSSES='ssim,lpips'
LEARNING_RATE='0.01';
NUM_SAMPLE=16;
LOG_NAME='test';
VERSION='1215-MDSlatent-tri-neq-1';
LOG_SAMPLE_EVERY=2;
ROOT_DIR='~/data/FFHQ';
LATENT_PATH='MDS_feat_30000.npy';
TARGET_DIR='images256x256';
N_WORKERS=8;
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
--norm_type ${NORM_LAYER_TYPE} \
--losses %{LOSSES} \
--lr ${LEARNING_RATE} \
--num_sample ${NUM_SAMPLE} \
--log_name ${LOG_NAME} \
--version ${VERSION} \
--log_sample_every ${LOG_SAMPLE_EVERY} \
--log_every_n_steps 50 \
--flush_logs_every_n_steps 500 

# --sync_batchnorm
# --resume_from_checkpoint

