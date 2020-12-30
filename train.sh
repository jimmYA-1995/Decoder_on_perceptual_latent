GPUS='0,1';
BATCH_SIZE_PER_GPU=16;
MAX_EPOCHS=500;
LATENT_SIZE=1024;
TRAIN_DATA=32000;
VAL_DATA=10000;
TEST_DATA=1000;
NORM_LAYER_TYPE='spectral_norm';
LOSSES='ssim,lpips';
LEARNING_RATE='0.001';
LR_SCHEDULER='None';
NUM_SAMPLE=16;
LOG_NAME='mds1024-cropface'; # lpips-tri-neq
VERSION='embed_tri-ineq-after5'; # mse-a1-fr1-after1-2
LOG_SAMPLE_EVERY=2;
ROOT_DIR='~/data/FFHQ';
LATENT_PATH='FFHQ_MDS_feat-cropface_70000.npy'; # 'FFHQ_MDS_feat-cropface_70000.npy';
TARGET_DIR='images256x256-cropface';
N_WORKERS=8;
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
--norm_type ${NORM_LAYER_TYPE} \
--losses ${LOSSES} \
--lr ${LEARNING_RATE} \
--lr_scheduler ${LR_SCHEDULER} \
--num_sample ${NUM_SAMPLE} \
--log_name ${LOG_NAME} \
--version ${VERSION} \
--log_sample_every ${LOG_SAMPLE_EVERY} \
--log_every_n_steps ${LOG_EVERY_N_STEPS} \
--flush_logs_every_n_steps ${FLUSH_LOGS_EVERY_N_STEPS} \
--resume_from_checkpoint 'runs/mds1024-cropface/embed_1e-3_ft1/checkpoints/epoch=399.ckpt'

# --sync_batchnorm
# --resume_from_checkpoint

