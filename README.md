# Uncertainty_Estimation
# Train

python train.py \
  --model iterative \
  --dataset "RT" \
  --dataset_path "/home/ltnghia02/MEDICAL_ITERATIVE/dataset/RTdata_Crop_1024" \
  --dropout_rate 0.0 \
  --image_channel 3 \
  --use_batchnorm \
  --add_channel \
  --batch_size 15 \
  --learning_rate 0.001 \
  --num_epoch 30 \
  --save_path "/home/ltnghia02/MEDICAL_ITERATIVE/checkpoints/RT_iter_batch" \
  --save_per_epoch 5 \
  --loss_function "focal" \
  --buffer_size  10\
  --gpus 5,6,4

# Predict

# Eval