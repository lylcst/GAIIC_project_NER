python main.py \
  --train_data_dir data/pretrained_unlabel_data_400000.json \
  --eval_data_dir data/eval_unlabeled_data_2000.json \
  --model_dir further_pretraining \
  --max_steps 200000 \
  --model_name_or_path pretrained_model/nezha-cn-base \
  --save_steps 100000 \
  --logging_steps 1000 \
  --cuda_device 1 \
  --do_eval
  # --num_train_epochs -1 