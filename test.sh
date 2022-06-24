python code/predict.py --test_data_file_path="$1" \
                       --best_model_name data/best_model \
                       --result_save_path data/submission \
                       --result_name results.txt \
                       --cuda_device 0 \
                       --seed 42