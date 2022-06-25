python run_ner_crf.py --task_name=jdner \
                      --data_dir datasets/JDNER\
                      --model_type bert\
                      --model_name_or_path model_outputbert/checkpoint-best_model \
                      --tokenizer_name bert-base-chinese \
                      --output_dir model_output\
                      --do_predict \
                      --overwrite_output_dir