# transformer-editorial



```
$ python3 run_edit.py --model_name_or_path allenai/longformer-base-4096 --output_dir /home/nlp/experiments/edit --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --do_eval --do_train --evaluation_strategy epoch --num_train_epochs 3 --k 2 --fold 5
```
