export CONTEXT=headline
export BS=4
export MAX_LENGTH=128

for k in 0 1 2 3 4; do python3 run_edit.py --output_dir /home/nlp/experiments/edit_synthetic/thesis/"lf_"$CONTEXT/$k --per_device_train_batch_size $BS --per_device_eval_batch_size $BS --do_eval --do_train --evaluation_strategy epoch --num_train_epochs 5 --model_name_or_path allenai/longformer-base-4096 --k $k --fold 5 --data_path data/synthetic_thesis.json --context $CONTEXT --max_length $MAX_LENGTH --overwrite_output_dir; done
