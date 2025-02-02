python3 run_semantic_segmentation.py \
    --model_name_or_path nvidia/mit-b2 \
    --dataset_name Sowmith1999/agaid_residue_only \
    --output_dir ./mit-b2-crop_v1/ \
    --remove_unused_columns False \
    # --do-reduce-labels \
    --do_train \
    --do_eval \
    --eval_on_start True \
    # --push_to_hub \
    # --push_to_hub_model_id segformer-finetuned-sidewalk-10k-steps \
    # --max_steps 10000 \
    --max_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 0.00006 \
    --lr_scheduler_type polynomial \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy best \
    --metric_for_best_model "eval_iou_residue" \
    --load_best_model_at_end True \
    --seed 1337
