#!/bin/bash

echo "Starting training on the full Slakh2100 dataset :o"

HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0] \
    model="MT3NetSegMemV2WithPrev" \
    dataset="SlakhFull" \
    dataset_use_tf_spectral_ops=False \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    dataset_prev_augment_frames=3 \
    trainer.strategy="ddp_find_unused_parameters_true" \
    trainer.check_val_every_n_epoch=5 \
    eval.eval_after_num_epoch=5 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=2 \
    eval.audio_dir="/mnt/e/archive/slakh2100_flac_redux/test/*/mix_16k.wav" \
    eval.midi_dir="/mnt/e/archive/slakh2100_flac_redux/test/" \
    eval.contiguous_inference=True
