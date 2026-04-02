#!/bin/bash
export PATH="/home/mt3/documents/iMT3/.conda/envs/mrmt3/bin:$PATH"

HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python -u train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0] \
    model="MT3NetSegMemV2WithPrev" \
    dataset="SlakhMini" \
    dataset_use_tf_spectral_ops=False \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    dataset_prev_augment_frames=3 \
    trainer.strategy="ddp_find_unused_parameters_true" \
    trainer.check_val_every_n_epoch=1 \
    eval.eval_after_num_epoch=0 \
    eval.eval_first_n_examples=1 \
    eval.eval_per_epoch=1 \
    trainer.max_epochs=1 \
    eval.audio_dir="/home/mt3/documents/iMT3/data/slakh2100_flac_redux_baby/test/*/mix_16k.wav" \
    eval.midi_dir="/home/mt3/documents/iMT3/data/slakh2100_flac_redux_baby/test/" \
    eval.contiguous_inference=True
