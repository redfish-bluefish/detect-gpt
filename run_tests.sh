#!/bin/bash

# Baseline comparisons
# python run.py --output_name baseline --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --chunk_size 40
python run.py --output_name baseline --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --chunk_size 40
python run.py --output_name baseline --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --chunk_size 40


# SQUAD
python run.py --output_name squad --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context --cache_dir ./.cache --skip_baselines --chunk_size 40
python run.py --output_name squad --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context --cache_dir ./.cache --skip_baselines --chunk_size 40
python run.py --output_name squad --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset squad --dataset_key context --cache_dir ./.cache --skip_baselines --chunk_size 40

# WritingPrompts
python run.py --output_name baseline --base_model_name gpt2-xl --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --dataset writing --chunk_size 40
python run.py --output_name baseline --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --dataset writing --chunk_size 40
python run.py --output_name baseline --base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --cache_dir ./.cache --skip_baselines --dataset writing --chunk_size 40

