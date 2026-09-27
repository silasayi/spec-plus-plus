# SPEC++ Paired-Reasoning Benchmark

This repository contains the data, results, and code accompanying the paper
"An Analysis of Adaptive Hard Negative Mining for Fine Grained Vision Language
Understanding" (Discover Computing, under review).

## Contents

 spec_plus_plus_pairs.json: The SPEC++ paired-reasoning benchmark, 200 pairs
  across 60 object classes.
 specpp_in_distribution_results.json: Fine-tuned CLIP accuracy on SPEC
  held-out test sets (per subset and mean).
 specpp_adaptive_results.json: Adaptive hard-negative training results on
  SPEC held-out test sets.
 train.py: Training and evaluation code (see below).

## How to Reproduce

1. Download the SPEC dataset from https://huggingface.co/datasets/wjpoom/SPEC
2. Install dependencies: `pip install open_clip_torch timm`
3. Run `python train.py` (see comments for configuration).

## Model weights

The fine-tuned model weights (specpp_in_distribution_model.pt) are
approximately 577 MB and exceed GitHub's per-file size limit. They are
available on request from the corresponding author. To reproduce
training from scratch, run train.py after downloading the SPEC dataset.

## Contact

Silas Ayitey, Kwame Nkrumah University of Science and Technology,
silasayitey@gmail.com
