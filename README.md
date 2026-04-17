# moe-surgeon

CLI tool for analyzing and pruning Mixture-of-Experts (MoE) models.

Target: Gemma 4 26B-A4B (128 experts, 8 active per token).

## Goal

Find dead-weight experts in MoE models and remove them to reduce model size without quality loss.
