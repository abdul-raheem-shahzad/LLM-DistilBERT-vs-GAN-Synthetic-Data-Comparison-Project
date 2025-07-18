"""
Configuration settings for LLM vs GAN comparison project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_1_PATH = PROJECT_ROOT / "project_1_llm_training"
PROJECT_2_PATH = PROJECT_ROOT / "project_2_gan_synthetic"
COMPARISON_PATH = PROJECT_ROOT / "comparison"

# Data settings
DATASET_NAME = "imdb"
MAX_SEQUENCE_LENGTH = 128
VOCAB_SIZE = 10000
BATCH_SIZE = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# LLM settings
LLM_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_heads": 2,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "epochs": 10,
    "early_stopping_patience": 3
}

# GAN settings
GAN_CONFIG = {
    "latent_dim": 100,
    "generator_hidden": 256,
    "discriminator_hidden": 256,
    "learning_rate_g": 2e-4,
    "learning_rate_d": 2e-4,
    "epochs": 50,
    "batch_size": 64,
    "critic_iterations": 5
}

# Training settings
RANDOM_SEED = 42
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Model save paths
LLM_MODEL_PATH = PROJECT_1_PATH / "models" / "llm_model.pth"
GAN_GENERATOR_PATH = PROJECT_2_PATH / "models" / "gan_generator.pth"
GAN_DISCRIMINATOR_PATH = PROJECT_2_PATH / "models" / "gan_discriminator.pth"

# Results paths
RESULTS_PATH = COMPARISON_PATH / "results"
VISUALIZATIONS_PATH = COMPARISON_PATH / "visualizations"

# Create directories if they don't exist
for path in [RESULTS_PATH, VISUALIZATIONS_PATH]:
    path.mkdir(parents=True, exist_ok=True) 