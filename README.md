# LLM vs GAN-Synthetic Data Comparison Project

This project compares the performance of a Language Model (LLM) trained on original data versus the same LLM trained on GAN-generated synthetic data.

## Project Structure

```
llm_research/
├── project_1_llm_training/          # Original LLM training
│   ├── data/                        # Original dataset # UPDATE (now it is using AirrStorm/DistilBERT-SST2-Yelp (Yelp/SST2))
│   ├── models/                      # Trained models
│   ├── train_llm.py                 # LLM training script
│   ├── evaluate_llm.py              # Evaluation script
│   └── requirements.txt             # Dependencies
├── project_2_gan_synthetic/         # GAN synthetic data generation
│   ├── data/                        # Synthetic dataset
│   ├── models/                      # GAN models
│   ├── train_gan.py                 # GAN training script
│   ├── generate_synthetic.py        # Synthetic data generation
│   └── requirements.txt             # Dependencies
├── comparison/                      # Comparison analysis
│   ├── compare_results.py           # Comparison script
│   ├── results/                     # Comparison results
│   └── visualizations/              # Charts and graphs
└── utils/                          # Shared utilities
    ├── data_loader.py              # Data loading utilities
    ├── metrics.py                  # Evaluation metrics
    └── config.py                   # Configuration
```

## Dataset: IMDB Movie Reviews

We use the IMDB movie review dataset because:
- Text-based classification (positive/negative sentiment)
- Well-suited for both LLMs and GANs
- Manageable size for training
- Clear evaluation metrics

## Workflow

1. **Project 1**: Train LLM on original IMDB data
2. **Project 2**: Train GAN to generate synthetic IMDB-like reviews
3. **Comparison**: Train same LLM architecture on synthetic data
4. **Analysis**: Compare accuracy, performance, and characteristics

## Setup Instructions

1. Install dependencies for each project:
   ```bash
   cd project_1_llm_training && pip install -r requirements.txt
   cd ../project_2_gan_synthetic && pip install -r requirements.txt
   ```

2. Run the experiments:
   ```bash
   # Train LLM on original data
   cd project_1_llm_training
   python train_llm.py
   python evaluate_llm.py
   
   # Generate synthetic data with GAN
   cd ../project_2_gan_synthetic
   python train_gan.py
   python generate_synthetic.py
   
   # Compare results
   cd ../comparison
   python compare_results.py
   ```

## Expected Outcomes

- Performance comparison between original and synthetic data training
- Analysis of GAN-generated text quality
- Insights into synthetic data augmentation for LLMs 
