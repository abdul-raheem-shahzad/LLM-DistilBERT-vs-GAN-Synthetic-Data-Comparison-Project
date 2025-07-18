"""
Generate synthetic data using trained GAN
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.data_loader import load_hf_data, save_synthetic_data
from gan_model import TextGAN, tokens_to_text

def generate_synthetic_dataset(gan, vocab, num_samples: int, device):
    """Generate synthetic dataset"""
    
    print(f"Generating {num_samples} synthetic samples...")
    
    # Generate samples in batches
    batch_size = 64
    all_texts = []
    all_labels = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
        # Calculate batch size for this iteration
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Generate random labels (balanced)
        labels = []
        for i in range(current_batch_size):
            if i < current_batch_size // 2:
                labels.append(0)  # Negative
            else:
                labels.append(1)  # Positive
        
        # Generate samples
        tokens, labels_tensor = gan.generate_samples(
            current_batch_size, 
            labels=labels, 
            temperature=0.8
        )
        
        # Convert to text
        texts = tokens_to_text(tokens, vocab)
        
        all_texts.extend(texts)
        all_labels.extend(labels_tensor.cpu().numpy())
    
    # Create dataset
    synthetic_data = list(zip(all_texts, all_labels))
    
    return synthetic_data

def evaluate_synthetic_quality(synthetic_data, original_data_loader, vocab):
    """Evaluate quality of synthetic data"""
    
    print("Evaluating synthetic data quality...")
    
    # Basic statistics
    synthetic_texts, synthetic_labels = zip(*synthetic_data)
    
    # Length statistics
    lengths = [len(text.split()) for text in synthetic_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Label distribution
    label_counts = np.bincount(synthetic_labels)
    
    # Vocabulary coverage
    all_words = set()
    for text in synthetic_texts:
        all_words.update(text.split())
    
    vocab_coverage = len(all_words) / len(vocab)
    
    # Compare with original data
    original_lengths = []
    original_labels = []
    
    for batch in original_data_loader:
        data, labels = batch
        # Convert back to text for length calculation
        for seq, label in zip(data, labels):
            text = tokens_to_text([seq], vocab)[0]
            original_lengths.append(len(text.split()))
            original_labels.append(label.item())
    
    original_avg_length = np.mean(original_lengths)
    original_std_length = np.std(original_lengths)
    original_label_counts = np.bincount(original_labels)
    
    quality_metrics = {
        'synthetic': {
            'avg_length': avg_length,
            'std_length': std_length,
            'label_distribution': label_counts.tolist(),
            'vocab_coverage': vocab_coverage
        },
        'original': {
            'avg_length': original_avg_length,
            'std_length': original_std_length,
            'label_distribution': original_label_counts.tolist()
        },
        'comparison': {
            'length_similarity': 1 - abs(avg_length - original_avg_length) / original_avg_length,
            'distribution_similarity': 1 - abs(label_counts[1] - original_label_counts[1]) / original_label_counts[1]
        }
    }
    
    return quality_metrics

def main():
    """Main function"""
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load original data and vocabulary
    print("Loading HuggingFace dataset...")
    train_loader, val_loader, test_loader, tokenizer, num_classes = load_hf_data()
    
    # Load trained GAN
    print("Loading trained GAN...")
    gan = TextGAN(tokenizer.vocab_size, GAN_CONFIG, device)
    gan.load_models(str(GAN_GENERATOR_PATH), str(GAN_DISCRIMINATOR_PATH))
    
    # Generate synthetic dataset
    num_synthetic_samples = 10000  # Generate 10k synthetic samples
    synthetic_data = generate_synthetic_dataset(gan, tokenizer, num_synthetic_samples, device)
    
    # Save synthetic data
    synthetic_data_path = PROJECT_2_PATH / "data" / "synthetic_imdb.txt"
    save_synthetic_data(synthetic_data, str(synthetic_data_path))
    
    # Evaluate quality
    quality_metrics = evaluate_synthetic_quality(synthetic_data, train_loader, tokenizer)
    
    # Save quality metrics
    quality_path = PROJECT_2_PATH / "synthetic_quality.json"
    from utils.metrics import save_results
    save_results(quality_metrics, str(quality_path))
    
    # Print quality report
    print("\nSynthetic Data Quality Report:")
    print("=" * 50)
    
    print(f"Synthetic samples generated: {len(synthetic_data)}")
    print(f"Average text length: {quality_metrics['synthetic']['avg_length']:.1f} words")
    print(f"Label distribution: {quality_metrics['synthetic']['label_distribution']}")
    print(f"Vocabulary coverage: {quality_metrics['synthetic']['vocab_coverage']:.3f}")
    
    print(f"\nComparison with original data:")
    print(f"Length similarity: {quality_metrics['comparison']['length_similarity']:.3f}")
    print(f"Distribution similarity: {quality_metrics['comparison']['distribution_similarity']:.3f}")
    
    # Show some examples
    print(f"\nSample synthetic texts:")
    print("=" * 50)
    for i, (text, label) in enumerate(synthetic_data[:5]):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"Sample {i+1} ({sentiment}): {text[:100]}...")
    
    print(f"\nSynthetic data generation completed!")
    print(f"Synthetic data saved to: {synthetic_data_path}")
    print(f"Quality metrics saved to: {quality_path}")

if __name__ == "__main__":
    main() 