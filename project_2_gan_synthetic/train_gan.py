

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.data_loader import load_hf_data
from gan_model import TextGAN, tokens_to_text

def train_gan(train_loader, vocab, config, device):
    """Train GAN model"""
    
    # Create GAN
    gan = TextGAN(len(vocab), config, device)
    
    # Training history
    history = {
        'd_loss': [],
        'g_loss': []
    }
    
    print(f"Training GAN with {len(vocab)} vocabulary size")
    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training GAN")
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(device), labels.to(device)
            
            # Train discriminator multiple times
            for _ in range(config['critic_iterations']):
                d_loss, g_loss = gan.train_step(data, labels)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_Loss': f'{d_loss:.4f}',
                'G_Loss': f'{g_loss:.4f}'
            })
        
        # Average losses
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)
        
        print(f"Epoch {epoch+1} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Generate sample text every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\nGenerating sample texts...")
            sample_tokens, sample_labels = gan.generate_samples(5, temperature=0.8)
            sample_texts = tokens_to_text(sample_tokens, vocab)
            
            for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
                sentiment = "Positive" if label.item() == 1 else "Negative"
                print(f"Sample {i+1} ({sentiment}): {text[:100]}...")
    
    return gan, history

def plot_gan_training(history, save_path=None):
    """Plot GAN training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['g_loss'], label='Generator Loss', color='orange')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading HuggingFace dataset for GAN training...")
    train_loader, val_loader, test_loader, tokenizer, num_classes = load_hf_data(batch_size=GAN_CONFIG['batch_size'])
    
    # Train GAN
    print("Starting GAN training...")
    gan, history = train_gan(train_loader, tokenizer, GAN_CONFIG, device)
    
    # Save models
    gan.save_models(str(GAN_GENERATOR_PATH), str(GAN_DISCRIMINATOR_PATH))
    
    # Plot training history
    plot_path = PROJECT_2_PATH / "gan_training_history.png"
    plot_gan_training(history, str(plot_path))
    
    # Save training history
    history_path = PROJECT_2_PATH / "gan_training_history.json"
    from utils.metrics import save_results
    save_results(history, str(history_path))
    
    print(f"\nGAN training completed!")
    print(f"Generator saved to: {GAN_GENERATOR_PATH}")
    print(f"Discriminator saved to: {GAN_DISCRIMINATOR_PATH}")
    print(f"Training history saved to: {history_path}")
    print(f"Training plot saved to: {plot_path}")

if __name__ == "__main__":
    main() 
