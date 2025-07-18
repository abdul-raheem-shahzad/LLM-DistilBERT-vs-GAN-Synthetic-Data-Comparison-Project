"""
Train LLM on original IMDB dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
from transformers import AutoModelForSequenceClassification, AdamW

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_hf_data
from utils.metrics import evaluate_model, plot_training_history, save_results
from utils.config import *

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (encodings, labels) in enumerate(progress_bar):
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for encodings, labels in val_loader:
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total

def train_llm(train_loader, val_loader, config, device, num_classes):
    """Main training function"""
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained('AirrStorm/DistilBERT-SST2-Yelp', num_labels=num_classes).to(device)
    
    # Loss and optimizer
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training HuggingFace model with {num_classes} classes")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, None, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, None, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), LLM_MODEL_PATH)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(LLM_MODEL_PATH))
    
    return model, history

def main():
    """Main function"""
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading HuggingFace dataset...")
    train_loader, val_loader, test_loader, tokenizer, num_classes = load_hf_data()
    
    # Train model
    print("Starting LLM training...")
    model, history = train_llm(train_loader, val_loader, LLM_CONFIG, device, num_classes)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics, _, _ = evaluate_model(model, test_loader, device, None, tokenizer=tokenizer)
    
    print(f"Test Results:")
    for metric, value in test_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'training_history': history,
        'model_config': LLM_CONFIG,
        'num_classes': num_classes
    }
    
    results_path = PROJECT_1_PATH / "results.json"
    save_results(results, str(results_path))
    
    # Plot training history
    plot_path = PROJECT_1_PATH / "training_history.png"
    plot_training_history(history, str(plot_path))
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {results_path}")
    print(f"Training history plot saved to: {plot_path}")
    print(f"Best model saved to: {LLM_MODEL_PATH}")

if __name__ == "__main__":
    main() 