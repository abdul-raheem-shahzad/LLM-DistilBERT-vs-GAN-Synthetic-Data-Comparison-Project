"""
Compare results between LLM trained on original vs synthetic data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AdamW

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *
from utils.data_loader import load_hf_data, load_synthetic_data
from utils.metrics import (
    evaluate_model, plot_confusion_matrix, compare_models_results, 
    generate_comparison_report, save_results
)

def train_llm_on_synthetic_data(synthetic_data, config, device, num_classes, tokenizer):
    """Train LLM on synthetic data"""
    
    print("Training LLM on synthetic data...")
    
    synthetic_dataset = synthetic_data
    
    # Split into train/val
    train_size = int(0.8 * len(synthetic_dataset))
    val_size = len(synthetic_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.long)
        return encodings, labels
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained('AirrStorm/DistilBERT-SST2-Yelp', num_labels=num_classes).to(device)
    
    # Loss and optimizer
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for encodings, labels in train_loader:
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
            train_total += labels.size(0)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for encodings, labels in val_loader:
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(labels.view_as(pred)).sum().item()
                val_total += labels.size(0)
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    return model, history

def main():
    """Main comparison function"""
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load original data
    print("Loading HuggingFace dataset...")
    train_loader, val_loader, test_loader, tokenizer, num_classes = load_hf_data()
    
    # Load original LLM results
    print("Loading original LLM results...")
    original_results_path = PROJECT_1_PATH / "results.json"
    from utils.metrics import load_results
    original_results = load_results(str(original_results_path))
    
    # Load synthetic data
    print("Loading synthetic data...")
    synthetic_data_path = PROJECT_2_PATH / "data" / "synthetic_imdb.txt"
    synthetic_data = load_synthetic_data(str(synthetic_data_path), tokenizer)
    
    # Train LLM on synthetic data
    synthetic_model, synthetic_history = train_llm_on_synthetic_data(
        synthetic_data, LLM_CONFIG, device, num_classes, tokenizer
    )
    
    # Evaluate synthetic-trained model on test set
    print("Evaluating synthetic-trained model on test set...")
    synthetic_test_metrics, synthetic_predictions, synthetic_probabilities = evaluate_model(
        synthetic_model, test_loader, device, None, tokenizer=tokenizer
    )
    
    # Load original model for comparison
    print("Loading original trained model...")
    original_model = AutoModelForSequenceClassification.from_pretrained('AirrStorm/DistilBERT-SST2-Yelp', num_labels=num_classes).to(device)
    original_model.load_state_dict(torch.load(LLM_MODEL_PATH, map_location=device))
    
    # Evaluate original model on test set
    print("Evaluating original model on test set...")
    original_test_metrics, original_predictions, original_probabilities = evaluate_model(
        original_model, test_loader, device, None, tokenizer=tokenizer
    )
    
    # Create comparison results
    comparison_results = {
        'original_data_training': original_test_metrics,
        'synthetic_data_training': synthetic_test_metrics
    }
    
    # Generate comprehensive report
    print("Generating comparison report...")
    report_path = COMPARISON_PATH / "results" / "comparison_report.json"
    comparison_report = generate_comparison_report(
        original_test_metrics, 
        synthetic_test_metrics, 
        str(report_path)
    )
    
    # Create visualizations
    print("Creating comparison visualizations...")
    
    # 1. Model performance comparison
    comparison_plot_path = COMPARISON_PATH / "visualizations" / "model_comparison.png"
    compare_models_results(comparison_results, str(comparison_plot_path))
    
    # 2. Confusion matrices
    cm_original = np.array(original_test_metrics['confusion_matrix'])
    cm_synthetic = np.array(synthetic_test_metrics['confusion_matrix'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original model confusion matrix
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax1.set_title('Original Data Training')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Synthetic model confusion matrix
    sns.heatmap(cm_synthetic, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax2.set_title('Synthetic Data Training')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    cm_plot_path = COMPARISON_PATH / "visualizations" / "confusion_matrices.png"
    plt.savefig(str(cm_plot_path), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Training history comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original training history
    original_history = original_results['training_history']
    ax1.plot(original_history['train_loss'], label='Train Loss')
    ax1.plot(original_history['val_loss'], label='Val Loss')
    ax1.set_title('Original Data Training - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(original_history['train_acc'], label='Train Acc')
    ax2.plot(original_history['val_acc'], label='Val Acc')
    ax2.set_title('Original Data Training - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Synthetic training history
    ax3.plot(synthetic_history['train_loss'], label='Train Loss')
    ax3.plot(synthetic_history['val_loss'], label='Val Loss')
    ax3.set_title('Synthetic Data Training - Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(synthetic_history['train_acc'], label='Train Acc')
    ax4.plot(synthetic_history['val_acc'], label='Val Acc')
    ax4.set_title('Synthetic Data Training - Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    history_plot_path = COMPARISON_PATH / "visualizations" / "training_history_comparison.png"
    plt.savefig(str(history_plot_path), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nOriginal Data Training Results:")
    for metric, value in original_test_metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nSynthetic Data Training Results:")
    for metric, value in synthetic_test_metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nImprovements:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        original_val = original_test_metrics[metric]
        synthetic_val = synthetic_test_metrics[metric]
        improvement = synthetic_val - original_val
        improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0
        print(f"  {metric}: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    print(f"\nResults saved to:")
    print(f"  Comparison report: {report_path}")
    print(f"  Model comparison plot: {comparison_plot_path}")
    print(f"  Confusion matrices: {cm_plot_path}")
    print(f"  Training history comparison: {history_plot_path}")

if __name__ == "__main__":
    main() 