"""
Evaluation metrics for LLM and GAN comparison
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_proba: torch.Tensor = None) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    
    # Convert to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Basic metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    # Additional metrics if probabilities are provided
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    if y_proba is not None:
        y_proba_np = y_proba.cpu().numpy()
        # Calculate AUC if we have probabilities
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true_np, y_proba_np[:, 1])
            metrics['auc'] = auc
        except:
            pass
    
    return metrics

def evaluate_model(model, data_loader, device, criterion=None, tokenizer=None):
    """Evaluate HuggingFace model on given data loader"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for encodings, labels in data_loader:
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            if loss is not None:
                total_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    if criterion is not None or (loss is not None):
        metrics['loss'] = total_loss / len(data_loader)
    return metrics, all_predictions, all_probabilities

def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history: Dict, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_models_results(results_dict: Dict, save_path=None):
    """Compare results from different models"""
    
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        axes[i].bar(models, values, color=['skyblue', 'lightcoral'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results: Dict, filepath: str):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    results_copy = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_copy[key] = value.tolist()
        elif isinstance(value, dict):
            results_copy[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_copy[key][k] = v.tolist()
                else:
                    results_copy[key][k] = v
        else:
            results_copy[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=4)

def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_comparison_report(original_results: Dict, synthetic_results: Dict, save_path: str):
    """Generate comprehensive comparison report"""
    
    report = {
        "experiment_info": {
            "original_data_training": original_results,
            "synthetic_data_training": synthetic_results
        },
        "comparison": {}
    }
    
    # Compare key metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        original_val = original_results.get(metric, 0)
        synthetic_val = synthetic_results.get(metric, 0)
        
        improvement = synthetic_val - original_val
        improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0
        
        report["comparison"][metric] = {
            "original": original_val,
            "synthetic": synthetic_val,
            "improvement": improvement,
            "improvement_percentage": improvement_pct
        }
    
    # Save report
    save_results(report, save_path)
    
    return report 