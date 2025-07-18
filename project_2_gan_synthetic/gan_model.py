"""
GAN models for synthetic text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import random

class Generator(nn.Module):
    """Generator for text GAN"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, vocab_size: int, max_length: int, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.device = device
        
        # LSTM-based generator
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Label embedding for conditional generation
        self.label_embedding = nn.Embedding(2, latent_dim)
        
    def forward(self, z, labels=None, temperature=1.0):
        """
        Generate text from noise
        z: (batch_size, latent_dim)
        labels: (batch_size,) - 0 for negative, 1 for positive
        """
        batch_size = z.size(0)
        
        # If labels provided, condition on them
        if labels is not None:
            label_emb = self.label_embedding(labels)
            z = z + label_emb
        
        # Expand z to sequence length
        z = z.unsqueeze(1).expand(-1, self.max_length, -1)
        
        # Generate sequence
        lstm_out, _ = self.lstm(z)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_out)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        
        # Sample tokens
        tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
        tokens = tokens.view(batch_size, self.max_length)
        
        return tokens, logits

class Discriminator(nn.Module):
    """Discriminator for text GAN"""
    
    def __init__(self, vocab_size: int, hidden_dim: int, max_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM-based discriminator
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len) - token indices
        """
        # Embed tokens
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden state
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concatenate forward and backward
        
        # Classify
        output = self.classifier(final_hidden)
        
        return output.squeeze()

class TextGAN:
    """Text GAN wrapper class"""
    
    def __init__(self, vocab_size: int, config: dict, device):
        self.vocab_size = vocab_size
        self.config = config
        self.device = device
        
        # Create models
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            hidden_dim=config['generator_hidden'],
            vocab_size=vocab_size,
            max_length=MAX_SEQUENCE_LENGTH,
            device=device
        ).to(device)
        
        self.discriminator = Discriminator(
            vocab_size=vocab_size,
            hidden_dim=config['discriminator_hidden'],
            max_length=MAX_SEQUENCE_LENGTH
        ).to(device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate_g'],
            betas=(0.5, 0.999)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['learning_rate_d'],
            betas=(0.5, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def train_step(self, real_data, real_labels):
        """Single training step"""
        batch_size = real_data.size(0)
        
        # Labels
        real_labels_tensor = torch.ones(batch_size).to(self.device)
        fake_labels_tensor = torch.zeros(batch_size).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_data)
        d_real_loss = self.criterion(real_output, real_labels_tensor)
        
        # Fake data
        z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
        fake_data, _ = self.generator(z, real_labels)
        fake_output = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels_tensor)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
        fake_data, _ = self.generator(z, real_labels)
        fake_output = self.discriminator(fake_data)
        
        g_loss = self.criterion(fake_output, real_labels_tensor)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_samples(self, num_samples: int, labels: List[int] = None, temperature: float = 1.0):
        """Generate synthetic samples"""
        self.generator.eval()
        
        if labels is None:
            labels = [random.randint(0, 1) for _ in range(num_samples)]
        
        labels_tensor = torch.tensor(labels).to(self.device)
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['latent_dim']).to(self.device)
            tokens, _ = self.generator(z, labels_tensor, temperature)
        
        return tokens, labels_tensor
    
    def save_models(self, generator_path: str, discriminator_path: str):
        """Save models"""
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
    
    def load_models(self, generator_path: str, discriminator_path: str):
        """Load models"""
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))

def tokens_to_text(tokens, vocab, max_length=MAX_SEQUENCE_LENGTH):
    """Convert token indices back to text"""
    # Create reverse vocabulary
    idx_to_token = {idx: token for token, idx in vocab.get_stoi().items()}
    
    texts = []
    for seq in tokens:
        text_tokens = []
        for idx in seq:
            if idx.item() == vocab['<pad>']:
                break
            if idx.item() in idx_to_token:
                text_tokens.append(idx_to_token[idx.item()])
            else:
                text_tokens.append('<unk>')
        texts.append(' '.join(text_tokens))
    
    return texts 