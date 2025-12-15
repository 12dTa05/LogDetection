"""
Deep Learning Models for Log Anomaly Detection
Matches original implementation with ForecastBasedModel base class,
CNN with Conv2d, LSTM with Attention, and fit() method.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Attention(nn.Module):
    """
    Attention mechanism for LSTM.
    Matches original: self.attn = Attention(hidden_size * num_directions, window_size)
    """
    
    def __init__(self, hidden_size: int, window_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
        """
        # Attention weights
        attn_weights = self.attention(lstm_output)  # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class ForecastBasedModel(nn.Module):
    """
    Base class for forecast-based anomaly detection models.
    Includes fit() method matching original implementation.
    """
    
    def __init__(
        self,
        meta_data: Dict[str, Any],
        model_save_path: str = "./stable_model",
        feature_type: str = "sequentials",
        label_type: str = "anomaly",
        eval_type: str = "session",
        topk: int = 0,
        use_tfidf: bool = False,
        embedding_dim: int = 16,
        freeze: bool = False,
        gpu: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.meta_data = meta_data
        self.model_save_path = model_save_path
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.topk = topk
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        self.gpu = gpu
        
        # Extract from meta_data
        self.num_labels = meta_data.get("num_labels", meta_data.get("vocab_size", 30))
        self.vocab_size = meta_data.get("vocab_size", 30)
        self.window_size = meta_data.get("window_size", 30)
        
        # Device
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def save_model(self, path: Optional[str] = None):
        """Save model to file."""
        path = path or self.model_save_path
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: Optional[str] = None):
        """Load model from file."""
        path = path or self.model_save_path
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.load_state_dict(state['model_state_dict'])
        else:
            self.load_state_dict(state)
    
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        epochs: int = 100,
        learning_rate: float = 0.01,
        patience: int = 3,
        **kwargs
    ) -> Tuple[Dict, List]:
        """
        Train the model.
        
        Matches original:
        best_result, history = model.fit(
            train_loader=dataloader_train,
            epochs=params["epoches"],
            learning_rate=params["learning_rate"],
        )
        """
        self.to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_f1 = -1
        best_epoch = 0
        patience_counter = 0
        history = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            self.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                loss, logits = self.forward(batch_x, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader)
            
            # Evaluate
            if test_loader:
                acc, prec, rec, f1 = self.evaluate(test_loader)
            else:
                acc, prec, rec, f1 = 0, 0, 0, 0
            
            epoch_time = time.time() - epoch_start
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
            })
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Acc: {acc:.4f} | F1: {f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        print("-" * 60)
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best F1 Score: {best_f1:.4f} at epoch {best_epoch}")
        print(f"Model saved to: {self.model_save_path}")
        
        best_result = {
            'accuracy': acc,
            'f1': best_f1,
            'best_epoch': best_epoch,
        }
        
        return best_result, history
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Evaluate model on test set."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.forward(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return acc, prec, rec, f1


class Transformer(ForecastBasedModel):
    """
    Transformer model for log anomaly detection.
    """
    
    def __init__(
        self,
        meta_data: Dict[str, Any],
        embedding_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        hidden_size: int = 512,
        dropout: float = 0.1,
        model_save_path: str = "./transformer_models",
        feature_type: str = "sequentials",
        label_type: str = "anomaly",
        eval_type: str = "session",
        topk: int = 0,
        use_tfidf: bool = False,
        freeze: bool = False,
        gpu: int = 1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        
        self.num_labels = meta_data.get("num_labels", meta_data.get("vocab_size", 30))
        self.feature_type = feature_type
        self.hidden_size = hidden_size
        self.use_tfidf = use_tfidf
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights()
    
    def forward(self, x, labels=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        padding_mask = (x.sum(dim=-1) == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1)
        
        logits = self.fc(x)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        
        return logits


class LSTM(ForecastBasedModel):
    """
    LSTM model for log anomaly detection.
    
    Matches original:
    class LSTM(ForecastBasedModel):
        def __init__(self, meta_data, hidden_size=100, num_directions=2,
                     num_layers=1, window_size=None, use_attention=False,
                     embedding_dim=16, model_save_path="./lstm_models", ...)
    """
    
    def __init__(
        self,
        meta_data: Dict[str, Any],
        hidden_size: int = 100,
        num_directions: int = 2,
        num_layers: int = 1,
        window_size: int = None,
        use_attention: bool = False,
        embedding_dim: int = 16,
        model_save_path: str = "./lstm_models",
        feature_type: str = "sequentials",
        label_type: str = "next_log",
        eval_type: str = "session",
        topk: int = 5,
        use_tfidf: bool = False,
        freeze: bool = False,
        gpu: int = -1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        
        self.num_labels = meta_data.get("num_labels")
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size or meta_data.get("window_size", 30)
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2)
        )
        
        # Attention (optional)
        if self.use_attention:
            assert window_size is not None, "Window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, self.window_size)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Prediction layer
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions,
            self.num_labels
        )
        
        self.init_weights()
    
    def forward(self, x, labels=None):
        x = self.embedding(x)  # (batch, seq, emb)
        
        # LSTM
        output, (hidden, cell) = self.rnn(x)  # output: (batch, seq, hidden * num_dir)
        
        if self.use_attention:
            # Use attention mechanism
            representation = self.attn(output)
        else:
            # Use last hidden state
            if self.num_directions == 2:
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]
            representation = hidden
        
        logits = self.prediction_layer(representation)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        
        return logits


class CNN(ForecastBasedModel):
    """
    CNN model for log anomaly detection.
    
    Matches original:
    class CNN(ForecastBasedModel):
        def __init__(self, meta_data, kernel_sizes=[2, 3, 4],
                     hidden_size=100, embedding_dim=16,
                     model_save_path="./cnn_models", ...)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, hidden_size, (K, embedding_dim)) for K in kernel_sizes
        ])
    """
    
    def __init__(
        self,
        meta_data: Dict[str, Any],
        kernel_sizes: List[int] = [2, 3, 4],
        hidden_size: int = 100,
        embedding_dim: int = 16,
        model_save_path: str = "./cnn_models",
        feature_type: str = "sequentials",
        label_type: str = "anomaly",
        eval_type: str = "session",
        topk: int = 0,
        use_tfidf: bool = False,
        freeze: bool = False,
        gpu: int = -1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        
        self.num_labels = meta_data.get("num_labels")
        self.feature_type = feature_type
        self.hidden_size = hidden_size
        self.use_tfidf = use_tfidf
        
        # Handle kernel_sizes - can be string or list
        if isinstance(kernel_sizes, str):
            kernel_sizes = list(map(int, kernel_sizes.split(',')))
        self.kernel_sizes = kernel_sizes
        
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Conv2d layers matching original:
        # nn.Conv2d(1, hidden_size, (K, embedding_dim)) for K in kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, hidden_size, (K, embedding_dim)) for K in kernel_sizes
        ])
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Prediction layer
        self.prediction_layer = nn.Linear(
            self.hidden_size * len(kernel_sizes),
            self.num_labels
        )
        
        self.init_weights()
    
    def forward(self, x, labels=None):
        # Embedding: (batch, seq) -> (batch, seq, emb_dim)
        x = self.embedding(x)
        
        # Add channel dimension for Conv2d: (batch, 1, seq, emb_dim)
        x = x.unsqueeze(1)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Conv2d output: (batch, hidden_size, seq-K+1, 1)
            c = F.relu(conv(x))
            # Squeeze last dim: (batch, hidden_size, seq-K+1)
            c = c.squeeze(3)
            # Max pool over sequence: (batch, hidden_size)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)
        
        # Concatenate: (batch, hidden_size * num_kernels)
        x = torch.cat(conv_outputs, dim=1)
        
        logits = self.prediction_layer(x)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        
        return logits


def get_model(model_name: str, meta_data: Dict[str, Any], **kwargs):
    """
    Factory function to get model by name.
    """
    models = {
        'transformer': Transformer,
        'lstm': LSTM,
        'cnn': CNN,
    }
    
    model_class = models.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return model_class(meta_data=meta_data, **kwargs)
