import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

# Add project root to path BEFORE importing detection
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from detection.model import Transformer, LSTM, CNN, get_model

DEFAULT_CONFIG = {
    "model_name": "",
    "use_attention": False,
    "hidden_size": 128,          # Match reference (was 100)
    "num_layers": 2,
    "num_directions": 2,
    "embedding_dim": 32,         # Match reference (was 16)
    "dataset": "HDFS",
    "data_dir": "../data_processed/HDFS/hdfs_1.0_tar",
    "window_size": 30,
    "stride": 1,
    "feature_type": "sequentials",
    "label_type": "anomaly",
    "use_tfidf": False,
    "max_token_len": 50,
    "min_token_count": 1,
    "epoches": 100,
    "batch_size": 1024,
    "learning_rate": 0.01,
    "topk": 10,
    "patience": 3,
    "random_seed": 42,
    "gpu": -1                    # Match reference (was 0)
}


def load_data(data_path: str):
    print(f"\nLoading data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X_train = torch.LongTensor(data['X_train'])
    X_test = torch.LongTensor(data['X_test'])
    
    if 'y_train_anomaly' in data:
        y_train = torch.LongTensor(data['y_train_anomaly'])
        y_test = torch.LongTensor(data['y_test_anomaly'])
    else:
        y_train = torch.LongTensor(data['y_train'])
        y_test = torch.LongTensor(data['y_test'])
    
    vocab_size = data.get('vocab_size', len(data.get('log2id_train', {})))
    window_size = data.get('window_size', 30)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Window size: {window_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    meta_data = {
        'vocab_size': vocab_size,
        'num_labels': 2,  # binary: normal/anomaly
        'window_size': window_size,
    }
    
    return X_train, y_train, X_test, y_test, meta_data


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size: int = 1024):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_transformer(meta_data: Dict, model_save_path: str, params: Dict, train_loader: DataLoader, test_loader: DataLoader):
    """
    model = Transformer(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
    best_result, history = model.fit(...)
    """
    model = Transformer(
        meta_data=meta_data,
        model_save_path=model_save_path,
        embedding_dim=params.get('embedding_dim', 32),
        hidden_size=params.get('hidden_size', 128),
        num_layers=params.get('num_layers', 2),
        nhead=params.get('nhead', 4),
        dropout=params.get('dropout', 0.1),
        feature_type=params.get('feature_type', 'sequentials'),
        label_type=params.get('label_type', 'anomaly'),
        use_tfidf=params.get('use_tfidf', False),
        topk=params.get('topk', 10),
        gpu=params.get('gpu', 0),
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_result, history = model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=params["epoches"],
        learning_rate=params["learning_rate"],
        patience=params.get('patience', 3),
    )
    
    return model, best_result, history

def train_lstm(meta_data: Dict, model_save_path: str, params: Dict,
               train_loader: DataLoader, test_loader: DataLoader):
    """
    model = LSTM(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
    best_result, history = model.fit(...)
    """
    model = LSTM(
        meta_data=meta_data,
        model_save_path=model_save_path,
        hidden_size=params.get('hidden_size', 100),
        num_directions=params.get('num_directions', 2),
        num_layers=params.get('num_layers', 1),
        window_size=meta_data.get('window_size'),
        use_attention=params.get('use_attention', False),
        embedding_dim=params.get('embedding_dim', 16),
        feature_type=params.get('feature_type', 'sequentials'),
        label_type=params.get('label_type', 'next_log'),
        eval_type=params.get('eval_type', 'session'),
        topk=params.get('topk', 5),
        use_tfidf=params.get('use_tfidf', False),
        gpu=params.get('gpu', -1),
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_result, history = model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=params["epoches"],
        learning_rate=params["learning_rate"],
        patience=params.get('patience', 3),
    )
    
    return model, best_result, history

def train_cnn(meta_data: Dict, model_save_path: str, params: Dict,
              train_loader: DataLoader, test_loader: DataLoader):
    """
    model = CNN(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
    best_result, history = model.fit(...)
    """
    model = CNN(
        meta_data=meta_data,
        model_save_path=model_save_path,
        kernel_sizes=params.get('kernel_sizes', [2, 3, 4]),
        hidden_size=params.get('hidden_size', 100),
        embedding_dim=params.get('embedding_dim', 16),
        feature_type=params.get('feature_type', 'sequentials'),
        label_type=params.get('label_type', 'anomaly'),
        eval_type=params.get('eval_type', 'session'),
        topk=params.get('topk', 0),
        use_tfidf=params.get('use_tfidf', False),
        gpu=params.get('gpu', -1),
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_result, history = model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=params["epoches"],
        learning_rate=params["learning_rate"],
        patience=params.get('patience', 3),
    )
    
    return model, best_result, history

def main():
    parser = argparse.ArgumentParser(description='Train deep learning models for log anomaly detection')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'lstm', 'cnn'])
    parser.add_argument('--data', type=str, default='data_processed/session_data.pkl')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_directions', type=int, default=2)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--kernel_sizes', type=str, default='2,3,4')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=-1,)
    parser.add_argument('--config', type=str, default=None,)
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            params = json.load(f)
    else:
        params = DEFAULT_CONFIG.copy()
    
    params['model_type'] = args.model
    params['epoches'] = args.epochs
    params['batch_size'] = args.batch_size
    params['learning_rate'] = args.lr
    params['embedding_dim'] = args.embedding_dim
    params['hidden_size'] = args.hidden_size
    params['num_layers'] = args.num_layers
    params['num_directions'] = args.num_directions
    params['use_attention'] = args.use_attention
    params['kernel_sizes'] = list(map(int, args.kernel_sizes.split(',')))
    params['patience'] = args.patience
    params['gpu'] = args.gpu

    print(f"Training {args.model.upper()} Model")

    X_train, y_train, X_test, y_test, meta_data = load_data(args.data)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=params['batch_size'])

    model_dir = os.path.join(project_root, 'detection', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f'{args.model}_model.pt')

    if args.model == 'transformer':
        model, best_result, history = train_transformer(meta_data, model_save_path, params, train_loader, test_loader)
    elif args.model == 'lstm':
        model, best_result, history = train_lstm(meta_data, model_save_path, params, train_loader, test_loader)
    elif args.model == 'cnn':
        model, best_result, history = train_cnn(meta_data, model_save_path, params, train_loader, test_loader)

    model.load_model(model_save_path)
    acc, prec, rec, f1 = model.evaluate(test_loader)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return best_result

if __name__ == '__main__':
    main()
