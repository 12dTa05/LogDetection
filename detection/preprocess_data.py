"""
Detection Preprocessing CLI
Data preprocessing for detection models.

Usage:
    python preprocess_data.py [-h] [--train_anomaly_ratio TRAIN_ANOMALY_RATIO]

Matches original CLI interface.
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_structured_logs(structured_log_path: str) -> pd.DataFrame:
    """Load structured logs from CSV."""
    print(f"Loading structured logs from {structured_log_path}...")
    df = pd.read_csv(structured_log_path)
    print(f"Loaded {len(df)} log entries")
    return df


def load_labels(label_path: str) -> dict:
    """Load anomaly labels."""
    print(f"Loading labels from {label_path}...")
    labels = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    block_id = parts[0].strip()
                    label = parts[1].strip()
                    if block_id != 'BlockId':
                        labels[block_id] = 1 if label.lower() == 'anomaly' else 0
    
    normal_count = sum(1 for v in labels.values() if v == 0)
    anomaly_count = sum(1 for v in labels.values() if v == 1)
    print(f"Loaded {len(labels)} labels (Normal: {normal_count}, Anomaly: {anomaly_count})")
    
    return labels


def build_vocabulary(df: pd.DataFrame) -> tuple:
    """Build log2id and id2log mappings."""
    templates = df['EventTemplate'].unique().tolist()
    
    log2id = {'<PAD>': 0, '<UNK>': 1}
    for i, template in enumerate(templates):
        log2id[template] = i + 2
    
    id2log = {v: k for k, v in log2id.items()}
    
    print(f"Built vocabulary with {len(log2id)} entries")
    return log2id, id2log


def create_session_dict(df: pd.DataFrame, labels: dict, log2id: dict) -> dict:
    """Create session dictionary with templates and labels."""
    session_dict = {}
    
    for block_id, group in df.groupby('BlockId'):
        templates = group['EventTemplate'].tolist()
        event_ids = [log2id.get(t, 1) for t in templates]
        label = labels.get(block_id, 0)
        
        session_dict[block_id] = {
            'templates': templates,
            'event_ids': event_ids,
            'label': label
        }
    
    print(f"Created {len(session_dict)} sessions")
    return session_dict


def create_sliding_windows(session_dict: dict, window_size: int = 30, stride: int = 1) -> dict:
    """Create sliding window features."""
    features_dict = {}
    
    for block_id, session in session_dict.items():
        event_ids = session['event_ids']
        label = session['label']
        
        windows = []
        window_labels = []
        window_anomalies = []
        
        if len(event_ids) <= window_size:
            # Pad with zeros
            window = event_ids + [0] * (window_size - len(event_ids))
            windows.append(window[:window_size])
            window_labels.append(0)
            window_anomalies.append(label)
        else:
            for i in range(0, len(event_ids) - window_size, stride):
                window = event_ids[i:i + window_size]
                next_log = event_ids[i + window_size] if i + window_size < len(event_ids) else 0
                
                windows.append(window)
                window_labels.append(next_log)
                window_anomalies.append(label)
        
        if not windows:
            window = event_ids + [0] * (window_size - len(event_ids))
            windows.append(window[:window_size])
            window_labels.append(0)
            window_anomalies.append(label)
        
        features_dict[block_id] = {
            'features': {
                'sequentials': np.array(windows, dtype=np.int64)
            },
            'window_labels': np.array(window_labels, dtype=np.int64),
            'window_anomalies': np.array(window_anomalies, dtype=np.int64)
        }
    
    total_windows = sum(len(f['window_labels']) for f in features_dict.values())
    print(f"Created {total_windows} windows from {len(features_dict)} sessions")
    
    return features_dict


def train_test_split(features_dict: dict, train_ratio: float = 0.8, 
                     train_anomaly_ratio: float = None) -> tuple:
    """Split data into train and test sets."""
    block_ids = list(features_dict.keys())
    np.random.seed(42)
    np.random.shuffle(block_ids)
    
    split_idx = int(len(block_ids) * train_ratio)
    train_ids = block_ids[:split_idx]
    test_ids = block_ids[split_idx:]
    
    # Collect windows
    def collect_windows(ids):
        X, y_next, y_anomaly = [], [], []
        for bid in ids:
            f = features_dict[bid]
            X.extend(f['features']['sequentials'].tolist())
            y_next.extend(f['window_labels'].tolist())
            y_anomaly.extend(f['window_anomalies'].tolist())
        return np.array(X), np.array(y_next), np.array(y_anomaly)
    
    X_train, y_train_next, y_train_anomaly = collect_windows(train_ids)
    X_test, y_test_next, y_test_anomaly = collect_windows(test_ids)
    
    print(f"\nTrain: {len(X_train)} windows, Test: {len(X_test)} windows")
    print(f"Train anomalies: {y_train_anomaly.sum()}, Test anomalies: {y_test_anomaly.sum()}")
    
    return (X_train, y_train_next, y_train_anomaly, 
            X_test, y_test_next, y_test_anomaly)


def preprocess_hdfs(train_anomaly_ratio: float = None):
    """Main preprocessing function for HDFS dataset."""
    # Paths
    structured_log_path = os.path.join(project_root, 'data_processed', 'HDFS', 'HDFS_structured.csv')
    label_path = os.path.join(project_root, 'data', 'HDFS', 'anomaly_label.csv')
    output_path = os.path.join(project_root, 'data_processed', 'HDFS', 'session_data.pkl')
    
    # Parameters
    window_size = 30
    stride = 1
    
    print("=" * 60)
    print("HDFS Log Preprocessing")
    print("=" * 60)
    
    if train_anomaly_ratio is not None:
        print(f"Train anomaly ratio: {train_anomaly_ratio}")
    
    # Load data
    df = load_structured_logs(structured_log_path)
    labels = load_labels(label_path)
    
    # Build vocabulary
    log2id, id2log = build_vocabulary(df)
    
    # Create sessions
    session_dict = create_session_dict(df, labels, log2id)
    
    # Create features
    features_dict = create_sliding_windows(session_dict, window_size, stride)
    
    # Train/test split
    (X_train, y_train_next, y_train_anomaly,
     X_test, y_test_next, y_test_anomaly) = train_test_split(features_dict)
    
    # Save
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_next,
        'y_test': y_test_next,
        'y_train_anomaly': y_train_anomaly,
        'y_test_anomaly': y_test_anomaly,
        'log2id_train': log2id,
        'id2log_train': id2log,
        'vocab_size': len(log2id),
        'window_size': window_size,
        'stride': stride,
        'session_dict': session_dict,
        'features_dict': features_dict,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved preprocessed data to {output_path}")
    print("=" * 60)
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Preprocessing script')
    
    parser.add_argument('--train_anomaly_ratio', type=float, default=None,
                        help='Train anomaly ratio. Default: None')
    
    args = parser.parse_args()
    
    preprocess_hdfs(train_anomaly_ratio=args.train_anomaly_ratio)


if __name__ == '__main__':
    main()
