import argparse
import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path BEFORE importing detection
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from detection.model import CNN

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for log anomaly detection')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--kernel_sizes', type=str, default='2,3,4')
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='../data_processed/hdfs')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--feature_type', type=str, default='sequentials', choices=['sequentials', 'semantics'])
    parser.add_argument('--label_type', type=str, default='anomaly')
    parser.add_argument('--use_tfidf', action='store_true')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--max_token_len', type=int, default=50)
    parser.add_argument('--min_token_count', type=int, default=1)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    
    data_path = os.path.join(project_root, 'data_processed', 'session_data.pkl')
    print(f"\nLoading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = torch.LongTensor(data['X_train'])
    X_test = torch.LongTensor(data['X_test'])
    y_train = torch.LongTensor(data['y_train_anomaly'])
    y_test = torch.LongTensor(data['y_test_anomaly'])
    
    vocab_size = data.get('vocab_size', 30)
    window_size = data.get('window_size', 30)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Window size: {window_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    meta_data = {
        'vocab_size': vocab_size,
        'num_labels': 2,
        'window_size': window_size,
    }

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_dir = os.path.join(project_root, 'detection', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'cnn_model.pt')

    kernel_sizes = list(map(int, args.kernel_sizes.split(',')))

    model = CNN(
        meta_data=meta_data,
        model_save_path=model_save_path,
        kernel_sizes=kernel_sizes,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        feature_type=args.feature_type,
        label_type=args.label_type,
        use_tfidf=args.use_tfidf,
        gpu=args.gpu,
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_result, history = model.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epoches,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    model.load_model(model_save_path)
    acc, prec, rec, f1 = model.evaluate(test_loader)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")


if __name__ == '__main__':
    main()
