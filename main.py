"""
Commands:
    parse       Parse log files
    preprocess  Preprocess data for training
    train       Train models
    server      Start API server
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog='')
    
    subparsers = parser.add_subparsers(dest='command')
    parse_parser = subparsers.add_parser('parse')
    parse_parser.add_argument('log_file', type=str)
    parse_parser.add_argument('--log_format', type=str, default='<Content>')
    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('--train_anomaly_ratio', type=float, default=None)
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--model', type=str, default='cnn', choices=['transformer', 'lstm', 'cnn'])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=1024)
    train_parser.add_argument('--learning_rate', type=float, default=0.01)
    train_parser.add_argument('--gpu', type=int, default=-1)
    server_parser = subparsers.add_parser('server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0')
    server_parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'parse':
        from parsers.drain import LogParser
        
        output_dir = os.path.join(project_root, 'data_processed', 'HDFS')
        os.makedirs(output_dir, exist_ok=True)
        
        log_name = os.path.splitext(os.path.basename(args.log_file))[0]
        output_file = os.path.join(output_dir, f'{log_name}_structured.csv')
        
        parser_obj = LogParser(log_format=args.log_format)
        parser_obj.parse_log_file(args.log_file, output_file)
        
    elif args.command == 'preprocess':
        from detection.preprocess_data import preprocess_hdfs
        preprocess_hdfs(train_anomaly_ratio=args.train_anomaly_ratio)
        
    elif args.command == 'train':
        if args.model == 'cnn':
            os.system(f'python train/cnn.py --epoches {args.epochs} '
                     f'--batch_size {args.batch_size} --learning_rate {args.learning_rate} '
                     f'--gpu {args.gpu}')
        elif args.model == 'lstm':
            os.system(f'python train/lstm.py --epoches {args.epochs} '
                     f'--batch_size {args.batch_size} --learning_rate {args.learning_rate} '
                     f'--gpu {args.gpu}')
        elif args.model == 'transformer':
            os.system(f'python train/train_transformer.py --model transformer '
                     f'--epochs {args.epochs} --batch_size {args.batch_size} '
                     f'--lr {args.learning_rate} --gpu {args.gpu}')
            
    elif args.command == 'server':
        print(f"Starting server on {args.host}:{args.port}")
        os.system(f'uvicorn communication.server:app --host {args.host} --port {args.port}')

if __name__ == '__main__':
    main()
