import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from parsers.drain import LogParser


def main():
    parser.add_argument('log_file', type=str)
    parser.add_argument('--log_format', type=str, default='<Content>')
    parser.add_argument('--state_file', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)

    print(f"Log file: {args.log_file}")
    print(f"Log format: {args.log_format}")
    print(f"State file: {args.state_file}")
    print(f"Config file: {args.config_file}")

    log_dir = os.path.dirname(args.log_file)
    log_name = os.path.splitext(os.path.basename(args.log_file))[0]
    output_dir = os.path.join(project_root, 'data_processed', 'HDFS')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{log_name}_structured.csv')

    log_parser = LogParser(log_format=args.log_format, depth=4, sim_th=0.5, max_children=100, extra_delimiters=['_'])

    log_parser.parse_log_file(args.log_file, output_file)
    
    print(f"\nOutput: {output_file}")

if __name__ == '__main__':
    main()
