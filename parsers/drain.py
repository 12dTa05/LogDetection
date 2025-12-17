import os
import sys
import time
import re
import pandas as pd

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.drain3_config import Drain3Config


class LogParser:
    def __init__(self, log_format: str = None, depth: int = None, st: float = None, persistence_path: str = None):
        """
        Args:
            log_format: Regex format for parsing log lines
            depth: Drain tree depth (default from config)
            st: Similarity threshold (default from config)
            persistence_path: Path to save/load drain3 state
        """
        self.depth = depth if depth is not None else Drain3Config.DEPTH
        self.st = st if st is not None else Drain3Config.SIMILARITY_THRESHOLD
        self.persistence_path = persistence_path
        
        self.log_format = log_format or Drain3Config.LOG_FORMAT

        self.config = TemplateMinerConfig()
        self.config.drain_depth = self.depth
        self.config.drain_sim_th = self.st
        self.config.profiling_enabled = Drain3Config.PROFILING_ENABLED
        
        # Set up persistence if path provided
        if persistence_path:
            self.config.persistence_handler = FilePersistence(persistence_path)
        
        self.template_miner = TemplateMiner(config=self.config)
        
        self.log_df = None
    
    def generate_logformat_regex(self, log_format: str):
        """
        Converts format '<Date> <Time> <Content>' to regex with named groups.
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)
        
        regex = ''
        for k in range(len(splitters)):
            if splitters[k].startswith('<'):
                header = splitters[k].strip('<>')
                headers.append(header)
                regex += f'(?P<{header}>.*?)'
            else:
                regex += re.escape(splitters[k])
        
        regex = f'^{regex}$'
        return regex, headers
    
    def load_log_data(self, log_file: str, regex: str, headers: list):
        log_messages = []
        line_count = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                match = re.match(regex, line)
                
                if match:
                    message = {header: match.group(header) for header in headers}
                    message['LineId'] = line_count
                    log_messages.append(message)
                else:
                    log_messages.append({
                        'LineId': line_count,
                        'Content': line
                    })
        
        return pd.DataFrame(log_messages)
    
    def initialize_template_miner(self):
        self.template_miner = TemplateMiner(config=self.config)
    
    def log_batch_progress(self, idx: int, batch_size: int, batch_start_time: float):
        elapsed = time.time() - batch_start_time
        print(f"Processed {idx} logs in {elapsed:.2f}s")
    
    def parse(self, log_file: str):
        """
        Args: log_file: Path to log file
        Returns: DataFrame with LineId, Content, EventId, EventTemplate
        """
        regex, headers = self.generate_logformat_regex(self.log_format)
        
        # Load log data
        self.log_df = self.load_log_data(log_file, regex, headers)
        self.log_df['EventId'] = pd.Series(dtype=int)
        self.log_df['EventTemplate'] = pd.Series(dtype=object)
        self.log_df['BlockId'] = pd.Series(dtype=object)

        self.initialize_template_miner()
        
        start_time = time.time()
        batch_start_time = start_time
        batch_size = 100000
        
        for idx, row in self.log_df.iterrows():
            content = str(row.get('Content', ''))
            result = self.template_miner.add_log_message(content)
            
            if idx > 0 and idx % batch_size == 0:
                self.log_batch_progress(idx, batch_size, batch_start_time)
                batch_start_time = time.time()

            if result["change_type"] != "none":
                self.log_change(idx, content, result)

            self.log_df.loc[idx, 'EventId'] = result["cluster_id"]
            
            # Extract BlockId from content
            block_match = re.search(r'blk_-?\d+', content)
            if block_match:
                self.log_df.loc[idx, 'BlockId'] = block_match.group()

        for cluster in self.template_miner.drain.clusters:
            mask = self.log_df['EventId'] == cluster.cluster_id
            self.log_df.loc[mask, 'EventTemplate'] = cluster.get_template()
        
        self.finalize_parsing(start_time)
        
        return self.log_df
    
    def log_change(self, idx: int, content: str, result: dict):
        change_type = result["change_type"]
        cluster_id = result["cluster_id"]
        if change_type == "cluster_created":
            template = result.get('template_mined', '')[:80]
            print(f"New cluster E{cluster_id}: {template}...")
    
    def finalize_parsing(self, start_time: float):
        total_time = time.time() - start_time
        num_clusters = len(self.template_miner.drain.clusters)
        print(f"\nParsing completed in {total_time:.2f}s")
        print(f"Total log lines: {len(self.log_df)}")
        print(f"Total clusters (event types): {num_clusters}")
    
    def save_structured_logs(self, output_file: str):
        if self.log_df is not None:
            self.log_df.to_csv(output_file, index=False)
            print(f"Saved structured logs to {output_file}")
    
    def get_event_templates(self):
        templates = {}
        for cluster in self.template_miner.drain.clusters:
            templates[cluster.cluster_id] = cluster.get_template()
        return templates


def parse_hdfs_logs(input_file: str, output_file: str) -> pd.DataFrame:
    # Set up persistence path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_file = os.path.join(project_root, 'parsers', Drain3Config.STATE_FILE)
    
    # Use config defaults (no need to specify depth and st)
    parser = LogParser(persistence_path=state_file)
    df = parser.parse(input_file)
    parser.save_structured_logs(output_file)
    
    print(f"Drain3 state automatically saved to {state_file}")
    
    return df

if __name__ == '__main__':
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'data', 'HDFS.log')  # Updated to use full dataset
    output_file = os.path.join(project_root, 'data_processed', 'HDFS_structured.csv')
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    df = parse_hdfs_logs(input_file, output_file)
    
    print("\nSample output:")
    print(df[['LineId', 'EventId', 'EventTemplate']].head(10))
