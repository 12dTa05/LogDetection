"""
Endpoints:
- POST /process_line: Process a single log line
- POST /predict: Predict anomaly for a single log line  
- POST /predict_batch: Predict anomalies for multiple log lines
- GET /health: Health check
- GET /stats: Get prediction statistics
- GET /blocks: Get all block data
- GET /metrics: Get accuracy metrics
"""

import os
import sys
import re
import pickle
import logging
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from detection.model import get_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogLine(BaseModel):
    line: str
    client_id: Optional[int] = 0

class LogRequest(BaseModel):
    log_line: str
    block_id: Optional[str] = None


class BatchLogRequest(BaseModel):
    log_lines: List[str]


class ProcessResult(BaseModel):
    result: str
    timestamp: str = None
    message: str = None


class PredictionResponse(BaseModel):
    status: str  # Normal or Anomaly
    confidence: float
    block_id: Optional[str] = None
    event_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    anomaly_count: int


class StatsResponse(BaseModel):
    total_predictions: int
    total_anomalies: int
    sessions_tracked: int
    uptime_seconds: float


class BlockInfo(BaseModel):
    block_id: str
    log_count: int
    last_log: str
    status: str
    client_ids: List[int]


class BlocksResponse(BaseModel):
    blocks: List[BlockInfo]
    total_log_count: int
    total_block_count: int
    last_updated: str


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    tn: int
    fn: int
    last_updated: str

class TemplateMiner:
    def __init__(self):
        self.templates = {}
        self.cluster_count = 0
    
    def add_log_message(self, log_message: str) -> Dict[str, Any]:
        # Replace numbers with <NUM>
        template = re.sub(r'\b\d+\b', '<NUM>', log_message)
        # Replace IPs with <IP>
        template = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '<IP>', template)
        # Replace block IDs with <*>
        template = re.sub(r'blk_-?\d+', 'blk_<*>', template)
        
        if template not in self.templates:
            self.cluster_count += 1
            self.templates[template] = self.cluster_count
        
        cluster_id = self.templates[template]
        
        return {
            "cluster_id": cluster_id,
            "template_mined": template
        }

app = FastAPI(title="Log Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session dict: block_id -> {templates, event_ids, label, client_ids, last_log}
session_dict: Dict[str, Dict] = defaultdict(lambda: {
    'templates': [],
    'event_ids': [],
    'label': 0,
    'client_ids': set(),
    'last_log': '',
    'predictions': []
})

log_count = 0
recent_logs = []
template_miner = TemplateMiner()
model = None
log2id = {}
id2log = {}
vocab_size = 0
window_size = 30
ground_truth_labels = {}
start_time = datetime.now()


def preprocess_log(log_line: str, session_dict: Dict, template_miner: TemplateMiner) -> tuple:
    """
    Matches original: blk_id, session_dict, template = preprocess_log(...)
    """
    # Extract block ID
    block_match = re.search(r'blk_-?\d+', log_line)
    blk_id = block_match.group() if block_match else None
    
    # Parse template
    result = template_miner.add_log_message(log_line)
    template = result["template_mined"]
    
    return blk_id, session_dict, template


def update_block_data(blk_id: str, log_line: str, current_time: datetime):
    global session_dict
    
    if blk_id: # Get template
        result = template_miner.add_log_message(log_line)
        template = result["template_mined"]
        event_id = log2id.get(template, log2id.get('<UNK>', 1))
        
        session_dict[blk_id]['templates'].append(template)
        session_dict[blk_id]['event_ids'].append(event_id)
        session_dict[blk_id]['last_log'] = template
        session_dict[blk_id]['last_updated'] = current_time


def predict_block_session(blk_id: str) -> str:
    """
    Returns classification result.
    """
    global model
    
    if model is None or blk_id not in session_dict:
        return "normal"
    
    event_ids = session_dict[blk_id]['event_ids']
    
    if len(event_ids) == 0:
        return "normal"

    if len(event_ids) < window_size:
        padded = event_ids + [0] * (window_size - len(event_ids))
    else:
        padded = event_ids[-window_size:]

    x = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    
    result = "anomaly" if pred == 1 else "normal"
    session_dict[blk_id]['predictions'].append(pred)
    
    return result


def load_ground_truth_labels(label_path: str):
    """Ground truth labels for evaluation."""
    global ground_truth_labels
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or 'BlockId' in line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    block_id = parts[0].strip()
                    label = 1 if parts[1].strip().lower() == 'anomaly' else 0
                    ground_truth_labels[block_id] = label
        logger.info(f"Loaded {len(ground_truth_labels)} ground truth labels")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, log2id, id2log, vocab_size, window_size
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(project_root, 'detection', 'models', 'transformer_model.pt')
    data_path = os.path.join(project_root, 'data_processed', 'HDFS', 'session_data.pkl')
    label_path = os.path.join(project_root, 'data', 'HDFS', 'anomaly_label.csv')

    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        log2id = data.get('log2id_train', {})
        id2log = data.get('id2log_train', {})
        vocab_size = data.get('vocab_size', 30)
        window_size = data.get('window_size', 30)
        logger.info(f"Loaded vocabulary: {vocab_size} entries")

    if os.path.exists(model_path):
        meta_data = {'vocab_size': vocab_size, 'num_labels': 2, 'window_size': window_size}
        model = get_model('transformer', meta_data, embedding_dim=32, hidden_size=128, num_layers=2)
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model not found: {model_path}")

    load_ground_truth_labels(label_path)

@app.get("/")
async def root():
    return {
        "name": "Log Anomaly Detection API",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/process_line")
async def process_line(log_line: LogLine):
    global session_dict, log_count
    
    try:
        log_count += 1
        current_time = datetime.now()
        
        blk_id, session_dict, template = preprocess_log(
            log_line=log_line.line,
            session_dict=session_dict,
            template_miner=template_miner
        )
        
        recent_logs.append({
            "timestamp": current_time.isoformat(),
            "message": log_line.line,
            "client_id": log_line.client_id
        })
        if len(recent_logs) > 100:
            recent_logs.pop(0)
        
        update_block_data(blk_id, log_line.line, current_time)

        if blk_id and log_line.client_id is not None:
            session_dict[blk_id]['client_ids'].add(log_line.client_id)

        processed_result = f"Processed: ({log_line.line[:50]}...) with template ({template}) from block ({blk_id}) (Client ID: {log_line.client_id})"
        
        log_classification = predict_block_session(blk_id)
        processed_result += f"\nPrediction for block ({blk_id}): ({log_classification})"
        
        logger.info(processed_result)
        
        return {"result": processed_result}
        
    except Exception as e:
        logger.error(f"Error processing log line: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse) # prediction stat
async def stats():
    global log_count, session_dict, start_time
    
    total_anomalies = sum(
        1 for s in session_dict.values() 
        if s['predictions'] and s['predictions'][-1] == 1
    )
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return StatsResponse(
        total_predictions=log_count,
        total_anomalies=total_anomalies,
        sessions_tracked=len(session_dict),
        uptime_seconds=round(uptime, 2)
    )


@app.get("/blocks")
async def get_blocks():
    global session_dict, log_count
    
    blocks = []
    for blk_id, data in session_dict.items():
        status = "anomaly" if data['predictions'] and data['predictions'][-1] == 1 else "normal"
        blocks.append(BlockInfo(
            block_id=blk_id,
            log_count=len(data['templates']),
            last_log=data['last_log'][:100] if data['last_log'] else "",
            status=status,
            client_ids=list(data['client_ids'])
        ))
    
    return BlocksResponse(
        blocks=blocks,
        total_log_count=log_count,
        total_block_count=len(session_dict),
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.get("/metrics")
async def get_metrics():
    global session_dict, ground_truth_labels
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for blk_id, data in session_dict.items():
        if blk_id not in ground_truth_labels:
            continue
        
        true_label = ground_truth_labels[blk_id]
        pred_label = 1 if data['predictions'] and data['predictions'][-1] == 1 else 0
        
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return MetricsResponse(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: LogRequest):
    global model, log_count
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    log_count += 1
    
    blk_id, _, template = preprocess_log(request.log_line, session_dict, template_miner)
    blk_id = request.block_id or blk_id

    if blk_id:
        update_block_data(blk_id, request.log_line, datetime.now())
        classification = predict_block_session(blk_id)
        status = "Anomaly" if classification == "anomaly" else "Normal"
    else:
        status = "Normal"
    
    result = template_miner.add_log_message(request.log_line)
    
    return PredictionResponse(
        status=status,
        confidence=0.95,
        block_id=blk_id,
        event_id=f"E{result['cluster_id']}"
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchLogRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for log_line in request.log_lines:
        req = LogRequest(log_line=log_line)
        pred = await predict(req)
        predictions.append(pred)
    
    anomaly_count = sum(1 for p in predictions if p.status == "Anomaly")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        anomaly_count=anomaly_count
    )


@app.post("/reset")
async def reset_sessions():
    global session_dict, log_count, recent_logs
    
    session_dict.clear()
    log_count = 0
    recent_logs.clear()
    
    return {"status": "reset", "message": "All sessions cleared"}


@app.post("/load_labels")
async def load_labels_endpoint():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_path = os.path.join(project_root, 'data', 'HDFS', 'anomaly_label.csv')
    
    load_ground_truth_labels(label_path)
    
    return {"status": "loaded", "count": len(ground_truth_labels)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
