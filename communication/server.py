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
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# Add project root to path BEFORE importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model import get_model
from config.drain3_config import Drain3Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogLine(BaseModel):
    line: str
    client_id: Optional[int] = 0
    model_type: Optional[str] = "transformer"  # Support model selection

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

# TemplateMiner is now imported from drain3
# See startup_event() for initialization with saved state

app = FastAPI(title="Log Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state - WARNING: For production with multiple workers, use Redis or Database
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
template_miner = None  # Will be initialized in startup_event with drain3
model = None
current_model_type = "transformer"  # Track current model type
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
    # Extract block ID - support both HDFS blocks (blk_*) and Spark RDD blocks (rdd_*)
    block_match = re.search(r'(blk_-?\d+|rdd_\d+_\d+)', log_line)
    blk_id = block_match.group() if block_match else None
    
    # If no block found, try to extract task/stage ID as session identifier
    if not blk_id:
        # Try task pattern: "task 31.0 in stage 29.0"
        task_match = re.search(r'task (\d+\.\d+) in stage (\d+\.\d+)', log_line)
        if task_match:
            blk_id = f"task_{task_match.group(1)}_stage_{task_match.group(2)}"
    
    # Parse template using drain3
    result = template_miner.add_log_message(log_line)
    template = result["template_mined"]
    
    return blk_id, session_dict, template


def update_block_data(blk_id: str, template: str, current_time: datetime):
    """
    Update block data with template (no longer parse template here to avoid duplication).
    Template should come from preprocess_log().
    """
    global session_dict
    
    if blk_id:
        # Initialize block_id entry if it doesn't exist
        if blk_id not in session_dict:
            session_dict[blk_id] = {
                'templates': [],
                'event_ids': [],
                'predictions': [],
                'client_ids': set(),
                'last_log': '',
                'last_updated': current_time
            }
        
        event_id = log2id.get(template, log2id.get('<UNK>', 1))
        
        session_dict[blk_id]['templates'].append(template)
        session_dict[blk_id]['event_ids'].append(event_id)
        session_dict[blk_id]['last_log'] = template
        session_dict[blk_id]['last_updated'] = current_time


def predict_block_session(blk_id: str) -> str:
    """
    Returns classification result using sliding windows (matching training logic).
    """
    global model
    
    if model is None or blk_id not in session_dict:
        return "normal"
    
    event_ids = session_dict[blk_id]['event_ids']
    
    if len(event_ids) == 0:
        return "normal"

    # Create windows matching training logic
    windows = []
    if len(event_ids) <= window_size:
        # Pad with zeros for short sequences
        padded = event_ids + [0] * (window_size - len(event_ids))
        windows.append(padded[:window_size])
    else:
        # Sliding window for long sequences (matching preprocessing)
        stride = 1  # Same as training
        for i in range(0, len(event_ids) - window_size + 1, stride):
            window = event_ids[i:i + window_size]
            windows.append(window)
    
    # Predict on all windows
    predictions = []
    with torch.no_grad():
        for window in windows:
            x = torch.tensor([window], dtype=torch.long)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)
    
    # Majority voting OR any anomaly detection
    # Using "any anomaly" - if any window is anomaly, session is anomaly
    anomaly_count = sum(predictions)
    result = "anomaly" if anomaly_count > 0 else "normal"
    
    session_dict[blk_id]['predictions'].append(1 if result == "anomaly" else 0)
    
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
    """Load model and initialize drain3 template miner on startup."""
    global model, log2id, id2log, vocab_size, window_size, template_miner, current_model_type
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(project_root, 'detection', 'models', f'{current_model_type}_model.pt')
    data_path = os.path.join(project_root, 'data_processed', 'session_data.pkl')
    label_path = os.path.join(project_root, 'data', 'anomaly_label.csv')
    drain_state_path = os.path.join(project_root, 'parsers', Drain3Config.STATE_FILE)

    # Initialize drain3 TemplateMiner using centralized config
    config = TemplateMinerConfig()
    config.drain_depth = Drain3Config.DEPTH
    config.drain_sim_th = Drain3Config.SIMILARITY_THRESHOLD
    config.profiling_enabled = Drain3Config.PROFILING_ENABLED
    
    template_miner = TemplateMiner(config=config)
    
    # Load saved drain state if exists
    if os.path.exists(drain_state_path):
        try:
            with open(drain_state_path, 'rb') as f:
                template_miner.drain = pickle.load(f)
            logger.info(f"Loaded drain3 state from {drain_state_path}")
            logger.info(f"Drain3 initialized with {len(template_miner.drain.clusters)} clusters")
        except Exception as e:
            logger.warning(f"Could not load drain3 state: {e}. Using fresh instance.")
            logger.info(f"Drain3 initialized with 0 clusters (fresh)")
    else:
        logger.warning(f"Drain3 state file not found: {drain_state_path}. Using fresh instance.")
        logger.info(f"Drain3 initialized with 0 clusters (fresh)")
    
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        log2id = data.get('log2id_train', {})
        id2log = data.get('id2log_train', {})
        vocab_size = data.get('vocab_size', 30)
        window_size = data.get('window_size', 30)
        logger.info(f"Loaded vocabulary: {vocab_size} entries")

    # Load model with comprehensive error handling
    if os.path.exists(model_path):
        try:
            meta_data = {'vocab_size': vocab_size, 'num_labels': 2, 'window_size': window_size}
            
            # Load appropriate model based on current_model_type
            if current_model_type == "transformer":
                model = get_model('transformer', meta_data, embedding_dim=32, hidden_size=128, num_layers=2)
            elif current_model_type == "lstm":
                model = get_model('lstm', meta_data, hidden_size=100, num_directions=2, num_layers=1)
            elif current_model_type == "cnn":
                model = get_model('cnn', meta_data, kernel_sizes=[2,3,4], hidden_size=100)
            
            state = torch.load(model_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model.eval()
            
            # Validate model with test input
            test_input = torch.randint(0, max(vocab_size, 10), (1, window_size))
            with torch.no_grad():
                _ = model(test_input)
            
            logger.info(f"Loaded {current_model_type} model from {model_path}")
            logger.info("Model validation successful")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Server will run without model (predictions disabled)")
            model = None
    else:
        logger.warning(f"Model not found: {model_path}")
        logger.info("Server will run without model (predictions disabled)")
        model = None

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
        
        update_block_data(blk_id, template, current_time)

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
        update_block_data(blk_id, template, datetime.now())
        classification = predict_block_session(blk_id)
        status = "Anomaly" if classification == "anomaly" else "Normal"
    else:
        status = "Normal"
    
    # Get cluster_id from the already parsed result
    result = template_miner.drain.match(template)
    cluster_id = result.cluster_id if result else 0
    
    return PredictionResponse(
        status=status,
        confidence=0.95,
        block_id=blk_id,
        event_id=f"E{cluster_id}"
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
    label_path = os.path.join(project_root, 'data', 'anomaly_label.csv')
    
    load_ground_truth_labels(label_path)
    
    return {"status": "loaded", "count": len(ground_truth_labels)}


@app.post("/switch_model")
async def switch_model(model_type: str):
    """
    Switch to a different model (transformer, lstm, or cnn).
    Allows dynamic model switching without restarting the server.
    """
    global model, current_model_type
    
    valid_models = ["transformer", "cnn", "lstm"]
    model_type_lower = model_type.lower()
    
    if model_type_lower not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Choose from {valid_models}"
        )
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'detection', 'models', f'{model_type_lower}_model.pt')
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model file not found: {model_path}"
        )
    
    try:
        meta_data = {
            'vocab_size': vocab_size, 
            'num_labels': 2, 
            'window_size': window_size
        }
        
        # Load appropriate model with specific configurations
        if model_type_lower == "transformer":
            new_model = get_model('transformer', meta_data, 
                                embedding_dim=32, hidden_size=128, num_layers=2)
        elif model_type_lower == "lstm":
            new_model = get_model('lstm', meta_data, 
                                hidden_size=100, num_directions=2, num_layers=1)
        elif model_type_lower == "cnn":
            new_model = get_model('cnn', meta_data, 
                                kernel_sizes=[2,3,4], hidden_size=100)
        
        # Load weights
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            new_model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            new_model.load_state_dict(state, strict=False)
        
        new_model.eval()
        
        # Validate model with test input
        test_input = torch.randint(0, max(vocab_size, 10), (1, window_size))
        with torch.no_grad():
            _ = new_model(test_input)
        
        # Switch to new model
        model = new_model
        current_model_type = model_type_lower
        
        logger.info(f"Successfully switched to {model_type_lower} model")
        
        return {
            "status": "success", 
            "model": model_type_lower,
            "message": f"Switched to {model_type_lower} model"
        }
        
    except Exception as e:
        logger.error(f"Failed to switch to {model_type_lower} model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load {model_type_lower} model: {str(e)}"
        )




@app.get("/current_model")
async def get_current_model():
    """Get information about the currently loaded model."""
    return {
        "model_type": current_model_type,
        "model_loaded": model is not None,
        "vocab_size": vocab_size,
        "window_size": window_size
    }


@app.get("/blocks")
async def get_blocks(client_id: Optional[int] = None):
    """
    Get all blocks and their information.
    Optionally filter by client_id.
    """
    blocks_list = []
    total_log_count = 0
    
    for blk_id, data in session_dict.items():
        # Filter by client_id if provided
        if client_id is not None and client_id not in data.get('client_ids', set()):
            continue
        
        log_count = len(data.get('event_ids', []))
        total_log_count += log_count
        
        # Get last log (last template)
        templates = data.get('event_ids', [])
        last_template_id = templates[-1] if templates else 0
        last_log = id2log.get(last_template_id, "")
        
        # Get status from predictions
        predictions = data.get('predictions', [])
        if predictions:
            # If any prediction is anomaly (1), mark as anomaly
            status = "anomaly" if any(predictions) else "normal"
        else:
            status = "unknown"
        
        blocks_list.append({
            "block_id": blk_id,
            "log_count": log_count,
            "last_log": last_log[:100],  # Truncate for display
            "status": status,
            "client_ids": list(data.get('client_ids', set()))
        })
    
    return {
        "blocks": blocks_list,
        "total_log_count": total_log_count,
        "total_block_count": len(blocks_list),
        "last_updated": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """
    Calculate and return evaluation metrics based on predictions vs ground truth.
    """
    if not ground_truth_labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    tp = fp = tn = fn = 0
    
    for blk_id, data in session_dict.items():
        predictions = data.get('predictions', [])
        if not predictions:
            continue
        
        # Use majority voting OR any-anomaly strategy
        predicted = 1 if any(predictions) else 0
        
        # Get ground truth
        actual = ground_truth_labels.get(blk_id, 0)
        
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 0 and actual == 1:
            fn += 1
    
    total = tp + fp + tn + fn
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "last_updated": datetime.now().isoformat()
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

