# Production Deployment Guide

## ⚠️ Known Issues and Solutions

### 1. Template Mismatch Issue (CRITICAL - FIXED)

**Problem**: The original server used simple regex-based template parsing while training used the sophisticated Drain3 algorithm. This caused template inconsistency and reduced model accuracy in production.

**Solution**: ✅ **FIXED** - Server now uses drain3 with the same configuration as training:
- Load drain3 state from `parsers/drain3_state.bin`
- Configuration: `depth=4`, `st=0.5` (matches training)
- Templates are now consistent between training and inference

**How to use**:
1. After parsing logs with `parsers/drain.py`, the drain3 state is automatically saved to `parsers/drain3_state.bin`
2. Server automatically loads this state on startup
3. If state file doesn't exist, server creates a fresh instance (warning logged)

---

### 2. Global State Issue with Multi-Workers

**Problem**: Current implementation uses global variables (`session_dict`, `template_miner`, etc.) which don't work correctly with multiple uvicorn workers.

**Current Code** (communication/server.py):
```python
# WARNING: For production with multiple workers, use Redis or Database
session_dict: Dict[str, Dict] = defaultdict(lambda: {...})
```

**Solution for Production**: Use Redis or Database for shared state.

#### Option A: Redis Implementation (Recommended)

1. **Install Redis dependencies**:
```bash
pip install redis aioredis
```

2. **Update server.py imports**:
```python
import redis
import json
from typing import Dict, Any

# Initialize Redis client
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)
```

3. **Replace global session_dict with Redis**:
```python
def get_session_data(blk_id: str) -> Dict:
    """Get session data from Redis."""
    data = redis_client.get(f"session:{blk_id}")
    if data:
        return json.loads(data)
    return {
        'templates': [],
        'event_ids': [],
        'label': 0,
        'client_ids': [],
        'last_log': '',
        'predictions': []
    }

def set_session_data(blk_id: str, data: Dict):
    """Save session data to Redis."""
    # Convert set to list for JSON serialization
    if 'client_ids' in data and isinstance(data['client_ids'], set):
        data['client_ids'] = list(data['client_ids'])
    redis_client.set(f"session:{blk_id}", json.dumps(data))
    redis_client.expire(f"session:{blk_id}", 86400)  # 24 hour TTL
```

4. **Update functions to use Redis**:
```python
def update_block_data(blk_id: str, log_line: str, current_time: datetime):
    if blk_id:
        result = template_miner.add_log_message(log_line)
        template = result["template_mined"]
        event_id = log2id.get(template, log2id.get('<UNK>', 1))
        
        # Get from Redis
        session_data = get_session_data(blk_id)
        session_data['templates'].append(template)
        session_data['event_ids'].append(event_id)
        session_data['last_log'] = template
        session_data['last_updated'] = current_time.isoformat()
        
        # Save to Redis
        set_session_data(blk_id, session_data)
```

5. **Run with multiple workers**:
```bash
uvicorn communication.server:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Option B: PostgreSQL/MySQL Implementation

1. **Create database schema**:
```sql
CREATE TABLE sessions (
    block_id VARCHAR(255) PRIMARY KEY,
    templates TEXT,  -- JSON array
    event_ids TEXT,  -- JSON array
    label INTEGER DEFAULT 0,
    client_ids TEXT,  -- JSON array
    last_log TEXT,
    predictions TEXT,  -- JSON array
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_last_updated ON sessions(last_updated);
```

2. **Use SQLAlchemy ORM**:
```python
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

Base = declarative_base()

class Session(Base):
    __tablename__ = 'sessions'
    
    block_id = Column(String(255), primary_key=True)
    templates = Column(Text)  # JSON
    event_ids = Column(Text)  # JSON
    label = Column(Integer, default=0)
    client_ids = Column(Text)  # JSON
    last_log = Column(Text)
    predictions = Column(Text)  # JSON
    last_updated = Column(DateTime)

# Initialize
engine = create_engine('postgresql://user:pass@localhost/logdetection')
SessionLocal = sessionmaker(bind=engine)
```

---

### 3. URL Hardcoding in Client

**Problem**: Server URL hardcoded in C# client code.

**Solution**: ✅ **FIXED** - Configuration file created at `client/appsettings.json`:
```json
{
  "ServerUrl": "http://localhost:8000",
  "DefaultModel": "Transformer",
  "DefaultClientId": 1,
  "RequestTimeout": 30,
  "LogFilePath": "",
  "RefreshInterval": 2000
}
```

**How to use in C#**:
```csharp
using System.Text.Json;
using System.IO;

public class AppConfig
{
    public string ServerUrl { get; set; }
    public string DefaultModel { get; set; }
    public int DefaultClientId { get; set; }
    public int RequestTimeout { get; set; }
    public string LogFilePath { get; set; }
    public int RefreshInterval { get; set; }
}

// Load config
var configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "appsettings.json");
var json = File.ReadAllText(configPath);
var config = JsonSerializer.Deserialize<AppConfig>(json);

// Use in HttpClient
var httpClient = new HttpClient();
httpClient.BaseAddress = new Uri(config.ServerUrl);
httpClient.Timeout = TimeSpan.FromSeconds(config.RequestTimeout);
```

---

## 🚀 Production Deployment Checklist

### Before Deployment
- [ ] Run log parsing with `python parsers/drain.py` to generate drain3 state
- [ ] Verify `parsers/drain3_state.bin` exists
- [ ] Train models and save to `detection/models/`
- [ ] Verify `data_processed/HDFS/session_data.pkl` exists
- [ ] Set up Redis or Database for session storage (if using multiple workers)
- [ ] Update `appsettings.json` with production server URL
- [ ] Add environment variables for sensitive configs

### Deployment
```bash
# Single worker (development)
uvicorn communication.server:app --host 0.0.0.0 --port 8000

# Multiple workers (production with Redis)
uvicorn communication.server:app --host 0.0.0.0 --port 8000 --workers 4

# With SSL
uvicorn communication.server:app --host 0.0.0.0 --port 443 --ssl-keyfile=/path/to/key.pem --ssl-certfile=/path/to/cert.pem
```

### Monitoring
- [ ] Set up logging to file or external service
- [ ] Monitor Redis/Database connections
- [ ] Track API response times
- [ ] Monitor model prediction latency
- [ ] Set up alerts for high error rates

---

## 📊 Performance Optimization

### 1. Template Miner Caching
The drain3 template miner is now loaded once on startup and reused, which is much more efficient than creating new instances per request.

### 2. Model Inference Batching
For high-throughput scenarios, consider batching predictions:
```python
@app.post("/predict_batch_optimized")
async def predict_batch_optimized(requests: List[LogRequest]):
    # Batch process multiple requests
    sequences = []
    block_ids = []
    
    for req in requests:
        blk_id = extract_block_id(req.log_line)
        session = get_session_data(blk_id)
        sequences.append(prepare_sequence(session))
        block_ids.append(blk_id)
    
    # Single batch inference
    with torch.no_grad():
        X = torch.tensor(sequences)
        logits = model(X)
        predictions = torch.argmax(logits, dim=1)
    
    return results
```

### 3. Database Connection Pooling
If using database, use connection pooling:
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/logdetection',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

---

## 🔒 Security Considerations

1. **API Authentication**: Add API key authentication
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Input Validation**: Validate all log inputs
4. **HTTPS**: Always use HTTPS in production
5. **CORS**: Restrict CORS to specific origins in production

---

## 📈 Scaling Strategy

### Small Scale (< 1000 req/s)
- Single server with Redis
- 4-8 uvicorn workers

### Medium Scale (1000-10000 req/s)
- Load balancer (nginx/HAProxy)
- Multiple server instances
- Redis cluster
- Separate model serving (TorchServe)

### Large Scale (> 10000 req/s)
- Kubernetes deployment
- Horizontal pod autoscaling
- Distributed tracing (Jaeger)
- Message queue (Kafka/RabbitMQ)
- Model serving infrastructure (NVIDIA Triton)

---

## 🛠️ Troubleshooting

### Issue: "drain3_state.bin not found"
**Solution**: Run the parsing script first:
```bash
python parsers/drain.py
```

### Issue: Template mismatch errors
**Solution**: Ensure drain3 state was created with the same config (depth=4, st=0.5)

### Issue: Redis connection errors with multiple workers
**Solution**: Check Redis is running and accessible:
```bash
redis-cli ping
```

### Issue: Model not loading
**Solution**: Verify model file exists and matches architecture:
```bash
ls -lh detection/models/transformer_model.pt
```
