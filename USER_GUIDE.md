# Hướng Dẫn Sử Dụng Hệ Thống Log Anomaly Detection

**Phiên bản:** 1.0 (Fixed)  
**Ngày cập nhật:** 2025-12-16  
**Trạng thái:** ✅ Đã sửa tất cả lỗi nghiêm trọng

---

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#1-yêu-cầu-hệ-thống)
2. [Cài Đặt](#2-cài-đặt)
3. [Workflow Hoàn Chỉnh](#3-workflow-hoàn-chỉnh)
4. [Sử Dụng Server](#4-sử-dụng-server)
5. [Sử Dụng Client](#5-sử-dụng-client)
6. [API Endpoints](#6-api-endpoints)
7. [Switching Models](#7-switching-models)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Yêu Cầu Hệ Thống

### Python Backend
- **Python**: 3.8 hoặc mới hơn (đã test với 3.13.9)
- **OS**: Windows/Linux/macOS
- **RAM**: Tối thiểu 4GB
- **Storage**: ~2GB cho models và data

### .NET Client
- **.NET SDK**: 8.0 hoặc mới hơn
- **OS**: Windows (WinForms)

---

## 2. Cài Đặt

### Bước 1: Cài Đặt Python Dependencies

```powershell
# Tại thư mục gốc của project
cd p:\Rust\LogDetection

# Cài đặt các packages cần thiết
pip install -r requirements.txt
```

**Kiểm tra cài đặt:**
```powershell
python -c "import torch; import fastapi; import drain3; print('✅ All packages installed')"
```

### Bước 2: Cài Đặt .NET Client (Optional)

```powershell
cd client
dotnet restore
dotnet build
```

---

## 3. Workflow Hoàn Chỉnh

### 🔄 Pipeline Xử Lý Dữ Liệu

```
Raw Logs (HDFS_2k.log)
    ↓
[1. Parse] → HDFS_structured.csv + drain3_state.bin
    ↓
[2. Preprocess] → session_data.pkl
    ↓
[3. Train] → transformer/lstm/cnn_model.pt (ĐÃ CÓ SẴN)
    ↓
[4. Server] → FastAPI Server (localhost:8000)
    ↓
[5. Client] → .NET WinForms / API Calls
```

---

### Bước 1: Parse Raw Logs

**Mục đích:** Chuyển đổi raw logs thành structured format và extract templates

```powershell
# Parse HDFS logs
python parsers\drain.py

# Hoặc sử dụng main.py
python main.py parse data\HDFS\HDFS_2k.log
```

**Output:**
- ✅ `data_processed/HDFS/HDFS_structured.csv` - Structured logs với BlockId, EventTemplate
- ✅ `parsers/drain3_state.bin` - Drain3 clustering state

**Kiểm tra:**
```powershell
# Xem 10 dòng đầu
Get-Content data_processed\HDFS\HDFS_structured.csv -Head 10
```

---

### Bước 2: Preprocess Data

**Mục đích:** Tạo sliding windows và vocabulary cho training/inference

```powershell
python detection\preprocess_data.py

# Hoặc
python main.py preprocess
```

**Output:**
- ✅ `data_processed/HDFS/session_data.pkl` - Training/test data với:
  - X_train, X_test (sliding windows)
  - y_train_anomaly, y_test_anomaly (labels)
  - log2id, id2log (vocabulary)
  - vocab_size, window_size (metadata)

**Kiểm tra:**
```powershell
python -c "import pickle; data=pickle.load(open('data_processed/HDFS/session_data.pkl','rb')); print(f'Vocab: {data[\"vocab_size\"]}, Windows: {len(data[\"X_train\"])}')"
```

---

### Bước 3: Train Models (OPTIONAL - Đã có models)

Hệ thống đã có 3 models đã train sẵn:
- ✅ `detection/models/transformer_model.pt` (755 KB)
- ✅ `detection/models/lstm_model.pt` (384 KB)
- ✅ `detection/models/cnn_model.pt` (66 KB)

**Nếu muốn train lại:**

#### Train Transformer
```powershell
python train\train_transformer.py --model transformer --epochs 100 --batch_size 1024 --lr 0.01
```

#### Train LSTM
```powershell
python train\lstm.py --epoches 100 --batch_size 1024 --learning_rate 0.01 --hidden_size 100 --num_directions 2
```

#### Train CNN
```powershell
python train\cnn.py --epoches 100 --batch_size 1024 --learning_rate 0.01 --hidden_size 100 --kernel_sizes 2,3,4
```

**Hoặc dùng main.py:**
```powershell
python main.py train --model transformer --epochs 100
python main.py train --model lstm --epochs 100
python main.py train --model cnn --epochs 100
```

---

## 4. Sử Dụng Server

### Khởi Động Server

```powershell
# Method 1: Trực tiếp
python communication\server.py

# Method 2: Qua main.py
python main.py server --host 0.0.0.0 --port 8000

# Method 3: Với uvicorn
uvicorn communication.server:app --host 0.0.0.0 --port 8000 --reload
```

**Server sẽ startup và:**
1. ✅ Load Drain3 state từ `parsers/drain3_state.bin`
2. ✅ Load vocabulary từ `data_processed/HDFS/session_data.pkl`
3. ✅ Load Transformer model (default)
4. ✅ Load ground truth labels
5. ✅ Start listening tại `http://localhost:8000`

**Kiểm tra server:**
```powershell
# Trong terminal khác
curl http://localhost:8000/health

# Hoặc trên browser
# Mở: http://localhost:8000/docs (FastAPI Swagger UI)
```

**Expected output:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 5. Sử Dụng Client

### Method 1: .NET WinForms Client

```powershell
cd client
dotnet run
```

**Giao diện Main Form:**
1. **Model Selection:** Chọn Transformer/CNN/LSTM
2. **New Log Uploader Client:** Mở form upload logs
3. **Open Server Monitor:** Mở form monitor metrics

**Log Uploader Form:**
1. Click "Upload Log File"
2. Chọn file log (ví dụ: `data/HDFS/HDFS_2k.log`)
3. Logs sẽ được gửi lên server từng dòng
4. Xem progress và results real-time

**Server Monitor Form:**
- Hiển thị blocks được process
- Metrics: Accuracy, Precision, Recall, F1
- Confusion matrix: TP, FP, TN, FN
- Auto-refresh mỗi 5 giây

### Method 2: Python Script

```python
import requests

# Send single log line
response = requests.post(
    "http://localhost:8000/process_line",
    json={
        "line": "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906",
        "client_id": 1,
        "model_type": "transformer"
    }
)
print(response.json())
```

### Method 3: cURL

```powershell
# Process a log line
curl -X POST "http://localhost:8000/process_line" `
  -H "Content-Type: application/json" `
  -d '{\"line\": \"081109 203518 143 INFO dfs.DataNode: Receiving block blk_123\", \"client_id\": 1}'

# Get blocks
curl http://localhost:8000/blocks

# Get metrics
curl http://localhost:8000/metrics
```

---

## 6. API Endpoints

### Core Endpoints

#### GET `/health`
**Mô tả:** Health check server

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET `/current_model`
**Mô tả:** Xem model hiện tại

**Response:**
```json
{
  "model_type": "transformer",
  "model_loaded": true,
  "vocab_size": 30,
  "window_size": 30
}
```

#### POST `/process_line`
**Mô tả:** Xử lý một log line

**Request:**
```json
{
  "line": "081109 203518 143 INFO dfs.DataNode: Receiving block blk_123",
  "client_id": 1,
  "model_type": "transformer"
}
```

**Response:**
```json
{
  "result": "Processed: (081109 203518...) with template (...) from block (blk_123) (Client ID: 1)\nPrediction for block (blk_123): (normal)"
}
```

#### POST `/predict`
**Mô tả:** Predict anomaly cho log line

**Request:**
```json
{
  "log_line": "081109 203518 143 INFO dfs.DataNode: Receiving block blk_123",
  "block_id": null
}
```

**Response:**
```json
{
  "status": "Normal",
  "confidence": 0.95,
  "block_id": "blk_123",
  "event_id": "E5"
}
```

#### GET `/blocks`
**Mô tả:** Lấy danh sách blocks

**Response:**
```json
{
  "blocks": [
    {
      "block_id": "blk_123",
      "log_count": 15,
      "last_log": "Receiving block...",
      "status": "normal",
      "client_ids": [1]
    }
  ],
  "total_log_count": 100,
  "total_block_count": 10,
  "last_updated": "2025-12-16 22:30:00"
}
```

#### GET `/metrics`
**Mô tả:** Lấy accuracy metrics

**Response:**
```json
{
  "accuracy": 0.9545,
  "precision": 0.9200,
  "recall": 0.9100,
  "f1_score": 0.9150,
  "tp": 91,
  "fp": 8,
  "tn": 850,
  "fn": 9,
  "last_updated": "2025-12-16 22:30:00"
}
```

#### POST `/reset`
**Mô tả:** Reset tất cả sessions

**Response:**
```json
{
  "status": "reset",
  "message": "All sessions cleared"
}
```

---

## 7. Switching Models

### ✨ NEW FEATURE: Dynamic Model Switching

Bạn có thể switch giữa 3 models **KHÔNG CẦN restart server**!

### Method 1: API Call

```powershell
# Switch to CNN
curl -X POST "http://localhost:8000/switch_model?model_type=cnn"

# Switch to LSTM
curl -X POST "http://localhost:8000/switch_model?model_type=lstm"

# Switch to Transformer
curl -X POST "http://localhost:8000/switch_model?model_type=transformer"
```

**Response:**
```json
{
  "status": "success",
  "model": "cnn",
  "message": "Switched to cnn model"
}
```

### Method 2: Python

```python
import requests

response = requests.post(
    "http://localhost:8000/switch_model",
    params={"model_type": "lstm"}
)
print(response.json())
```

### Method 3: Client (TODO)

Trong .NET client, bạn có thể thêm button để call API này.

---

## 8. Troubleshooting

### ❌ Error: "Model not found"

**Nguyên nhân:** Model file không tồn tại

**Giải pháp:**
```powershell
# Kiểm tra models
ls detection\models\*.pt

# Nếu thiếu, train lại
python train\train_transformer.py --model transformer --epochs 10
```

### ❌ Error: "KeyError: 'BlockId'"

**Nguyên nhân:** File CSV không có cột BlockId (đã sửa)

**Giải pháp:**
```powershell
# Parse lại logs với version mới
python parsers\drain.py

# Sau đó preprocess lại
python detection\preprocess_data.py
```

### ❌ Error: "Server connection failed"

**Nguyên nhân:** Server chưa chạy hoặc port bị block

**Giải pháp:**
```powershell
# Kiểm tra server đang chạy
netstat -ano | findstr :8000

# Restart server
python communication\server.py
```

### ❌ Error: "Duplicate template parsing"

**Trạng thái:** ✅ ĐÃ SỬA

Lỗi này đã được fix - server không còn parse template 2 lần.

### ❌ Error: "Inconsistent window padding"

**Trạng thái:** ✅ ĐÃ SỬA

Inference giờ đã sử dụng sliding windows giống như training.

---

## 📊 Example Workflow

### Scenario: Upload và phân tích file log

```powershell
# 1. Start server
python communication\server.py

# Trong terminal khác:

# 2. Parse logs (nếu chưa có)
python parsers\drain.py

# 3. Process logs
curl -X POST "http://localhost:8000/process_line" `
  -H "Content-Type: application/json" `
  -d '{\"line\": \"081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906\", \"client_id\": 1}'

# 4. Check metrics
curl http://localhost:8000/metrics

# 5. Switch to CNN model
curl -X POST "http://localhost:8000/switch_model?model_type=cnn"

# 6. Process more logs với CNN
curl -X POST "http://localhost:8000/process_line" `
  -H "Content-Type: application/json" `
  -d '{\"line\": \"081109 203520 148 INFO dfs.DataNode: PacketResponder 1\", \"client_id\": 1}'

# 7. Get blocks
curl http://localhost:8000/blocks
```

---

## 🎯 Best Practices

### 1. Workflow Chuẩn
```
Parse → Preprocess → (Train) → Start Server → Use Client/API
```

### 2. Model Selection
- **Transformer:** Tốt nhất cho accuracy (~26K params)
- **LSTM:** Balance giữa speed và accuracy (~95K params)
- **CNN:** Nhanh nhất, accuracy vẫn tốt (~15K params)

### 3. Performance Tips
- Sử dụng batch processing cho nhiều logs
- Switch model dựa trên workload
- Monitor memory usage với nhiều sessions

### 4. Production Deployment
- Sử dụng `--reload` cho development
- Sử dụng Gunicorn/Uvicorn workers cho production
- Implement Redis cho session storage (nhiều workers)
- Add authentication cho API endpoints

---

## 📝 Changelog

### Version 1.0 (2025-12-16) - Fixed Release

**🔴 CRITICAL FIXES:**
1. ✅ Fixed duplicate template parsing
2. ✅ Fixed missing BlockId column
3. ✅ Fixed model type mismatch
4. ✅ Fixed inconsistent window padding

**🟡 IMPROVEMENTS:**
5. ✅ Centralized Drain3 configuration
6. ✅ Added comprehensive error handling
7. ✅ Added model validation on startup

**✨ NEW FEATURES:**
8. ✅ Dynamic model switching without restart
9. ✅ `/switch_model` endpoint
10. ✅ `/current_model` endpoint
11. ✅ Sliding window predictions

---

## 🆘 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:

1. **Log files:** Server sẽ print errors ra console
2. **FastAPI docs:** http://localhost:8000/docs
3. **Health check:** http://localhost:8000/health
4. **Review:** `LOGIC_ERRORS_REPORT.md` - Danh sách các lỗi đã sửa

---

## 🎓 Tóm Tắt Các Lệnh Quan Trọng

```powershell
# Cài đặt
pip install -r requirements.txt

# Parse logs
python parsers\drain.py

# Preprocess
python detection\preprocess_data.py

# Start server
python communication\server.py

# Test health
curl http://localhost:8000/health

# Switch model
curl -X POST "http://localhost:8000/switch_model?model_type=cnn"

# Process log
curl -X POST "http://localhost:8000/process_line" -H "Content-Type: application/json" -d '{\"line\": \"your log here\"}'

# Get metrics
curl http://localhost:8000/metrics
```

---

**Hệ thống giờ đã sẵn sàng sử dụng! 🚀**
