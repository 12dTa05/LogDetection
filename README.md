# Log Anomaly Detection System

Hệ thống phát hiện bất thường dữ liệu log sử dụng Deep Learning (Transformer, LSTM, CNN).

## Cấu trúc dự án

```
LogAnomalyDetection/
├── communication/           # FastAPI Server
│   └── server.py           # API endpoints: /process_line, /blocks, /metrics
├── data/                   # Dữ liệu log thô
│   └── HDFS/              # Dataset HDFS với anomaly_label.csv
├── data_processed/         # Dữ liệu đã xử lý
│   └── HDFS/              # session_data.pkl, structured logs
├── detection/              # Model AI
│   ├── model.py           # Transformer, LSTM, CNN models
│   ├── models/            # Saved model files (.pt)
│   └── preprocess_data.py # CLI preprocessing
├── parsers/                # Drain log parser
│   ├── drain.py           # drain3 library wrapper
│   ├── main.py            # CLI: python main.py log_file
│   └── utils.py           # Utility functions
├── train/                  # Training scripts
│   ├── cnn.py             # CNN training
│   ├── lstm.py            # LSTM training
│   └── train_transformer.py # Transformer training
├── client/                 # .NET WinForms Client
│   ├── Program.cs         # MainForm, LogUploaderForm, ServerMonitorForm
│   └── LogAnomalyClient.csproj
├── main.py                 # Entry point với subcommands
└── requirements.txt        # Python dependencies
```

## Cài đặt

```bash
# Python dependencies
pip install -r requirements.txt

# .NET Client (requires .NET 8.0)
cd client
dotnet build
```

## Sử dụng

### 1. Parse log thô
```bash
python parsers/main.py data/HDFS/HDFS_2k.log
# hoặc
python main.py parse data/HDFS/HDFS_2k.log
```

### 2. Tiền xử lý dữ liệu
```bash
python detection/preprocess_data.py
# hoặc
python main.py preprocess
```

### 3. Huấn luyện model

**Transformer:**
```bash
python train/train_transformer.py --model transformer --epochs 100
```

**CNN:**
```bash
python train/cnn.py --epoches 100 --kernel_sizes 2,3,4 --hidden_size 100
```

**LSTM:**
```bash
python train/lstm.py --epoches 100 --hidden_size 100 --num_directions 2
```

### 4. Chạy Server
```bash
python communication/server.py
# Server chạy tại http://localhost:8000
```

### 5. Chạy .NET Client
```bash
cd client
dotnet run
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/process_line` | POST | Xử lý một log line |
| `/blocks` | GET | Lấy danh sách blocks |
| `/metrics` | GET | Lấy accuracy metrics |
| `/predict` | POST | Predict anomaly |
| `/reset` | POST | Reset sessions |

## .NET Client Features

- **MainForm**: Chọn model (Transformer/CNN/LSTM), mở Log Uploader và Server Monitor
- **Log Uploader**: Upload file log, xử lý realtime, hiển thị progress
- **Server Monitor**: Hiển thị blocks, metrics (Accuracy, Precision, Recall, F1)

## Models

| Model | Parameters | Description |
|-------|------------|-------------|
| Transformer | ~26K | Self-attention với positional encoding |
| LSTM | ~95K | Bidirectional LSTM với optional attention |
| CNN | ~15K | Conv2d với kernel sizes [2,3,4] |

## License

MIT License
