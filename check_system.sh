#!/bin/bash
# Script kiểm tra hệ thống LogDetection
# Sử dụng: bash check_system.sh

echo "=================================================="
echo "  LogDetection System Check"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check conda environment
echo "1. Kiểm tra môi trường Conda..."
if conda env list | grep -q "IoT"; then
    echo -e "${GREEN}✓${NC} Môi trường IoT tồn tại"
else
    echo -e "${RED}✗${NC} Môi trường IoT không tồn tại"
    echo "   Tạo môi trường: conda create -n IoT python=3.10"
fi
echo ""

# Check required files
echo "2. Kiểm tra files cần thiết..."

if [ -f "data/HDFS.log" ]; then
    SIZE=$(du -h data/HDFS.log | cut -f1)
    echo -e "${GREEN}✓${NC} data/HDFS.log ($SIZE)"
else
    echo -e "${RED}✗${NC} data/HDFS.log không tồn tại"
fi

if [ -f "data/anomaly_label.csv" ]; then
    SIZE=$(du -h data/anomaly_label.csv | cut -f1)
    echo -e "${GREEN}✓${NC} data/anomaly_label.csv ($SIZE)"
else
    echo -e "${RED}✗${NC} data/anomaly_label.csv không tồn tại"
fi
echo ""

# Check processed data
echo "3. Kiểm tra data đã xử lý..."

if [ -f "data_processed/HDFS_structured.csv" ]; then
    SIZE=$(du -h data_processed/HDFS_structured.csv | cut -f1)
    echo -e "${GREEN}✓${NC} data_processed/HDFS_structured.csv ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} data_processed/HDFS_structured.csv chưa có"
    echo "   Chạy: python parsers/drain.py"
fi

if [ -f "data_processed/session_data.pkl" ]; then
    SIZE=$(du -h data_processed/session_data.pkl | cut -f1)
    echo -e "${GREEN}✓${NC} data_processed/session_data.pkl ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} data_processed/session_data.pkl chưa có"
    echo "   Chạy: python detection/preprocess_data.py"
fi

if [ -f "parsers/drain3_state.bin" ]; then
    SIZE=$(du -h parsers/drain3_state.bin | cut -f1)
    echo -e "${GREEN}✓${NC} parsers/drain3_state.bin ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} parsers/drain3_state.bin chưa có"
    echo "   Sẽ được tạo khi chạy parsers/drain.py"
fi
echo ""

# Check models
echo "4. Kiểm tra models..."

if [ -f "detection/models/transformer_model.pt" ]; then
    SIZE=$(du -h detection/models/transformer_model.pt | cut -f1)
    echo -e "${GREEN}✓${NC} transformer_model.pt ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} transformer_model.pt chưa có"
    echo "   Chạy: python train/train_transformer.py --model transformer --epochs 100"
fi

if [ -f "detection/models/lstm_model.pt" ]; then
    SIZE=$(du -h detection/models/lstm_model.pt | cut -f1)
    echo -e "${GREEN}✓${NC} lstm_model.pt ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} lstm_model.pt chưa có"
    echo "   Chạy: python train/lstm.py --epoches 100"
fi

if [ -f "detection/models/cnn_model.pt" ]; then
    SIZE=$(du -h detection/models/cnn_model.pt | cut -f1)
    echo -e "${GREEN}✓${NC} cnn_model.pt ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} cnn_model.pt chưa có"
    echo "   Chạy: python train/cnn.py --epoches 100"
fi
echo ""

# Check Python packages (if in IoT env)
echo "5. Kiểm tra Python packages..."
if [ "$CONDA_DEFAULT_ENV" = "IoT" ]; then
    echo "   Đang trong môi trường: $CONDA_DEFAULT_ENV"
    
    PACKAGES=("torch" "drain3" "pandas" "fastapi" "uvicorn")
    for pkg in "${PACKAGES[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
            echo -e "   ${GREEN}✓${NC} $pkg ($VERSION)"
        else
            echo -e "   ${RED}✗${NC} $pkg chưa cài"
        fi
    done
else
    echo -e "   ${YELLOW}⚠${NC} Không trong môi trường IoT"
    echo "   Chạy: conda activate IoT"
fi
echo ""

# Check directory structure
echo "6. Kiểm tra cấu trúc thư mục..."
DIRS=("data" "data_processed" "parsers" "detection" "detection/models" "train" "communication" "config")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "   ${GREEN}✓${NC} $dir/"
    else
        echo -e "   ${RED}✗${NC} $dir/ không tồn tại"
    fi
done
echo ""

# Summary
echo "=================================================="
echo "  Tóm Tắt"
echo "=================================================="
echo ""
echo "Workflow chuẩn:"
echo "  1. conda activate IoT"
echo "  2. python parsers/drain.py"
echo "  3. python detection/preprocess_data.py"
echo "  4. python train/train_transformer.py --model transformer --epochs 100"
echo "  5. python communication/server.py"
echo ""
echo "Xem chi tiết: cat PATH_FIXES.md"
echo ""
