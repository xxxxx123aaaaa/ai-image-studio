#!/bin/bash
# AI Image Studio — 生产启动脚本
# 用法：bash start.sh

set -e
cd "$(dirname "$0")"

# 安装依赖（首次运行）
pip install -r requirements.txt -q

# 创建必要目录
mkdir -p outputs gallery

# 用 gunicorn 启动（生产级）
# 如果没有 gunicorn，先 pip install gunicorn
if command -v gunicorn &>/dev/null; then
    exec gunicorn app:app \
        --bind 0.0.0.0:7860 \
        --workers 2 \
        --threads 4 \
        --timeout 300 \
        --keep-alive 5 \
        --access-logfile - \
        --error-logfile -
else
    # fallback：Flask 开发服务器
    python app.py
fi
