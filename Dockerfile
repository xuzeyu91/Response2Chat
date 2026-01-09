# 使用官方 Python 3.11 slim 镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装依赖
# 先复制 requirements.txt 以利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py .

# 暴露端口
EXPOSE 8000

# 设置默认环境变量（可在运行时覆盖）
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEFAULT_TIMEOUT=300

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["python", "main.py"]
