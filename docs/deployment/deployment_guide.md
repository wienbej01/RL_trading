
# Multi-Ticker RL Trading System Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Multi-Ticker RL Trading System in production environments. It covers system requirements, installation procedures, configuration, and operational best practices.

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 8 cores (Intel i7 or AMD equivalent)
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 8 GB VRAM (for training)
- **Storage**: 500 GB SSD
- **Network**: Stable internet connection for data feeds

#### Recommended Requirements
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 64+ GB
- **GPU**: NVIDIA RTX 3090/4090 or A100 with 24+ GB VRAM
- **Storage**: 1 TB NVMe SSD
- **Network**: High-speed, low-latency connection

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04 LTS or later (recommended)
- **Alternative**: CentOS 8+ or RHEL 8+
- **Development**: macOS 10.15+ or Windows 10+ (with WSL2)

#### Python Environment
- **Python**: 3.8 or higher
- **Virtual Environment**: venv or conda
- **Package Manager**: pip or conda

#### Dependencies
- **PyTorch**: 1.10+ with CUDA support
- **Stable Baselines3**: 1.6+
- **Pandas**: 1.3+
- **NumPy**: 1.20+
- **Scikit-learn**: 1.0+
- **Optuna**: 2.10+
- **Matplotlib/Seaborn**: For visualization
- **TensorBoard**: For monitoring

#### Data Sources
- **Market Data**: Polygon.io or Databento API access
- **Alternative**: Local data files in supported formats

## Installation

### Step 1: System Preparation

#### Update System Packages
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt upgrade -y

# For CentOS/RHEL
sudo yum update -y
```

#### Install System Dependencies
```bash
# For Ubuntu/Debian
sudo apt install -y python3-pip python3-venv python3-dev build-essential cmake

# For CentOS/RHEL
sudo yum install -y python3-pip python3-virtualenv python3-devel gcc gcc-c++ cmake
```

#### Install NVIDIA Drivers (if using GPU)
```bash
# Add NVIDIA package repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install drivers
sudo apt install -y nvidia-driver-470 nvidia-cuda-toolkit

# Reboot system
sudo reboot
```

### Step 2: Environment Setup

#### Create Virtual Environment
```bash
# Navigate to project directory
cd /path/to/rl-intraday

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install project dependencies
pip install -r requirements.txt
```

### Step 3: Configuration

#### Environment Variables
```bash
# Create environment file
cp .env.example .env

# Edit environment variables
nano .env
```

Example `.env` file:
```bash
# API Keys
POLYGON_API_KEY=your_polygon_api_key_here

# Data Paths
RL_DATA_ROOT=/path/to/data
RL_CACHE_DIR=/path/to/cache
RL_POLYGON_DIR=/path/to/polygon/data

# System Settings
PYTHONPATH=/path/to/rl-intraday:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0
```

#### Configuration File
```bash
# Copy example configuration
cp configs/settings.yaml.example configs/settings.yaml

# Edit configuration
nano configs/settings.yaml
```

### Step 4: Data Preparation

#### Create Data Directory Structure
```bash
# Create data directories
mkdir -p data/{raw,processed,cache}
mkdir -p data/polygon/{historical,realtime}
mkdir -p data/databento/{historical,realtime}
mkdir -p data/models
mkdir -p data/results
mkdir -p data/logs
```

#### Download Historical Data
```bash
# Run data download script
python scripts/download_data.py --start-date 2020-01-01 --end-date 2023-12-31

# Or use data loader directly
python -c "
from src.data.multiticker_data_loader import MultiTickerDataLoader
from src.utils.config_loader import load_config

config = load_config()
loader = MultiTickerDataLoader(config['data'])
data = loader.load_data()
print(f'Data loaded: {data.shape}')
"
```

### Step 5: System Testing

#### Run Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
```

#### Run Integration Test
```bash
# Run basic integration test
python scripts/test_integration.py

# Run full system test
python scripts/test_system.py --config configs/settings.yaml
```

## Deployment Modes

### Development Deployment

#### Local Development Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Set development configuration
export RL_ENV=development

# Run development server
python scripts/dev_server.py
```

#### Development Configuration
```yaml
# configs/development.yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  data_source: "polygon"
  
training:
  total_timesteps: 10000
  learning_rate: 0.0003
  batch_size: 32
  
logging:
  level: "DEBUG"
  tensorboard: true
```

### Staging Deployment

#### Staging Server Setup
```bash
# Create staging user and directory
sudo useradd -m -s /bin/bash staging
sudo su - staging

# Clone repository
git clone https://github.com/your-org/rl-intraday.git