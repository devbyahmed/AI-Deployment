# Smart Event Crowd Intelligence System - Deployment Guide

## 🚀 Quick Start

### One-Command Setup

```bash
git clone https://github.com/your-username/smart-crowd-intelligence.git
cd smart-crowd-intelligence
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## 📋 Prerequisites

### System Requirements

| Component | Requirement | Recommended |
|-----------|-------------|-------------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| CPU | 4+ cores | 8+ cores |
| RAM | 8GB+ | 16GB+ |
| GPU | NVIDIA RTX 3060+ | NVIDIA RTX 4080+ |
| Storage | 20GB+ | 50GB+ SSD |
| Network | 1Gbps | 10Gbps |

### Software Dependencies

#### Core Requirements
- **Docker** 20.10+
- **Docker Compose** 2.0+
- **NVIDIA Docker Runtime** (for GPU support)
- **Git**

#### Optional (for native development)
- **Python** 3.10+
- **NVIDIA DeepStream SDK** 7.0+
- **CUDA** 11.8+
- **FFmpeg** 4.4+

## 🔧 Installation Steps

### 1. Install Docker

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker-compose --version
```

### 2. Install NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 3. Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/your-username/smart-crowd-intelligence.git
cd smart-crowd-intelligence

# Make setup script executable
chmod +x scripts/setup.sh

# Run automated setup
./scripts/setup.sh

# Or for development environment
./scripts/setup.sh --dev
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up -d

# Check service status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

### Development Deployment

```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Access development tools
# - API: http://localhost:8001
# - Jupyter: http://localhost:8888 (token: development_token)
# - Database Admin: http://localhost:5050
# - Redis Admin: http://localhost:8082
```

### Service URLs (Production)

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost | Main interface |
| API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| Kibana | http://localhost:5601 | Log analysis |

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit with your values
nano .env
```

**Critical Settings:**
```bash
# Database (change passwords!)
DB_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# Security
SECRET_KEY=your_secret_key_here
API_KEY=your_api_key_here

# GPU Configuration
GPU_ID=0
CUDA_VISIBLE_DEVICES=0

# Alerts
ALERT_WEBHOOK_URL=https://your-webhook.com/alerts
```

### System Configuration

Edit `config/config.yaml`:

```yaml
# AI Model thresholds
ai_models:
  crowd_detection:
    confidence_threshold: 0.5
  behavior_analysis:
    confidence_threshold: 0.7

# Alert thresholds
alerts:
  thresholds:
    crowd_count:
      medium: 50
      high: 100
      critical: 200
    density:
      medium: 15
      high: 25
      critical: 35
```

## 🧪 Testing

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check system status
curl http://localhost:8000/status

# Test WebSocket connection
wscat -c ws://localhost:8000/ws
```

### Load Testing

```bash
# Install testing tools
pip install locust httpx pytest

# Run API load tests
locust -f tests/load/api_load_test.py --host=http://localhost:8000

# Run unit tests
pytest tests/ -v --cov=.
```

### Video Stream Testing

```bash
# Test with sample video
curl -X POST http://localhost:8000/streams \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "test_camera_001",
    "uri": "file:///app/sample_videos/crowd_sample.mp4",
    "location": "Test Location"
  }'

# Test with RTSP stream
curl -X POST http://localhost:8000/streams \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "rtsp_camera_001", 
    "uri": "rtsp://username:password@camera-ip:554/stream",
    "location": "Main Entrance"
  }'
```

## 🔒 Security Setup

### Production Security Checklist

- [ ] **Change default passwords** in `.env`
- [ ] **Enable SSL/TLS** certificates
- [ ] **Configure firewall** rules
- [ ] **Set up authentication** (API keys, JWT)
- [ ] **Enable rate limiting**
- [ ] **Configure security headers**
- [ ] **Set up backup procedures**
- [ ] **Enable audit logging**

### SSL Certificate Setup

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/key.pem \
  -out docker/ssl/cert.pem

# For production, use Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

### Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS

# Enable firewall
sudo ufw enable
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# docker/monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sci-api'
    static_configs:
      - targets: ['sci_api:9090']
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
```

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin / admin_password_2024
3. Import dashboards from `docker/monitoring/grafana/dashboards/`

### Alert Rules

Configure Prometheus alerts:

```yaml
# Alerting rules
groups:
  - name: sci_alerts
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
```

## 🔄 Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec sci_postgres pg_dump -U sci_user crowd_intelligence > backup_$DATE.sql

# Restore from backup
docker exec -i sci_postgres psql -U sci_user crowd_intelligence < backup_$DATE.sql
```

### Configuration Backup

```bash
# Backup configurations
tar -czf sci_config_backup_$(date +%Y%m%d).tar.gz \
  .env config/ docker/
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Restart Docker if needed
sudo systemctl restart docker
```

#### Database Connection Issues

```bash
# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres

# Wait for database to initialize
sleep 30
```

#### High Memory Usage

```bash
# Check container resource usage
docker stats

# Adjust memory limits in docker-compose.yml
services:
  sci_api:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### Stream Connection Failures

```bash
# Test stream connectivity
ffprobe rtsp://camera-ip:554/stream

# Check DeepStream logs
docker-compose logs sci_api | grep deepstream

# Verify network connectivity
ping camera-ip
telnet camera-ip 554
```

### Performance Optimization

#### GPU Optimization

```yaml
# docker-compose.yml
services:
  sci_api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Database Tuning

```bash
# PostgreSQL optimization
# Add to docker-compose.yml
environment:
  - POSTGRES_SHARED_PRELOAD_LIBRARIES=pg_stat_statements
  - POSTGRES_MAX_CONNECTIONS=200
  - POSTGRES_SHARED_BUFFERS=256MB
```

#### Network Optimization

```bash
# Increase network buffers
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  sci_api:
    deploy:
      replicas: 3
  
  nginx:
    # Configure load balancing
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=smart-crowd-intelligence

# Scale replicas
kubectl scale deployment sci-api --replicas=5
```

## 📝 Maintenance

### Regular Tasks

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Docker images
docker-compose pull
docker-compose up -d

# Clean up old data
docker system prune -a

# Rotate logs
logrotate /etc/logrotate.d/sci
```

### Health Monitoring

```bash
#!/bin/bash
# health_check.sh
curl -f http://localhost:8000/health || \
  (echo "API health check failed" | mail -s "SCI Alert" admin@company.com)
```

## 🆘 Support

### Getting Help

1. **Documentation**: Read the full README.md
2. **GitHub Issues**: Report bugs and request features
3. **Discord**: Join the community chat
4. **Email**: support@crowd-intelligence.com

### Log Collection

```bash
# Collect logs for support
./scripts/collect_logs.sh

# Creates: sci_logs_$(date).tar.gz
# Send to support team
```

### Performance Monitoring

```bash
# Monitor system resources
htop
iotop
nvidia-smi -l 1

# Monitor container resources
docker stats --no-stream
```

---

## 📞 Emergency Procedures

### System Recovery

```bash
# Complete system restart
docker-compose down
docker system prune -a
./scripts/setup.sh --no-deps --no-build
docker-compose up -d
```

### Data Recovery

```bash
# Restore from backup
docker-compose down
# Restore database backup
# Restore configuration files
docker-compose up -d
```

**🔴 Critical Alert Contact: +1-XXX-XXX-XXXX**

---

*For the latest deployment information, visit: https://docs.crowd-intelligence.com*