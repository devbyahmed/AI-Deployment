<<<<<<< HEAD
# AI-Deployment
=======
# Smart Event Crowd Intelligence System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA DeepStream](https://img.shields.io/badge/NVIDIA-DeepStream-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/deepstream-sdk)

A comprehensive, enterprise-grade real-time crowd analysis platform that processes live video feeds to provide instant insights about crowd density, movement patterns, and safety metrics for event organizers.

## 🎯 Overview

The Smart Event Crowd Intelligence System leverages advanced computer vision, AI/ML models, and real-time streaming capabilities to deliver:

- **Real-time crowd monitoring** with sub-500ms latency
- **Advanced AI analysis** using custom CNN, LSTM, and autoencoder models
- **10 distinct behavior classifications** including panic detection and anomaly identification
- **Multi-stream processing** supporting 4-8 concurrent video feeds per GPU
- **Comprehensive alerting system** with escalation and rate limiting
- **Professional dashboard** with real-time visualization and analytics

## 🏗️ Architecture

```
smart-crowd-intelligence/
├── deepstream_core/     # Video processing pipeline (NVIDIA DeepStream)
├── ai_models/          # Custom AI/ML models (PyTorch)
├── backend/           # FastAPI server with WebSocket support
├── frontend/          # Real-time dashboard interface
├── config/           # Configuration management
├── docker/           # Containerization and deployment
├── scripts/          # Setup and utility scripts
└── docs/             # Documentation
```

### Technology Stack

- **Video Processing**: NVIDIA DeepStream SDK with GStreamer
- **AI/ML**: PyTorch with custom neural networks (CNN, LSTM, Autoencoders)
- **Backend**: FastAPI with WebSocket support, asyncio
- **Database**: PostgreSQL with Redis caching
- **Frontend**: Modern HTML5/CSS3/JavaScript with Chart.js
- **Deployment**: Docker with GPU support
- **Monitoring**: Prometheus, Grafana, ELK Stack

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 20GB+ available disk space
- **Software**: Docker, Docker Compose, NVIDIA Docker runtime

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/your-username/smart-crowd-intelligence.git
cd smart-crowd-intelligence

# Run automated setup
./scripts/setup.sh
```

### Development Setup

```bash
# Setup for development environment
./scripts/setup.sh --dev

# Or manual setup
docker-compose -f docker/docker-compose.dev.yml up -d
```

### Production Setup

```bash
# Setup for production
./scripts/setup.sh

# Or manual setup
docker-compose -f docker/docker-compose.yml up -d
```

## 📁 File Structure and Explanations

### DeepStream Core Pipeline (`deepstream_core/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `pipeline_config.py` | Multi-stream pipeline management | DeepStream pipeline setup, GStreamer integration, multi-source handling |
| `crowd_detector.py` | Primary crowd detection | YOLOv5 integration, HOG fallback, real-time object detection |
| `movement_tracker.py` | Object tracking and movement analysis | Hungarian algorithm, velocity calculation, optical flow |
| `density_calculator.py` | Crowd density estimation | Multiple algorithms (grid, kernel, Voronoi), hotspot detection |
| `alert_system.py` | Comprehensive alerting | Escalation rules, rate limiting, multi-channel notifications |

### AI Models (`ai_models/`)

| File | Purpose | Architecture | Features |
|------|---------|-------------|----------|
| `crowd_detection_model.py` | Crowd counting and detection | Custom CNN with spatial attention | Multi-task learning, real-time inference |
| `density_estimation_model.py` | Density heatmap generation | CSRNet implementation | Multi-scale features, perspective correction |
| `behavior_analysis_model.py` | Behavior classification | LSTM + CNN + Autoencoder ensemble | 10 behavior types, anomaly detection |

### Backend API (`backend/`)

| File | Purpose | Key Components |
|------|---------|----------------|
| `main.py` | FastAPI application | REST endpoints, WebSocket handling, middleware |
| `websocket_manager.py` | Real-time communication | Connection management, broadcasting, subscriptions |
| `database.py` | PostgreSQL integration | Connection pooling, async operations, migrations |
| `redis_cache.py` | Caching and sessions | Performance optimization, real-time data storage |
| `data_processor.py` | Data transformation | Analytics aggregation, time-series processing |

### Frontend Dashboard (`frontend/`)

| File | Purpose | Features |
|------|---------|----------|
| `index.html` | Main dashboard interface | Real-time metrics, video streams, alert management |
| `styles.css` | Responsive styling | Modern UI/UX, mobile-friendly design |
| `dashboard.js` | Client-side logic | WebSocket integration, chart updates, interactions |
| `alerts.js` | Alert management | Real-time notifications, filtering, acknowledgments |

### Configuration (`config/`)

| File | Purpose | Contains |
|------|---------|----------|
| `config.yaml` | Main configuration | All system settings, thresholds, feature flags |
| `deepstream_app_config.txt` | DeepStream settings | Pipeline configuration, model paths |
| `tracker_config.txt` | Tracking parameters | Hungarian algorithm settings, tracking thresholds |

### Docker Deployment (`docker/`)

| File | Purpose | Features |
|------|---------|----------|
| `Dockerfile` | Multi-stage container build | Production and development targets, GPU support |
| `docker-compose.yml` | Production deployment | Full stack with monitoring, load balancing |
| `docker-compose.dev.yml` | Development environment | Hot reloading, debugging tools, admin interfaces |
| `nginx.conf` | Reverse proxy configuration | Load balancing, SSL termination, security headers |

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/crowd_intelligence
REDIS_URL=redis://:password@redis:6379/0

# AI Models
MODELS_PATH=/app/models
GPU_ID=0

# Alerts
ENABLE_ALERTS=true
ALERT_WEBHOOK_URL=https://your-webhook.com/alerts

# Security
SECRET_KEY=your-secret-key
API_KEY=your-api-key
```

### System Configuration

Edit `config/config.yaml` for detailed settings:

- **AI Model thresholds**
- **Alert escalation rules**
- **Stream processing parameters**
- **Performance tuning**
- **Feature flags**

## 🎛️ Deployment Options

### Development Environment

Includes debugging tools and hot reloading:

```bash
docker-compose -f docker/docker-compose.dev.yml up -d
```

**Services:**
- API: http://localhost:8001
- Database Admin: http://localhost:5050
- Redis Admin: http://localhost:8082
- Jupyter: http://localhost:8888

### Production Environment

Full production stack with monitoring:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

**Services:**
- Dashboard: http://localhost
- API: http://localhost:8000
- Monitoring: http://localhost:9090
- Grafana: http://localhost:3000
- Kibana: http://localhost:5601

### Kubernetes Deployment

For enterprise scaling (configuration files in `k8s/` directory):

```bash
kubectl apply -f k8s/
```

## 🧪 Testing

### Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v --cov=.

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/integration/ -v

# Test WebSocket connections
pytest tests/websocket/ -v

# Test AI model inference
pytest tests/models/ -v
```

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/api_load_test.py --host=http://localhost:8000
```

## 📊 Performance Metrics

### Estimated Performance

- **Processing Latency**: <500ms end-to-end
- **Concurrent Streams**: 4-8 streams per RTX 3080
- **Model Inference**: 30-60 FPS depending on resolution
- **WebSocket Connections**: 100+ concurrent clients
- **API Throughput**: 1000+ requests/second

### Optimization Tips

1. **GPU Memory**: Use batch processing for multiple streams
2. **CPU Usage**: Enable multi-threading for data processing
3. **Network**: Use RTSP over TCP for stable connections
4. **Storage**: Use SSD for model loading and data processing

## 🛡️ Security Considerations

### Production Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Enable SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up proper authentication
- [ ] Enable API rate limiting
- [ ] Configure security headers
- [ ] Set up backup procedures
- [ ] Enable audit logging

### Network Security

```bash
# Example firewall rules (UFW)
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 8000  # API (if directly exposed)
```

## 🔍 Monitoring and Logging

### Application Metrics

- **Performance**: Request latency, throughput, error rates
- **AI Models**: Inference time, accuracy metrics, model drift
- **System**: CPU, memory, GPU utilization
- **Business**: Crowd counts, alert frequencies, stream uptime

### Log Aggregation

Logs are collected in ELK Stack:
- **Application logs**: `/app/logs/`
- **DeepStream logs**: GStreamer debug output
- **System logs**: Docker container logs
- **Access logs**: NGINX request logs

### Alerting Rules

Configure Prometheus alerts for:
- High API latency (>1s)
- GPU memory usage (>90%)
- Failed model inference
- Database connection issues
- High error rates (>5%)

## 🤝 API Reference

### REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/streams` | GET/POST | Manage video streams |
| `/analytics/{stream_id}` | GET | Get stream analytics |
| `/alerts` | GET | Retrieve active alerts |
| `/analyze/image` | POST | Analyze single image |

### WebSocket Events

| Event | Purpose | Data Format |
|-------|---------|-------------|
| `stream_update` | Real-time stream data | `{stream_id, crowd_data, timestamp}` |
| `alert` | New alert notification | `{alert_id, level, message, timestamp}` |
| `system_status` | System status update | `{active_streams, alerts, performance}` |

### Example Usage

```python
import asyncio
import websockets
import json

async def connect_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Subscribe to stream updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "stream_id": "camera_001"
        }))
        
        # Listen for updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

asyncio.run(connect_websocket())
```

## 🎯 Use Cases

### Event Management
- **Music Festivals**: Monitor crowd density at stages
- **Sports Events**: Track crowd flow in stadiums
- **Conferences**: Manage attendee distribution

### Public Safety
- **Emergency Response**: Detect panic situations
- **Crowd Control**: Prevent dangerous congestion
- **Security Monitoring**: Identify anomalous behavior

### Business Intelligence
- **Retail Analytics**: Track customer flow patterns
- **Queue Management**: Optimize waiting times
- **Space Utilization**: Analyze area usage efficiency

## 🛠️ Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Docker if needed
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Database Connection Failed**
```bash
# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**High Memory Usage**
```bash
# Monitor resource usage
docker stats

# Adjust batch sizes in config.yaml
# Reduce concurrent streams
# Enable model caching
```

### Performance Tuning

**GPU Optimization**
- Use TensorRT for model acceleration
- Enable CUDA memory pooling
- Optimize batch sizes for your GPU

**Network Optimization**
- Use RTSP over TCP for stability
- Configure proper buffer sizes
- Enable video compression for remote streams

## 📚 Documentation

### Additional Resources

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [DeepStream Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/) - NVIDIA DeepStream documentation
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework

### Training Custom Models

See `docs/model_training.md` for detailed instructions on:
- Preparing training data
- Model architecture customization
- Transfer learning approaches
- Performance optimization

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Set up development environment:
   ```bash
   ./scripts/setup.sh --dev
   ```
4. Make your changes
5. Run tests and ensure code quality
6. Submit a pull request

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use ES6+, consistent formatting
- **Documentation**: Update README and docstrings
- **Testing**: Maintain test coverage >80%

### Commit Message Format

```
type(scope): description

Examples:
feat(api): add crowd behavior prediction endpoint
fix(deepstream): resolve memory leak in tracker
docs(readme): update deployment instructions
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** for DeepStream SDK and CUDA support
- **PyTorch** community for deep learning frameworks
- **FastAPI** for excellent web framework
- **OpenCV** for computer vision utilities
- **Contributors** who helped improve this project

## 📞 Support

### Commercial Support

For enterprise deployment, custom model training, and professional support:
- Email: support@crowd-intelligence.com
- Website: https://crowd-intelligence.com
- Documentation: https://docs.crowd-intelligence.com

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Technical questions and ideas
- **Wiki**: Community-contributed documentation
- **Discord**: Real-time community chat

---

**Built with ❤️ for safer, smarter events**

*Smart Event Crowd Intelligence System - Protecting people through intelligent monitoring*
>>>>>>> smart-crowd-intelligence-system
