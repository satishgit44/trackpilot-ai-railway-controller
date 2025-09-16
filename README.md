# 🚄 TrackPilot - AI Railway Traffic Controller

[![Build Status](https://github.com/satishgit44/trackpilot-ai-railway-controller/workflows/CI/badge.svg)](https://github.com/satishgit44/trackpilot-ai-railway-controller/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![React 18](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)

**Smart India Hackathon 2025 - Problem Statement ID: 25022**

A hybrid AI system for real-time train scheduling in congested railway sections with safety rule-checker, human oversight, and continuous learning capabilities.

## 🎯 Problem Statement

**Maximizing Section Throughput Using AI-Powered Precise Train Traffic Control**

Traditional railway traffic control systems struggle with:
- Suboptimal scheduling in congested sections
- Lack of real-time adaptation
- Limited human-AI collaboration
- Insufficient learning from operator expertise

## 🚀 Solution Overview

TrackPilot addresses these challenges through:

### 🧠 **Hybrid AI System**
- **Real-time scheduling** using Reinforcement Learning (PPO, A2C, DQN)
- **Imitation learning** from human operator expertise
- **Safety rule-checker** with mandatory constraint validation
- **Explainable AI** with confidence scores and reasoning

### 👥 **Human-AI Collaboration**
- **Controller override** capabilities with learning feedback loop
- **Audit trails** for all decisions and modifications  
- **Performance dashboards** for continuous improvement
- **Behavioral cloning** from override patterns

### 📊 **Real-time Visualization**
- **Space-time graph** visualization using D3.js
- **Live train tracking** and conflict detection
- **Interactive control panel** for operator decision-making
- **Performance metrics** and alert management

### 🏗️ **Production-Ready Architecture**
- **C++ core engine** with LibTorch for high-performance inference
- **Python training pipeline** for model development and retraining
- **React frontend** with real-time updates via WebSocket
- **Containerized deployment** with Docker and CI/CD pipelines

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features) 
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrackPilot System Architecture            │
├─────────────────────┬───────────────────┬────────────────────┤
│   Frontend (React)  │  Core Engine      │  Training Pipeline │
│   ├── D3.js Viz     │  ├── C++/LibTorch │  ├── PyTorch      │
│   ├── Control Panel │  ├── REST API     │  ├── RL/IL        │
│   └── Real-time UI  │  └── Rule Checker │  └── Model Export │
├─────────────────────┼───────────────────┼────────────────────┤
│        Data Layer   │   Infrastructure  │    Monitoring      │
│   ├── PostgreSQL    │  ├── Docker       │  ├── Prometheus   │
│   ├── Redis Cache   │  ├── Nginx        │  ├── Grafana      │
│   └── Override Logs │  └── CI/CD        │  └── Jaeger       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Core Engine** (`/core/`) - C++ with LibTorch
   - High-performance AI model inference
   - Safety rule validation engine
   - REST API for frontend communication
   - Real-time train tracking and scheduling

2. **AI Training** (`/python/`) - Python with PyTorch  
   - Reinforcement Learning algorithms (PPO, A2C, DQN)
   - Imitation Learning with behavioral cloning
   - Model export to TorchScript format
   - Override data processing and retraining

3. **Frontend** (`/web/`) - React with D3.js
   - Interactive space-time graph visualization
   - Real-time train position tracking  
   - Human override interface
   - Performance metrics dashboard

4. **Infrastructure** (`/infra/`) - DevOps and Deployment
   - Multi-stage Docker containers
   - Docker Compose for local development
   - GitHub Actions CI/CD pipelines
   - Production deployment configurations

## ✨ Features

### 🤖 AI-Powered Scheduling
- **Reinforcement Learning** models trained on railway scheduling scenarios
- **Multi-objective optimization** balancing efficiency, safety, and throughput
- **Real-time inference** with sub-second response times
- **Confidence scoring** for decision transparency

### 🛡️ Safety & Compliance
- **Mandatory safety rules** with automatic violation detection  
- **Signal timing constraints** and track capacity validation
- **Emergency override** capabilities for critical situations
- **Audit logging** of all decisions and operator actions

### 🎯 Human-AI Collaboration  
- **Explainable recommendations** with natural language reasoning
- **Override learning system** that improves from human feedback
- **Performance tracking** and continuous model improvement
- **Role-based access control** for different operator levels

### 📈 Real-time Monitoring
- **Live train tracking** with GPS integration capabilities
- **Conflict detection** and resolution suggestions  
- **Performance metrics** (throughput, delays, efficiency)
- **Alert management** with priority-based notifications

### 🔧 Production Features
- **High availability** with load balancing and failover
- **Scalable architecture** supporting multiple railway sections
- **API-first design** for integration with existing systems
- **Comprehensive logging** and monitoring infrastructure

## 🚀 Quick Start

### Prerequisites
- **Docker** and **Docker Compose** 
- **Git** for version control
- **Node.js 18+** (for local frontend development)
- **Python 3.10+** (for AI training)
- **C++17 compiler** (for core engine development)

### 1. Clone Repository
```bash
git clone https://github.com/satishgit44/trackpilot-ai-railway-controller.git
cd trackpilot-ai-railway-controller
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 3. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f trackpilot-core
```

### 4. Access Application
- **Frontend UI**: http://localhost:3000
- **Core API**: http://localhost:8080/api/docs
- **Grafana Dashboard**: http://localhost:3001 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090

### 5. Load Sample Data
```bash
# Load demo railway network and trains
docker-compose exec trackpilot-core /app/scripts/load-demo-data.sh

# Generate sample schedule
curl -X POST http://localhost:8080/api/schedule \
  -H "Content-Type: application/json" \
  -d '{"horizon_hours": 24}'
```

## 📦 Installation

<details>
<summary>🐋 Docker Installation (Recommended)</summary>

### Production Deployment
```bash
# Build and start production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Enable monitoring stack  
docker-compose --profile monitoring up -d

# Enable message queue
docker-compose --profile messaging up -d
```

### Development Environment
```bash  
# Start development stack with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access development tools
docker-compose exec trackpilot-core bash
```

</details>

<details>
<summary>💻 Local Development Installation</summary>

### Core Engine (C++)
```bash
cd core/

# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libboost-all-dev

# Download LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-*.zip

# Build
cmake -B build -DCMAKE_PREFIX_PATH=./libtorch
cmake --build build --parallel

# Run
./build/trackpilot_core --config config.json
```

### Python Training Environment  
```bash
cd python/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python training/train_rl_agent.py --config configs/ppo_config.json
```

### Frontend Development
```bash
cd web/

# Install dependencies  
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

</details>

## 🎮 Usage

### Basic Operations

#### 1. View Current Schedule
```bash
# Get current active schedule
curl -X GET http://localhost:8080/api/schedule/current

# Get specific time range
curl -X GET "http://localhost:8080/api/schedule?start=2024-01-15T10:00:00Z&end=2024-01-15T18:00:00Z"
```

#### 2. Add Trains
```bash
# Add new train to system
curl -X POST http://localhost:8080/api/trains \
  -H "Content-Type: application/json" \
  -d '{
    "train_id": "EXPRESS_123",
    "route": "Delhi-Mumbai", 
    "priority": 8,
    "speed_kmh": 120,
    "is_freight": false,
    "scheduled_arrival": "2024-01-15T14:30:00Z",
    "scheduled_departure": "2024-01-15T14:35:00Z"
  }'
```

#### 3. Generate New Schedule  
```bash
# Request AI-generated schedule
curl -X POST http://localhost:8080/api/schedule/generate \
  -H "Content-Type: application/json" \
  -d '{
    "horizon_hours": 24,
    "optimization_target": "throughput",
    "constraints": {
      "max_delay_minutes": 30,
      "priority_threshold": 5
    }
  }'
```

#### 4. Human Override
```bash
# Apply operator override  
curl -X POST http://localhost:8080/api/override \
  -H "Content-Type: application/json" \
  -d '{
    "train_id": "EXPRESS_123",
    "new_entry_time": "2024-01-15T14:40:00Z", 
    "reason": "safety_concern",
    "operator_id": "OP001",
    "feedback": "Adjusted for track maintenance window"
  }'
```

### Advanced Features

#### Real-time Updates via WebSocket
```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  
  switch(update.type) {
    case 'train_position':
      updateTrainPosition(update.data);
      break;
    case 'schedule_change':  
      refreshSchedule(update.data);
      break;
    case 'alert':
      showAlert(update.data);
      break;
  }
};
```

#### Performance Metrics
```bash
# Get system performance metrics
curl -X GET http://localhost:8080/api/metrics

# Response:
{
  "throughput": 45.2,
  "avg_delay_minutes": 3.7,
  "efficiency_score": 0.92,
  "safety_score": 0.98,
  "override_rate": 0.15,
  "last_updated": "2024-01-15T15:30:00Z"
}
```

#### Training Data Export
```bash  
# Export override data for retraining
curl -X POST http://localhost:8080/api/training/export \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-01-01",
    "end_date": "2024-01-31", 
    "format": "json",
    "include_metadata": true
  }'
```

## 📚 API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/trains` | List all trains |
| `POST` | `/api/trains` | Add new train |  
| `GET` | `/api/trains/{id}` | Get train details |
| `PUT` | `/api/trains/{id}` | Update train |
| `DELETE` | `/api/trains/{id}` | Remove train |
| `GET` | `/api/sections` | List railway sections |
| `POST` | `/api/schedule/generate` | Generate AI schedule |
| `GET` | `/api/schedule/current` | Get active schedule |
| `POST` | `/api/override` | Apply human override |
| `GET` | `/api/metrics` | System performance metrics |
| `GET` | `/api/health` | Health check endpoint |

### WebSocket Events

| Event Type | Description | Payload |
|------------|-------------|---------|
| `train_position` | Real-time position update | `{train_id, section_id, position_km, speed}` |
| `schedule_change` | Schedule modification | `{affected_trains, new_entries, reason}` |
| `alert` | System alert/warning | `{level, title, message, requires_action}` |
| `override_applied` | Human override event | `{operator_id, train_id, reason, timestamp}` |

### Authentication & Authorization

```bash
# Generate API token (if auth enabled)
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \  
  -d '{"username": "operator", "password": "password"}'

# Use token in requests
curl -X GET http://localhost:8080/api/trains \
  -H "Authorization: Bearer YOUR_TOKEN"
```

For complete API documentation, visit: http://localhost:8080/api/docs

## 🔧 Development

### Project Structure
```
trackpilot-ai-railway-controller/
├── core/                    # C++ Core Engine
│   ├── src/                 # Source code
│   ├── include/             # Header files  
│   ├── tests/               # Unit tests
│   └── CMakeLists.txt       # Build configuration
├── python/                  # AI Training Pipeline
│   ├── training/            # Training scripts
│   ├── models/              # Neural network models
│   ├── utils/               # Utility functions
│   └── requirements.txt     # Python dependencies
├── web/                     # React Frontend
│   ├── src/                 # Source code
│   ├── public/              # Static assets
│   ├── tests/               # Frontend tests  
│   └── package.json         # Node.js dependencies
├── infra/                   # Infrastructure & Deployment
│   ├── docker/              # Docker configurations
│   ├── k8s/                 # Kubernetes manifests
│   └── scripts/             # Deployment scripts
├── .github/                 # CI/CD Workflows
│   └── workflows/           # GitHub Actions
└── docs/                    # Documentation
    ├── api/                 # API documentation
    ├── architecture/        # System design
    └── user-guide/          # User manuals
```

### Development Workflow

1. **Setup Development Environment**
   ```bash
   # Clone repository
   git clone https://github.com/satishgit44/trackpilot-ai-railway-controller.git
   cd trackpilot-ai-railway-controller
   
   # Setup development containers
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Core Engine Development**
   ```bash
   # Enter development container
   docker-compose exec trackpilot-core bash
   
   # Build with debug symbols
   cd /src/core
   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build
   
   # Run tests
   ctest --test-dir build --verbose
   ```

3. **Python Model Development**  
   ```bash
   # Activate development environment
   docker-compose exec ai-trainer bash
   
   # Install development dependencies
   pip install pytest black flake8 jupyter
   
   # Run training experiment
   python training/train_rl_agent.py --config configs/dev_config.json
   ```

4. **Frontend Development**
   ```bash
   # Start development server with hot reload
   cd web/
   npm run dev
   
   # Run tests
   npm test
   
   # Type checking
   npm run type-check
   ```

### Code Style & Guidelines

- **C++**: Follow Google C++ Style Guide, use clang-format
- **Python**: Follow PEP 8, use Black formatter  
- **TypeScript/React**: Use Prettier, ESLint rules
- **Commit Messages**: Follow Conventional Commits specification

### Testing

```bash
# Run all tests
./scripts/run-tests.sh

# Individual component tests
docker-compose exec trackpilot-core ctest --verbose
docker-compose exec ai-trainer pytest -v  
docker-compose exec web npm test

# Integration tests
./scripts/integration-tests.sh
```

### Performance Profiling

```bash
# C++ Performance Profiling
docker-compose exec trackpilot-core valgrind --tool=callgrind ./build/trackpilot_core

# Python Memory Profiling  
docker-compose exec ai-trainer python -m memory_profiler training/train_rl_agent.py

# Frontend Bundle Analysis
cd web/ && npm run analyze
```

## 🚀 Deployment

### Production Deployment

<details>
<summary>🐋 Docker Deployment</summary>

```bash
# 1. Clone repository on production server
git clone https://github.com/satishgit44/trackpilot-ai-railway-controller.git
cd trackpilot-ai-railway-controller

# 2. Configure production environment
cp .env.example .env
# Edit .env with production values

# 3. Deploy with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 4. Setup SSL certificates (optional)
./scripts/setup-ssl.sh

# 5. Configure backup
./scripts/setup-backup.sh
```

</details>

<details>
<summary>☸️ Kubernetes Deployment</summary>

```bash  
# 1. Setup cluster configuration
kubectl apply -f infra/k8s/namespace.yaml

# 2. Deploy database and cache
kubectl apply -f infra/k8s/postgres.yaml
kubectl apply -f infra/k8s/redis.yaml  

# 3. Deploy application
kubectl apply -f infra/k8s/trackpilot-core.yaml
kubectl apply -f infra/k8s/service.yaml
kubectl apply -f infra/k8s/ingress.yaml

# 4. Monitor deployment
kubectl get pods -n trackpilot
kubectl logs -f deployment/trackpilot-core -n trackpilot
```

</details>

<details>
<summary>☁️ Cloud Deployment</summary>

#### AWS ECS
```bash
# Deploy using AWS Copilot
copilot app init trackpilot
copilot env init --name production  
copilot svc init --name core
copilot svc deploy --name core --env production
```

#### Google Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/trackpilot
gcloud run deploy --image gcr.io/PROJECT_ID/trackpilot --platform managed
```

#### Azure Container Instances
```bash
# Deploy to ACI
az container create \
  --resource-group trackpilot-rg \
  --name trackpilot-core \  
  --image trackpilot/core:latest \
  --ports 80 8080
```

</details>

### Environment Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NODE_ENV` | Environment mode | `production` | No |
| `DB_PASSWORD` | PostgreSQL password | `trackpilot123` | Yes |
| `REDIS_PASSWORD` | Redis password | `redis123` | Yes |
| `JWT_SECRET` | JWT signing secret | - | Yes |
| `CORS_ORIGINS` | Allowed CORS origins | `*` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `MONITORING_ENABLED` | Enable Prometheus metrics | `true` | No |
| `WANDB_API_KEY` | Weights & Biases API key | - | No |

### Scaling Configuration

#### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  trackpilot-core:
    deploy:
      replicas: 3
      
  nginx:
    volumes:
      - ./nginx/load-balancer.conf:/etc/nginx/nginx.conf
```

#### Vertical Scaling  
```yaml
# Resource allocation for high-traffic scenarios
services:
  trackpilot-core:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
        reservations:  
          memory: 2G
          cpus: "1.0"
```

### Monitoring & Alerting

#### Prometheus Alerts
```yaml
# monitoring/alerts.yml  
groups:
  - name: trackpilot
    rules:
      - alert: HighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 1.0
        for: 5m
        annotations:
          summary: "High API response time detected"
          
      - alert: LowSchedulingConfidence  
        expr: scheduling_confidence < 0.7
        for: 2m
        annotations:
          summary: "AI scheduling confidence below threshold"
```

#### Health Checks
```bash
# System health monitoring
curl http://localhost:8080/api/health

# Expected response:
{
  "status": "healthy",
  "components": {
    "database": "healthy", 
    "cache": "healthy",
    "ai_model": "healthy"
  },
  "uptime": "24h 15m 32s",
  "version": "1.0.0"
}
```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. **Fork the repository**
   ```bash
   gh repo fork satishgit44/trackpilot-ai-railway-controller
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-enhancement
   ```

3. **Make changes and test**
   ```bash
   # Make your changes
   ./scripts/run-tests.sh
   ./scripts/lint-check.sh
   ```

4. **Commit and push**
   ```bash
   git commit -m "feat: add amazing enhancement"
   git push origin feature/amazing-enhancement
   ```

5. **Create Pull Request**
   ```bash  
   gh pr create --title "Add amazing enhancement" --body "Description of changes"
   ```

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation for API changes  
- Ensure CI pipeline passes
- Add appropriate commit messages

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Provide detailed reproduction steps
- Include system information and logs
- Label issues appropriately

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Smart India Hackathon 2025** for the problem statement and opportunity
- **PyTorch Team** for the excellent deep learning framework  
- **D3.js Community** for powerful visualization capabilities
- **Railway Domain Experts** for valuable insights and feedback

## 📞 Contact & Support

- **Team**: Team Neuronauts
- **Repository**: https://github.com/satishgit44/trackpilot-ai-railway-controller
- **Issues**: https://github.com/satishgit44/trackpilot-ai-railway-controller/issues
- **Discussions**: https://github.com/satishgit44/trackpilot-ai-railway-controller/discussions

---

**Built with ❤️ by Team Neuronauts for Smart India Hackathon 2025**

*Revolutionizing Railway Traffic Control with AI-Human Collaboration*