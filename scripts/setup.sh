#!/bin/bash

# Smart Crowd Intelligence System Setup Script
# This script sets up the complete environment for the SCI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root"
   exit 1
fi

log_info "Starting Smart Crowd Intelligence System Setup..."

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"
DEV_COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.dev.yml"

# Parse command line arguments
ENVIRONMENT="production"
SKIP_GPU_CHECK=false
INSTALL_DEPS=true
BUILD_IMAGES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            ENVIRONMENT="development"
            shift
            ;;
        --skip-gpu)
            SKIP_GPU_CHECK=true
            shift
            ;;
        --no-deps)
            INSTALL_DEPS=false
            shift
            ;;
        --no-build)
            BUILD_IMAGES=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dev, --development    Setup for development environment"
            echo "  --skip-gpu             Skip GPU availability check"
            echo "  --no-deps              Skip dependency installation"
            echo "  --no-build             Skip Docker image building"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Setting up for $ENVIRONMENT environment"

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check operating system
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_warning "This system is designed for Linux. Some features may not work on $OSTYPE"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        log_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        log_info "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if [[ "$SKIP_GPU_CHECK" == false ]]; then
        if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            log_warning "NVIDIA Docker runtime not available. GPU acceleration will be disabled."
            log_info "To enable GPU support, install nvidia-docker2:"
            log_info "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        else
            log_success "NVIDIA Docker runtime detected"
        fi
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 8 ]]; then
        log_warning "System has less than 8GB RAM. Performance may be affected."
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG "$PROJECT_DIR" | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        log_warning "Less than 10GB disk space available. Consider freeing up space."
    fi
    
    log_success "System requirements check completed"
}

# Install system dependencies
install_dependencies() {
    if [[ "$INSTALL_DEPS" == false ]]; then
        log_info "Skipping dependency installation"
        return
    fi
    
    log_info "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        curl \
        wget \
        git \
        unzip \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        pkg-config \
        libcairo2-dev \
        libgirepository1.0-dev \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav
    
    log_success "System dependencies installed"
}

# Setup environment file
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating .env file from template..."
        cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
        
        # Generate random passwords
        DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
        API_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        JWT_SECRET=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
        
        # Replace placeholders in .env file
        sed -i "s/YOUR_DB_PASSWORD/$DB_PASSWORD/g" "$ENV_FILE"
        sed -i "s/YOUR_REDIS_PASSWORD/$REDIS_PASSWORD/g" "$ENV_FILE"
        sed -i "s/your-secret-key-here-change-this-in-production/$SECRET_KEY/g" "$ENV_FILE"
        sed -i "s/your-api-key-here/$API_KEY/g" "$ENV_FILE"
        sed -i "s/your-jwt-secret-here/$JWT_SECRET/g" "$ENV_FILE"
        
        # Set environment-specific values
        if [[ "$ENVIRONMENT" == "development" ]]; then
            sed -i "s/ENVIRONMENT=production/ENVIRONMENT=development/g" "$ENV_FILE"
            sed -i "s/DEBUG=false/DEBUG=true/g" "$ENV_FILE"
            sed -i "s/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/g" "$ENV_FILE"
        fi
        
        log_success "Environment file created with secure random passwords"
        log_warning "Please review and update $ENV_FILE with your specific configuration"
    else
        log_info "Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/uploads"
    mkdir -p "$PROJECT_DIR/backups"
    mkdir -p "$PROJECT_DIR/config/local"
    
    # Set proper permissions
    chmod 755 "$PROJECT_DIR/logs"
    chmod 755 "$PROJECT_DIR/data"
    chmod 755 "$PROJECT_DIR/uploads"
    
    log_success "Directories created"
}

# Download pre-trained models (placeholder function)
download_models() {
    log_info "Setting up AI models..."
    
    MODELS_DIR="$PROJECT_DIR/models"
    
    # Create placeholder model files for development
    if [[ "$ENVIRONMENT" == "development" ]]; then
        touch "$MODELS_DIR/crowd_detection_model.pth"
        touch "$MODELS_DIR/density_estimation_model.pth"
        touch "$MODELS_DIR/behavior_analysis_model.pth"
        
        log_info "Placeholder model files created for development"
        log_warning "Replace with actual trained models for production use"
    else
        log_warning "Please place your trained models in $MODELS_DIR/"
        log_info "Required models:"
        log_info "  - crowd_detection_model.pth"
        log_info "  - density_estimation_model.pth"
        log_info "  - behavior_analysis_model.pth"
    fi
    
    log_success "Model setup completed"
}

# Build Docker images
build_images() {
    if [[ "$BUILD_IMAGES" == false ]]; then
        log_info "Skipping Docker image building"
        return
    fi
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        docker-compose -f "$DEV_COMPOSE_FILE" build
    else
        docker-compose -f "$DOCKER_COMPOSE_FILE" build
    fi
    
    log_success "Docker images built successfully"
}

# Initialize database
init_database() {
    log_info "Initializing database..."
    
    cd "$PROJECT_DIR"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        COMPOSE_FILE="$DEV_COMPOSE_FILE"
    else
        COMPOSE_FILE="$DOCKER_COMPOSE_FILE"
    fi
    
    # Start database services
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    sleep 10
    
    # Run database migrations (if they exist)
    # docker-compose -f "$COMPOSE_FILE" exec sci_api alembic upgrade head
    
    log_success "Database initialized"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring configuration directories
    mkdir -p "$PROJECT_DIR/docker/monitoring"
    mkdir -p "$PROJECT_DIR/docker/monitoring/grafana/dashboards"
    mkdir -p "$PROJECT_DIR/docker/monitoring/grafana/datasources"
    
    # Create basic Prometheus configuration
    cat > "$PROJECT_DIR/docker/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sci-api'
    static_configs:
      - targets: ['sci_api:9090']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF
    
    log_success "Monitoring setup completed"
}

# Setup SSL certificates (development only)
setup_ssl() {
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log_info "Setting up development SSL certificates..."
        
        SSL_DIR="$PROJECT_DIR/docker/ssl"
        mkdir -p "$SSL_DIR"
        
        # Generate self-signed certificate for development
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/key.pem" \
            -out "$SSL_DIR/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log_success "Development SSL certificates created"
        log_warning "Do not use these certificates in production!"
    fi
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    cd "$PROJECT_DIR"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        COMPOSE_FILE="$DEV_COMPOSE_FILE"
        API_PORT="8001"
    else
        COMPOSE_FILE="$DOCKER_COMPOSE_FILE"
        API_PORT="8000"
    fi
    
    # Start all services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check API health
    if curl -f "http://localhost:$API_PORT/health" &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        docker-compose -f "$COMPOSE_FILE" logs sci_api
        return 1
    fi
    
    # Check WebSocket connection
    if curl -f "http://localhost/ws" &> /dev/null; then
        log_success "WebSocket endpoint accessible"
    else
        log_warning "WebSocket endpoint check failed (this is normal if NGINX is not running)"
    fi
    
    log_success "Installation validation completed"
}

# Display completion message
show_completion_message() {
    log_success "Smart Crowd Intelligence System setup completed!"
    echo
    log_info "Next steps:"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        echo "  1. API: http://localhost:8001"
        echo "  2. API Documentation: http://localhost:8001/docs"
        echo "  3. Database Admin: http://localhost:5050"
        echo "  4. Redis Admin: http://localhost:8082"
        echo "  5. Jupyter: http://localhost:8888 (token: development_token)"
    else
        echo "  1. Dashboard: http://localhost"
        echo "  2. API: http://localhost:8000"
        echo "  3. API Documentation: http://localhost:8000/docs"
        echo "  4. Monitoring: http://localhost:9090 (Prometheus)"
        echo "  5. Dashboards: http://localhost:3000 (Grafana)"
        echo "  6. Logs: http://localhost:5601 (Kibana)"
    fi
    
    echo
    log_info "Configuration files:"
    echo "  - Environment: $ENV_FILE"
    echo "  - Configuration: $PROJECT_DIR/config/config.yaml"
    echo "  - Docker Compose: $COMPOSE_FILE"
    
    echo
    log_info "Useful commands:"
    echo "  - Start services: docker-compose -f $COMPOSE_FILE up -d"
    echo "  - Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  - View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  - Rebuild: docker-compose -f $COMPOSE_FILE build"
    
    echo
    log_warning "Remember to:"
    echo "  1. Update $ENV_FILE with your specific configuration"
    echo "  2. Place trained models in $PROJECT_DIR/models/"
    echo "  3. Configure alert settings in config.yaml"
    echo "  4. Set up proper SSL certificates for production"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "  5. Review security settings before production deployment"
        echo "  6. Configure backup procedures"
        echo "  7. Set up monitoring alerts"
    fi
}

# Main execution
main() {
    check_system_requirements
    install_dependencies
    setup_environment
    create_directories
    download_models
    setup_monitoring
    setup_ssl
    build_images
    init_database
    validate_installation
    show_completion_message
}

# Run main function
main "$@"