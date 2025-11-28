#!/bin/bash

# Trinity KM AI Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    local dockerfile="${1:-Dockerfile}"
    print_status "Building Trinity KM AI Docker image using ${dockerfile}..."
    docker build -f "${dockerfile}" -t trinity-km-ai:latest .
    print_status "Image built successfully!"
}

# Function to build with Ubuntu base (fallback for gpt4all issues)
build_ubuntu() {
    print_status "Building with Ubuntu base image for better library support..."
    docker build -f Dockerfile.ubuntu -t trinity-km-ai-ubuntu:latest .
    print_status "Ubuntu-based image built successfully!"
}

# Function to run with Docker Compose
run_compose() {
    print_status "Starting Trinity KM AI with Docker Compose..."
    docker-compose up -d
    print_status "Services started! Access the application at http://localhost:8501"
    print_status "Ollama is available at http://localhost:11434"
}

# Function to run standalone container
run_standalone() {
    print_status "Running Trinity KM AI standalone container..."
    
    # Create necessary volumes
    docker volume create trinity-data 2>/dev/null || true
    
    docker run -d \
        --name trinity-km-ai \
        -p 8501:8501 \
        -v trinity-data:/app/data \
        -v "$(pwd)/config.json:/app/config.json:ro" \
        --restart unless-stopped \
        trinity-km-ai:latest
    
    print_status "Container started! Access the application at http://localhost:8501"
}

# Function to stop services
stop_services() {
    print_status "Stopping Trinity KM AI services..."
    docker-compose down 2>/dev/null || true
    docker stop trinity-km-ai 2>/dev/null || true
    docker rm trinity-km-ai 2>/dev/null || true
    print_status "Services stopped."
}

# Function to view logs
view_logs() {
    if docker ps --format "table {{.Names}}" | grep -q "trinity-km-ai"; then
        print_status "Showing logs for Trinity KM AI..."
        docker logs -f trinity-km-ai
    elif docker-compose ps | grep -q "trinity-km-ai"; then
        print_status "Showing logs for Trinity KM AI..."
        docker-compose logs -f trinity-km-ai
    else
        print_error "No running Trinity KM AI container found."
    fi
}

# Function to pull models for Ollama
setup_ollama() {
    print_status "Setting up Ollama models..."
    
    # Check if Ollama container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "ollama"; then
        print_warning "Ollama container is not running. Starting it first..."
        docker-compose up -d ollama
        sleep 10
    fi
    
    # Pull the default model
    print_status "Pulling gemma model..."
    docker exec trinity-km-ai_ollama_1 ollama pull gemma
    
    print_status "Ollama setup complete!"
}

# Function to show status
show_status() {
    print_status "Trinity KM AI Container Status:"
    echo
    
    # Check standalone container
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep trinity-km-ai; then
        echo
    fi
    
    # Check compose services
    if command -v docker-compose >/dev/null 2>&1; then
        print_status "Docker Compose Services:"
        docker-compose ps 2>/dev/null || echo "No compose services running"
    fi
}

# Main script logic
case "${1:-}" in
    "build")
        check_docker
        build_image "${2:-}"
        ;;
    "build-ubuntu")
        check_docker
        build_ubuntu
        ;;
    "run")
        check_docker
        build_image "${3:-}"
        if [[ "${2:-}" == "compose" ]]; then
            run_compose
        else
            run_standalone
        fi
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        view_logs
        ;;
    "status")
        show_status
        ;;
    "ollama")
        setup_ollama
        ;;
    "restart")
        stop_services
        sleep 2
        if [[ "${2:-}" == "compose" ]]; then
            run_compose
        else
            build_image "${3:-}"
            run_standalone
        fi
        ;;
    *)
        echo "Trinity KM AI Docker Management Script"
        echo
        echo "Usage: $0 {build|build-ubuntu|run|stop|logs|status|ollama|restart} [compose] [dockerfile]"
        echo
        echo "Commands:"
        echo "  build              Build the Docker image"
        echo "  build-ubuntu       Build with Ubuntu base (for gpt4all issues)"
        echo "  run                Build and run the container (add 'compose' for full stack)"
        echo "  run compose        Run with Docker Compose (includes Ollama)"
        echo "  stop               Stop all services"
        echo "  logs               View application logs"
        echo "  status             Show container status"
        echo "  ollama             Setup Ollama models"
        echo "  restart            Restart services (add 'compose' for full stack)"
        echo
        echo "Examples:"
        echo "  $0 build                     # Build with default Dockerfile"
        echo "  $0 build-ubuntu             # Build with Ubuntu base"
        echo "  $0 run                       # Run standalone"
        echo "  $0 run compose              # Run with Ollama"
        echo "  $0 restart compose          # Restart full stack"
        exit 1
        ;;
esac
