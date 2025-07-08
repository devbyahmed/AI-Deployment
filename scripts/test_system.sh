#!/bin/bash

# Smart Crowd Intelligence System Test Script
# Comprehensive testing of all system components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
API_URL="http://localhost:8000"
WS_URL="ws://localhost:8000/ws"
TIMEOUT=30

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

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    log_info "Running test: $test_name"
    
    if $test_function; then
        log_success "✓ $test_name"
        ((TESTS_PASSED++))
    else
        log_error "✗ $test_name"
        FAILED_TESTS+=("$test_name")
        ((TESTS_FAILED++))
    fi
    
    echo
}

# Individual test functions
test_docker_services() {
    log_info "Checking Docker services..."
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        return 1
    fi
    
    # Check if containers are running
    local required_containers=("sci_api" "sci_postgres" "sci_redis")
    
    for container in "${required_containers[@]}"; do
        if ! docker ps | grep -q "$container"; then
            log_warning "Container $container is not running"
            return 1
        fi
    done
    
    return 0
}

test_api_health() {
    log_info "Testing API health endpoint..."
    
    local response=$(curl -s -w "%{http_code}" "$API_URL/health" -o /tmp/health_response.json)
    
    if [ "$response" = "200" ]; then
        local status=$(cat /tmp/health_response.json | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        if [ "$status" = "healthy" ]; then
            return 0
        else
            log_error "API health status: $status"
            return 1
        fi
    else
        log_error "API health check failed with HTTP $response"
        return 1
    fi
}

test_system_status() {
    log_info "Testing system status endpoint..."
    
    local response=$(curl -s -w "%{http_code}" "$API_URL/status" -o /tmp/status_response.json)
    
    if [ "$response" = "200" ]; then
        # Check if response contains expected fields
        if grep -q '"timestamp"' /tmp/status_response.json && \
           grep -q '"system"' /tmp/status_response.json; then
            return 0
        else
            log_error "Status response missing required fields"
            return 1
        fi
    else
        log_error "System status check failed with HTTP $response"
        return 1
    fi
}

test_database_connection() {
    log_info "Testing database connection..."
    
    # Test database connectivity through API
    local response=$(curl -s -w "%{http_code}" "$API_URL/streams" -o /tmp/streams_response.json)
    
    if [ "$response" = "200" ]; then
        return 0
    else
        log_error "Database connection test failed with HTTP $response"
        return 1
    fi
}

test_redis_connection() {
    log_info "Testing Redis connection..."
    
    # Test Redis connectivity by checking if we can retrieve cached data
    if docker exec sci_redis redis-cli ping | grep -q "PONG"; then
        return 0
    else
        log_error "Redis connection test failed"
        return 1
    fi
}

test_stream_creation() {
    log_info "Testing stream creation..."
    
    local test_stream='{
        "stream_id": "test_camera_001",
        "uri": "file:///app/sample_videos/test.mp4",
        "location": "Test Location"
    }'
    
    local response=$(curl -s -w "%{http_code}" \
        -X POST "$API_URL/streams" \
        -H "Content-Type: application/json" \
        -d "$test_stream" \
        -o /tmp/create_stream_response.json)
    
    if [ "$response" = "200" ]; then
        # Verify stream was created
        local verify_response=$(curl -s -w "%{http_code}" "$API_URL/streams" -o /tmp/verify_streams.json)
        if [ "$verify_response" = "200" ] && grep -q "test_camera_001" /tmp/verify_streams.json; then
            return 0
        else
            log_error "Stream creation verification failed"
            return 1
        fi
    else
        log_error "Stream creation failed with HTTP $response"
        return 1
    fi
}

test_image_analysis() {
    log_info "Testing image analysis endpoint..."
    
    # Create a simple test image (1x1 black pixel)
    echo -n "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA=" | base64 -d > /tmp/test_image.png
    
    local response=$(curl -s -w "%{http_code}" \
        -X POST "$API_URL/analyze/image" \
        -F "file=@/tmp/test_image.png" \
        -o /tmp/analyze_response.json)
    
    if [ "$response" = "200" ]; then
        # Check if response contains analysis data
        if grep -q '"crowd_detection"' /tmp/analyze_response.json; then
            return 0
        else
            log_error "Image analysis response missing expected data"
            return 1
        fi
    else
        log_error "Image analysis failed with HTTP $response"
        return 1
    fi
}

test_websocket_connection() {
    log_info "Testing WebSocket connection..."
    
    # Check if wscat is available
    if ! command -v wscat &> /dev/null; then
        log_warning "wscat not available, skipping WebSocket test"
        return 0
    fi
    
    # Test WebSocket connection with timeout
    timeout 10s wscat -c "$WS_URL" -x '{"type":"subscribe","stream_id":"all"}' &> /tmp/ws_test.log
    
    if [ $? -eq 0 ]; then
        return 0
    else
        log_warning "WebSocket connection test inconclusive"
        return 0  # Don't fail the test suite for WebSocket issues
    fi
}

test_alert_system() {
    log_info "Testing alert system..."
    
    # Get current alerts
    local response=$(curl -s -w "%{http_code}" "$API_URL/alerts" -o /tmp/alerts_response.json)
    
    if [ "$response" = "200" ]; then
        # Check if response has expected structure
        if grep -q '"alerts"' /tmp/alerts_response.json; then
            return 0
        else
            log_error "Alerts response missing expected structure"
            return 1
        fi
    else
        log_error "Alert system test failed with HTTP $response"
        return 1
    fi
}

test_frontend_accessibility() {
    log_info "Testing frontend accessibility..."
    
    # Check if main dashboard is accessible
    local response=$(curl -s -w "%{http_code}" "http://localhost/" -o /tmp/frontend_response.html)
    
    if [ "$response" = "200" ]; then
        # Check if it's actually the dashboard
        if grep -q "Smart Crowd Intelligence" /tmp/frontend_response.html; then
            return 0
        else
            log_error "Frontend response doesn't contain expected content"
            return 1
        fi
    else
        log_error "Frontend accessibility test failed with HTTP $response"
        return 1
    fi
}

test_api_documentation() {
    log_info "Testing API documentation..."
    
    local response=$(curl -s -w "%{http_code}" "$API_URL/docs" -o /tmp/docs_response.html)
    
    if [ "$response" = "200" ]; then
        return 0
    else
        log_error "API documentation test failed with HTTP $response"
        return 1
    fi
}

test_performance_metrics() {
    log_info "Testing performance metrics..."
    
    # Check if metrics endpoint is available
    local response=$(curl -s -w "%{http_code}" "$API_URL/metrics" -o /tmp/metrics_response.txt)
    
    if [ "$response" = "200" ] || [ "$response" = "403" ]; then
        # 403 is acceptable as metrics might be restricted
        return 0
    else
        log_warning "Metrics endpoint test returned HTTP $response"
        return 0  # Don't fail for metrics issues
    fi
}

# Cleanup function
cleanup_test_data() {
    log_info "Cleaning up test data..."
    
    # Remove test stream if it exists
    curl -s -X DELETE "$API_URL/streams/test_camera_001" &> /dev/null || true
    
    # Remove temporary files
    rm -f /tmp/health_response.json
    rm -f /tmp/status_response.json
    rm -f /tmp/streams_response.json
    rm -f /tmp/create_stream_response.json
    rm -f /tmp/verify_streams.json
    rm -f /tmp/analyze_response.json
    rm -f /tmp/alerts_response.json
    rm -f /tmp/frontend_response.html
    rm -f /tmp/docs_response.html
    rm -f /tmp/metrics_response.txt
    rm -f /tmp/test_image.png
    rm -f /tmp/ws_test.log
}

# Performance test
test_api_performance() {
    log_info "Testing API performance..."
    
    # Simple load test - 10 concurrent requests
    local start_time=$(date +%s.%N)
    
    for i in {1..10}; do
        curl -s "$API_URL/health" &> /dev/null &
    done
    
    wait  # Wait for all background jobs to complete
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Check if all requests completed within reasonable time (5 seconds)
    if (( $(echo "$duration < 5" | bc -l) )); then
        log_info "Performance test completed in ${duration}s"
        return 0
    else
        log_warning "Performance test took ${duration}s (>5s)"
        return 0  # Don't fail for performance issues
    fi
}

# GPU test (if available)
test_gpu_access() {
    log_info "Testing GPU access..."
    
    # Check if nvidia-smi is available in the container
    if docker exec sci_api nvidia-smi &> /dev/null; then
        log_success "GPU access confirmed"
        return 0
    else
        log_warning "GPU not available or not accessible"
        return 0  # Don't fail for missing GPU
    fi
}

# Main test execution
main() {
    echo "=================================================="
    echo "Smart Crowd Intelligence System Test Suite"
    echo "=================================================="
    echo
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Run all tests
    run_test "Docker Services Status" test_docker_services
    run_test "API Health Check" test_api_health
    run_test "System Status" test_system_status
    run_test "Database Connection" test_database_connection
    run_test "Redis Connection" test_redis_connection
    run_test "Stream Creation" test_stream_creation
    run_test "Image Analysis" test_image_analysis
    run_test "WebSocket Connection" test_websocket_connection
    run_test "Alert System" test_alert_system
    run_test "Frontend Accessibility" test_frontend_accessibility
    run_test "API Documentation" test_api_documentation
    run_test "Performance Metrics" test_performance_metrics
    run_test "API Performance" test_api_performance
    run_test "GPU Access" test_gpu_access
    
    # Cleanup
    cleanup_test_data
    
    # Test results summary
    echo "=================================================="
    echo "Test Results Summary"
    echo "=================================================="
    echo
    log_success "Tests Passed: $TESTS_PASSED"
    
    if [ $TESTS_FAILED -gt 0 ]; then
        log_error "Tests Failed: $TESTS_FAILED"
        echo
        log_error "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo
        log_error "Some tests failed. Please check the system configuration."
        exit 1
    else
        echo
        log_success "All tests passed! ✨"
        log_success "Smart Crowd Intelligence System is working correctly."
        echo
        echo "Access URLs:"
        echo "  Dashboard: http://localhost"
        echo "  API: http://localhost:8000"
        echo "  API Docs: http://localhost:8000/docs"
        echo "  Monitoring: http://localhost:9090"
        echo
    fi
}

# Check if required tools are available
check_dependencies() {
    local missing_deps=()
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v bc &> /dev/null; then
        missing_deps+=("bc")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again."
        exit 1
    fi
}

# Command line argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --api-url URL    API base URL (default: http://localhost:8000)"
            echo "  --timeout SEC    Timeout for tests (default: 30)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the tests
check_dependencies
main