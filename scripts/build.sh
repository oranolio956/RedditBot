#!/bin/bash

# AI Conversation System - Docker Build and CI/CD Script
# Builds Docker images for all services with multi-platform support

set -euo pipefail

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_PLATFORM="${BUILD_PLATFORM:-linux/amd64,linux/arm64}"
PUSH_IMAGES="${PUSH_IMAGES:-false}"
BUILD_GPU="${BUILD_GPU:-false}"
BUILD_TYPE="${BUILD_TYPE:-production}"
CACHE_FROM="${CACHE_FROM:-true}"
NO_CACHE="${NO_CACHE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== Helper Functions ====================
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

show_help() {
    cat << EOF
AI Conversation System - Build Script

Usage: $0 [OPTIONS] [TARGETS...]

Targets:
  main             Build main application image (default)
  gpu              Build GPU-enabled application image
  monitoring       Build monitoring components
  all              Build all images

Options:
  -r, --registry URL       Docker registry URL
  -t, --tag TAG           Docker image tag (default: latest)
  -p, --platform PLATFORMS Multi-platform build (default: linux/amd64,linux/arm64)
  --push                  Push images to registry
  --gpu                   Build GPU-enabled images
  --build-type TYPE       Build type: production, development (default: production)
  --cache-from            Use build cache (default: true)
  --no-cache              Disable build cache
  -h, --help              Show this help message

Environment Variables:
  DOCKER_REGISTRY         Docker registry URL
  IMAGE_TAG              Docker image tag
  BUILD_PLATFORM         Target platforms for build
  DOCKER_BUILDKIT        Enable Docker BuildKit (recommended)
  CI                     Running in CI environment

Examples:
  $0                      # Build main application
  $0 --gpu --push        # Build and push GPU images
  $0 all --tag v1.2.3    # Build all images with specific tag
  $0 --registry my-registry.com --push

EOF
}

check_dependencies() {
    log "Checking build dependencies..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
    fi
    
    # Check Docker buildx
    if ! docker buildx version &> /dev/null; then
        error "Docker buildx is required for multi-platform builds"
    fi
    
    # Setup buildx builder
    if ! docker buildx inspect multiarch-builder &> /dev/null; then
        log "Creating multiarch builder..."
        docker buildx create --name multiarch-builder --platform "$BUILD_PLATFORM" --use
    else
        docker buildx use multiarch-builder
    fi
    
    success "Dependencies checked"
}

get_build_info() {
    # Get git information
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Use git tag as image tag if available
    if [ -n "$GIT_TAG" ] && [ "$IMAGE_TAG" = "latest" ]; then
        IMAGE_TAG="$GIT_TAG"
    fi
    
    log "Build Information:"
    log "  Git Commit: $GIT_COMMIT"
    log "  Git Branch: $GIT_BRANCH"
    log "  Git Tag: ${GIT_TAG:-none}"
    log "  Build Date: $BUILD_DATE"
    log "  Image Tag: $IMAGE_TAG"
}

prepare_build_args() {
    BUILD_ARGS=(
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "GIT_COMMIT=$GIT_COMMIT"
        --build-arg "GIT_BRANCH=$GIT_BRANCH"
        --build-arg "VERSION=$IMAGE_TAG"
        --build-arg "BUILDPLATFORM=$BUILD_PLATFORM"
    )
    
    if [ "$CACHE_FROM" = "true" ] && [ "$NO_CACHE" = "false" ]; then
        if [ -n "$DOCKER_REGISTRY" ]; then
            BUILD_ARGS+=(--cache-from "type=registry,ref=$DOCKER_REGISTRY/ai-conversation:cache")
            BUILD_ARGS+=(--cache-to "type=registry,ref=$DOCKER_REGISTRY/ai-conversation:cache,mode=max")
        else
            BUILD_ARGS+=(--cache-from "type=gha")
            BUILD_ARGS+=(--cache-to "type=gha,mode=max")
        fi
    fi
    
    if [ "$NO_CACHE" = "true" ]; then
        BUILD_ARGS+=(--no-cache)
    fi
    
    if [ "$PUSH_IMAGES" = "true" ]; then
        BUILD_ARGS+=(--push)
    else
        BUILD_ARGS+=(--load)
    fi
}

build_main_image() {
    local image_name="ai-conversation"
    local full_image_name="$image_name:$IMAGE_TAG"
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        full_image_name="$DOCKER_REGISTRY/$full_image_name"
    fi
    
    log "Building main application image: $full_image_name"
    
    docker buildx build \
        --platform "$BUILD_PLATFORM" \
        --target "$BUILD_TYPE" \
        --tag "$full_image_name" \
        --tag "${full_image_name%:*}:latest" \
        --file "$PROJECT_ROOT/Dockerfile" \
        "${BUILD_ARGS[@]}" \
        "$PROJECT_ROOT"
    
    success "Built main application image"
}

build_gpu_image() {
    local image_name="ai-conversation-gpu"
    local full_image_name="$image_name:$IMAGE_TAG"
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        full_image_name="$DOCKER_REGISTRY/$full_image_name"
    fi
    
    log "Building GPU-enabled image: $full_image_name"
    
    docker buildx build \
        --platform "linux/amd64" \
        --target "gpu-production" \
        --tag "$full_image_name" \
        --tag "${full_image_name%:*}:latest" \
        --file "$PROJECT_ROOT/Dockerfile.gpu" \
        "${BUILD_ARGS[@]}" \
        "$PROJECT_ROOT"
    
    success "Built GPU-enabled image"
}

build_monitoring_images() {
    log "Building monitoring images..."
    
    # Custom Prometheus with configuration
    local prometheus_image="ai-conversation-prometheus:$IMAGE_TAG"
    if [ -n "$DOCKER_REGISTRY" ]; then
        prometheus_image="$DOCKER_REGISTRY/$prometheus_image"
    fi
    
    cat > "$PROJECT_ROOT/monitoring.Dockerfile" << 'EOF'
FROM prom/prometheus:v2.47.0
COPY monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
COPY monitoring/prometheus/rules /etc/prometheus/rules
USER prometheus
EOF
    
    docker buildx build \
        --platform "$BUILD_PLATFORM" \
        --tag "$prometheus_image" \
        --file "$PROJECT_ROOT/monitoring.Dockerfile" \
        "${BUILD_ARGS[@]}" \
        "$PROJECT_ROOT"
    
    rm -f "$PROJECT_ROOT/monitoring.Dockerfile"
    
    success "Built monitoring images"
}

run_security_scan() {
    local image_name="${1:-ai-conversation:$IMAGE_TAG}"
    
    log "Running security scan on $image_name..."
    
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL --exit-code 1 "$image_name" || warn "Security issues found in $image_name"
    elif command -v docker &> /dev/null && docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image --severity HIGH,CRITICAL "$image_name" &> /dev/null; then
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image --severity HIGH,CRITICAL --exit-code 1 "$image_name" || warn "Security issues found in $image_name"
    else
        warn "Trivy not available, skipping security scan"
    fi
}

test_images() {
    log "Testing built images..."
    
    # Test main image
    local test_image="ai-conversation:$IMAGE_TAG"
    if [ -n "$DOCKER_REGISTRY" ]; then
        test_image="$DOCKER_REGISTRY/$test_image"
    fi
    
    # Basic smoke test
    log "Running smoke test on $test_image..."
    docker run --rm --entrypoint="" "$test_image" python -c "import app; print('Import successful')" || error "Smoke test failed"
    
    # Health check test
    log "Testing health check endpoint..."
    CONTAINER_ID=$(docker run -d -p 0:8000 "$test_image")
    sleep 10
    
    PORT=$(docker port "$CONTAINER_ID" 8000 | cut -d: -f2)
    if curl -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
        success "Health check passed"
    else
        error "Health check failed"
    fi
    
    docker stop "$CONTAINER_ID" > /dev/null
    docker rm "$CONTAINER_ID" > /dev/null
}

validate_images() {
    log "Validating image metadata..."
    
    local image_name="ai-conversation:$IMAGE_TAG"
    if [ -n "$DOCKER_REGISTRY" ]; then
        image_name="$DOCKER_REGISTRY/$image_name"
    fi
    
    # Check image labels
    docker inspect "$image_name" --format '{{json .Config.Labels}}' | jq . || warn "No labels found"
    
    # Check image size
    IMAGE_SIZE=$(docker images "$image_name" --format "{{.Size}}")
    log "Image size: $IMAGE_SIZE"
    
    # Warn if image is too large
    SIZE_BYTES=$(docker inspect "$image_name" --format='{{.Size}}')
    if [ "$SIZE_BYTES" -gt 2000000000 ]; then  # 2GB
        warn "Image size is larger than 2GB: $(($SIZE_BYTES / 1024 / 1024))MB"
    fi
}

generate_sbom() {
    local image_name="${1:-ai-conversation:$IMAGE_TAG}"
    
    log "Generating SBOM (Software Bill of Materials) for $image_name..."
    
    if command -v syft &> /dev/null; then
        syft "$image_name" -o spdx-json > "$PROJECT_ROOT/sbom.json"
        success "SBOM generated: sbom.json"
    else
        warn "Syft not available, skipping SBOM generation"
    fi
}

create_image_manifest() {
    log "Creating image manifest..."
    
    cat > "$PROJECT_ROOT/image-manifest.json" << EOF
{
  "build_info": {
    "git_commit": "$GIT_COMMIT",
    "git_branch": "$GIT_BRANCH",
    "git_tag": "${GIT_TAG:-}",
    "build_date": "$BUILD_DATE",
    "image_tag": "$IMAGE_TAG",
    "build_platform": "$BUILD_PLATFORM",
    "build_type": "$BUILD_TYPE"
  },
  "images": [
    {
      "name": "ai-conversation",
      "tag": "$IMAGE_TAG",
      "registry": "$DOCKER_REGISTRY",
      "target": "$BUILD_TYPE"
    }
EOF
    
    if [ "$BUILD_GPU" = "true" ]; then
        cat >> "$PROJECT_ROOT/image-manifest.json" << EOF
    ,{
      "name": "ai-conversation-gpu",
      "tag": "$IMAGE_TAG",
      "registry": "$DOCKER_REGISTRY",
      "target": "gpu-production"
    }
EOF
    fi
    
    cat >> "$PROJECT_ROOT/image-manifest.json" << 'EOF'
  ]
}
EOF
    
    success "Image manifest created: image-manifest.json"
}

cleanup_builder() {
    log "Cleaning up build artifacts..."
    
    # Clean up dangling images
    docker image prune -f > /dev/null || true
    
    # Clean up build cache if requested
    if [ "${CLEAN_CACHE:-false}" = "true" ]; then
        docker buildx prune -f
    fi
}

# ==================== Main Build Functions ====================
build_all() {
    log "Building all images..."
    
    build_main_image
    
    if [ "$BUILD_GPU" = "true" ]; then
        build_gpu_image
    fi
    
    build_monitoring_images
    
    success "All images built successfully"
}

# ==================== CI/CD Integration ====================
setup_ci_environment() {
    if [ "${CI:-false}" = "true" ]; then
        log "Setting up CI environment..."
        
        # Set up Docker buildx for CI
        docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
        docker buildx create --name ci-builder --driver docker-container --use
        docker buildx inspect --bootstrap
        
        # Login to registry if credentials are available
        if [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_PASSWORD:-}" ]; then
            echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin "$DOCKER_REGISTRY"
            PUSH_IMAGES="true"
        fi
    fi
}

# ==================== Main Script ====================
# Parse command line arguments
TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--platform)
            BUILD_PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGES="true"
            shift
            ;;
        --gpu)
            BUILD_GPU="true"
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --cache-from)
            CACHE_FROM="true"
            shift
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        main|gpu|monitoring|all)
            TARGETS+=("$1")
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Default to main if no targets specified
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("main")
fi

# Main execution
log "Starting AI Conversation System build..."
log "Registry: ${DOCKER_REGISTRY:-local}"
log "Tag: $IMAGE_TAG"
log "Platform: $BUILD_PLATFORM"
log "Build Type: $BUILD_TYPE"
log "Push Images: $PUSH_IMAGES"
log "GPU Build: $BUILD_GPU"

check_dependencies
setup_ci_environment
get_build_info
prepare_build_args

# Execute build targets
for target in "${TARGETS[@]}"; do
    case $target in
        main)
            build_main_image
            ;;
        gpu)
            BUILD_GPU="true"
            build_gpu_image
            ;;
        monitoring)
            build_monitoring_images
            ;;
        all)
            build_all
            ;;
        *)
            error "Unknown target: $target"
            ;;
    esac
done

# Post-build actions
if [ "$PUSH_IMAGES" = "false" ]; then
    test_images
    validate_images
    run_security_scan
    generate_sbom
fi

create_image_manifest
cleanup_builder

success "Build completed successfully!"

if [ "$PUSH_IMAGES" = "true" ]; then
    log "Images pushed to registry: $DOCKER_REGISTRY"
else
    log "Images built locally. Use --push to push to registry."
fi