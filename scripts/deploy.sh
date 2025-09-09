#!/bin/bash

# AI Conversation System - Production Deployment Script
# Complete deployment automation with pre-flight checks and rollback capability

set -euo pipefail

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_ROOT/k8s"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-ai-conversation}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-ai-conversation-monitoring}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GPU_ENABLED="${GPU_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"

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

check_dependencies() {
    log "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v kubectl &> /dev/null; then
        missing_deps+=("kubectl")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v helm &> /dev/null; then
        missing_deps+=("helm")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    success "All dependencies checked"
}

show_help() {
    cat << EOF
AI Conversation System - Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
  deploy          Deploy the complete system
  upgrade         Upgrade existing deployment
  rollback        Rollback to previous version
  destroy         Remove all resources
  status          Show deployment status
  logs            Show application logs

Options:
  -e, --environment ENV     Target environment (default: production)
  -n, --namespace NS        Kubernetes namespace (default: ai-conversation)
  -t, --image-tag TAG       Docker image tag (default: latest)
  -g, --gpu                 Enable GPU workers
  -d, --dry-run            Show what would be done without executing
  -s, --skip-preflight     Skip pre-flight checks
  -h, --help               Show this help message

Environment Variables:
  KUBECONFIG              Path to kubeconfig file
  DOCKER_REGISTRY         Docker registry URL
  ENVIRONMENT            Target environment
  IMAGE_TAG              Docker image tag
  GPU_ENABLED            Enable GPU support (true/false)

Examples:
  $0 deploy
  $0 deploy --gpu --image-tag v1.2.3
  $0 upgrade --namespace staging
  $0 status
  $0 rollback
  $0 destroy --dry-run

EOF
}

preflight_checks() {
    if [ "$SKIP_PREFLIGHT" = "true" ]; then
        warn "Skipping pre-flight checks"
        return 0
    fi
    
    log "Running pre-flight checks..."
    
    # Check Kubernetes cluster resources
    log "Checking cluster resources..."
    kubectl top nodes || warn "Cannot get node metrics"
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace $NAMESPACE does not exist, will be created"
    fi
    
    if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
        warn "Monitoring namespace $MONITORING_NAMESPACE does not exist, will be created"
    fi
    
    # Check storage classes
    log "Checking storage classes..."
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        warn "Storage class 'fast-ssd' not found, using default"
    fi
    
    # Validate Kubernetes manifests
    log "Validating Kubernetes manifests..."
    for manifest in "$K8S_DIR"/*.yaml; do
        if [ -f "$manifest" ]; then
            kubectl apply --dry-run=client -f "$manifest" > /dev/null || error "Invalid manifest: $manifest"
        fi
    done
    
    # Check Docker images
    log "Checking Docker image availability..."
    if [ "$DRY_RUN" = "false" ]; then
        if ! docker pull "ai-conversation:$IMAGE_TAG" 2>/dev/null; then
            warn "Cannot pull image ai-conversation:$IMAGE_TAG"
        fi
    fi
    
    success "Pre-flight checks completed"
}

create_namespaces() {
    log "Creating namespaces..."
    
    kubectl apply -f "$K8S_DIR/namespace.yaml" ${DRY_RUN:+--dry-run=client}
    
    # Label namespaces for network policies
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite ${DRY_RUN:+--dry-run=client}
    kubectl label namespace "$MONITORING_NAMESPACE" name="$MONITORING_NAMESPACE" --overwrite ${DRY_RUN:+--dry-run=client}
}

deploy_secrets() {
    log "Deploying secrets..."
    
    if [ -f "$PROJECT_ROOT/.env.production" ]; then
        log "Found production environment file"
        # Create secrets from environment file
        kubectl create secret generic ai-conversation-secrets \
            --namespace="$NAMESPACE" \
            --from-env-file="$PROJECT_ROOT/.env.production" \
            --dry-run=client -o yaml | kubectl apply -f - ${DRY_RUN:+--dry-run=client}
    else
        warn "No production environment file found, using template secrets"
        kubectl apply -f "$K8S_DIR/secrets.yaml" ${DRY_RUN:+--dry-run=client}
    fi
}

deploy_configmaps() {
    log "Deploying configuration..."
    
    kubectl apply -f "$K8S_DIR/configmap-production.yaml" ${DRY_RUN:+--dry-run=client}
}

deploy_storage() {
    log "Deploying storage resources..."
    
    kubectl apply -f "$K8S_DIR/pv-pvc.yaml" ${DRY_RUN:+--dry-run=client}
    
    # Wait for PVCs to be bound
    if [ "$DRY_RUN" = "false" ]; then
        log "Waiting for storage to be ready..."
        kubectl wait --for=condition=Bound pvc/models-pvc --namespace="$NAMESPACE" --timeout=300s
    fi
}

deploy_databases() {
    log "Deploying database services..."
    
    # Deploy PostgreSQL
    kubectl apply -f "$K8S_DIR/statefulset.yaml" ${DRY_RUN:+--dry-run=client}
    kubectl apply -f "$K8S_DIR/service-production.yaml" ${DRY_RUN:+--dry-run=client}
    
    if [ "$DRY_RUN" = "false" ]; then
        log "Waiting for PostgreSQL to be ready..."
        kubectl wait --for=condition=Ready pod/postgres-master-0 --namespace="$NAMESPACE" --timeout=600s
        
        log "Waiting for Redis cluster to be ready..."
        kubectl wait --for=condition=Ready pod -l app=ai-conversation-system,component=redis --namespace="$NAMESPACE" --timeout=300s
        
        # Initialize Redis cluster
        initialize_redis_cluster
    fi
}

initialize_redis_cluster() {
    log "Initializing Redis cluster..."
    
    # Get Redis pod IPs
    REDIS_IPS=$(kubectl get pods -l app=ai-conversation-system,component=redis \
        --namespace="$NAMESPACE" -o jsonpath='{.items[*].status.podIP}' | tr ' ' '\n' | sort | tr '\n' ':6379 ')
    REDIS_IPS="${REDIS_IPS% }"  # Remove trailing space
    
    # Create cluster
    kubectl exec redis-0 --namespace="$NAMESPACE" -- redis-cli --cluster create $REDIS_IPS --cluster-replicas 1 --cluster-yes
}

deploy_application() {
    log "Deploying application services..."
    
    # Update image tags in deployment manifests
    if [ "$IMAGE_TAG" != "latest" ]; then
        sed -i.bak "s|ai-conversation:latest|ai-conversation:$IMAGE_TAG|g" "$K8S_DIR/deployment-production.yaml"
        if [ "$GPU_ENABLED" = "true" ]; then
            sed -i.bak "s|ai-conversation-gpu:latest|ai-conversation-gpu:$IMAGE_TAG|g" "$K8S_DIR/deployment-production.yaml"
        fi
    fi
    
    # Deploy main application
    kubectl apply -f "$K8S_DIR/deployment-production.yaml" ${DRY_RUN:+--dry-run=client}
    
    # Deploy GPU workers if enabled
    if [ "$GPU_ENABLED" = "true" ]; then
        log "Deploying GPU-enabled workers..."
        kubectl apply -f <(grep -A 200 "ai-conversation-worker-ml-gpu" "$K8S_DIR/deployment-production.yaml") ${DRY_RUN:+--dry-run=client}
    fi
    
    if [ "$DRY_RUN" = "false" ]; then
        log "Waiting for application to be ready..."
        kubectl wait --for=condition=Available deployment/ai-conversation-app --namespace="$NAMESPACE" --timeout=600s
        kubectl wait --for=condition=Available deployment/ai-conversation-worker-general --namespace="$NAMESPACE" --timeout=300s
    fi
    
    # Restore original files
    if [ -f "$K8S_DIR/deployment-production.yaml.bak" ]; then
        mv "$K8S_DIR/deployment-production.yaml.bak" "$K8S_DIR/deployment-production.yaml"
    fi
}

deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $MONITORING_NAMESPACE
  labels:
    app: prometheus
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
      component: monitoring
  template:
    metadata:
      labels:
        app: prometheus
        component: monitoring
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.47.0
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --storage.tsdb.retention.time=30d
        - --web.console.libraries=/usr/share/prometheus/console_libraries
        - --web.console.templates=/usr/share/prometheus/consoles
        - --web.enable-lifecycle
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-pvc
EOF
    
    # Create Prometheus configuration
    kubectl create configmap prometheus-config \
        --namespace="$MONITORING_NAMESPACE" \
        --from-file="$MONITORING_DIR/prometheus/prometheus.yml" \
        --from-file="$MONITORING_DIR/prometheus/rules/" \
        --dry-run=client -o yaml | kubectl apply -f - ${DRY_RUN:+--dry-run=client}
    
    if [ "$DRY_RUN" = "false" ]; then
        log "Waiting for monitoring to be ready..."
        kubectl wait --for=condition=Available deployment/prometheus --namespace="$MONITORING_NAMESPACE" --timeout=300s
    fi
}

run_database_migrations() {
    log "Running database migrations..."
    
    if [ "$DRY_RUN" = "false" ]; then
        kubectl exec deployment/ai-conversation-app --namespace="$NAMESPACE" -- alembic upgrade head
    fi
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check pod status
    kubectl get pods --namespace="$NAMESPACE"
    
    # Check service endpoints
    kubectl get services --namespace="$NAMESPACE"
    
    # Run health checks
    if [ "$DRY_RUN" = "false" ]; then
        log "Running health checks..."
        
        # Wait for health endpoint to be ready
        kubectl wait --for=condition=Ready pod -l app=ai-conversation-system,component=web --namespace="$NAMESPACE" --timeout=300s
        
        # Test health endpoint
        APP_POD=$(kubectl get pods -l app=ai-conversation-system,component=web --namespace="$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
        kubectl exec "$APP_POD" --namespace="$NAMESPACE" -- curl -f http://localhost:8000/health || error "Health check failed"
        
        success "Health checks passed"
    fi
}

deploy_full() {
    log "Starting full deployment of AI Conversation System"
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    log "Image Tag: $IMAGE_TAG"
    log "GPU Enabled: $GPU_ENABLED"
    log "Dry Run: $DRY_RUN"
    
    preflight_checks
    create_namespaces
    deploy_secrets
    deploy_configmaps
    deploy_storage
    deploy_databases
    deploy_application
    deploy_monitoring
    run_database_migrations
    verify_deployment
    
    success "Deployment completed successfully!"
    
    if [ "$DRY_RUN" = "false" ]; then
        log "Access URLs:"
        EXTERNAL_IP=$(kubectl get service ai-conversation-app --namespace="$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$EXTERNAL_IP" ]; then
            log "Application: http://$EXTERNAL_IP"
        else
            log "Application: Use 'kubectl port-forward service/ai-conversation-app 8000:80 -n $NAMESPACE'"
        fi
        
        GRAFANA_IP=$(kubectl get service grafana-service --namespace="$MONITORING_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$GRAFANA_IP" ]; then
            log "Grafana: http://$GRAFANA_IP"
        else
            log "Grafana: Use 'kubectl port-forward service/grafana-service 3000:80 -n $MONITORING_NAMESPACE'"
        fi
    fi
}

show_status() {
    log "AI Conversation System Status"
    echo
    
    log "Application Pods:"
    kubectl get pods --namespace="$NAMESPACE" -l app=ai-conversation-system
    echo
    
    log "Services:"
    kubectl get services --namespace="$NAMESPACE"
    echo
    
    log "Storage:"
    kubectl get pvc --namespace="$NAMESPACE"
    echo
    
    log "Recent Events:"
    kubectl get events --namespace="$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

show_logs() {
    local component="${1:-app}"
    
    case $component in
        app|web)
            kubectl logs -l app=ai-conversation-system,component=web --namespace="$NAMESPACE" --tail=100 -f
            ;;
        worker)
            kubectl logs -l app=ai-conversation-system,component=worker-general --namespace="$NAMESPACE" --tail=100 -f
            ;;
        ml)
            kubectl logs -l app=ai-conversation-system,component=worker-ml --namespace="$NAMESPACE" --tail=100 -f
            ;;
        db|database)
            kubectl logs -l app=ai-conversation-system,component=postgres-master --namespace="$NAMESPACE" --tail=100 -f
            ;;
        redis)
            kubectl logs -l app=ai-conversation-system,component=redis --namespace="$NAMESPACE" --tail=100 -f
            ;;
        *)
            kubectl logs -l app=ai-conversation-system --namespace="$NAMESPACE" --tail=100 -f
            ;;
    esac
}

rollback_deployment() {
    log "Rolling back deployment..."
    
    # Rollback application deployment
    kubectl rollout undo deployment/ai-conversation-app --namespace="$NAMESPACE"
    kubectl rollout undo deployment/ai-conversation-worker-general --namespace="$NAMESPACE"
    kubectl rollout undo deployment/ai-conversation-worker-ml --namespace="$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/ai-conversation-app --namespace="$NAMESPACE"
    kubectl rollout status deployment/ai-conversation-worker-general --namespace="$NAMESPACE"
    kubectl rollout status deployment/ai-conversation-worker-ml --namespace="$NAMESPACE"
    
    success "Rollback completed"
}

destroy_deployment() {
    warn "This will destroy the entire AI Conversation System deployment!"
    
    if [ "$DRY_RUN" = "false" ]; then
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Deployment destruction cancelled"
            exit 0
        fi
    fi
    
    log "Destroying deployment..."
    
    # Delete application resources
    kubectl delete -f "$K8S_DIR/deployment-production.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    kubectl delete -f "$K8S_DIR/service-production.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    kubectl delete -f "$K8S_DIR/statefulset.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    kubectl delete -f "$K8S_DIR/pv-pvc.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    kubectl delete -f "$K8S_DIR/configmap-production.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    kubectl delete -f "$K8S_DIR/secrets.yaml" --namespace="$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    
    # Delete monitoring
    kubectl delete namespace "$MONITORING_NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    
    # Delete main namespace
    kubectl delete namespace "$NAMESPACE" ${DRY_RUN:+--dry-run=client} || true
    
    success "Deployment destroyed"
}

# ==================== Main Script ====================
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ENABLED="true"
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -s|--skip-preflight)
            SKIP_PREFLIGHT="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy)
            COMMAND="deploy"
            shift
            ;;
        upgrade)
            COMMAND="upgrade"
            shift
            ;;
        rollback)
            COMMAND="rollback"
            shift
            ;;
        destroy)
            COMMAND="destroy"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        logs)
            COMMAND="logs"
            LOG_COMPONENT="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Check dependencies
check_dependencies

# Execute command
case ${COMMAND:-deploy} in
    deploy)
        deploy_full
        ;;
    upgrade)
        deploy_application
        verify_deployment
        ;;
    rollback)
        rollback_deployment
        ;;
    destroy)
        destroy_deployment
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${LOG_COMPONENT:-app}"
        ;;
    *)
        error "Unknown command: ${COMMAND:-deploy}"
        ;;
esac