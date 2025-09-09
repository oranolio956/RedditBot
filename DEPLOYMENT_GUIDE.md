# AI Conversation System - Production Deployment Guide

This comprehensive guide covers deploying the AI Conversation System to production using Docker containers and Kubernetes, with support for 1000+ concurrent users, auto-scaling, monitoring, and high availability.

## 🏗️ Architecture Overview

The system consists of:
- **Web Applications**: Load-balanced FastAPI instances
- **Worker Processes**: Celery workers for background tasks
- **ML Workers**: Specialized workers for AI/ML processing (CPU/GPU)
- **Databases**: PostgreSQL with master-slave replication
- **Cache Layer**: Redis cluster for high performance
- **Monitoring**: Prometheus, Grafana, AlertManager
- **Load Balancer**: Nginx with SSL termination

## 📋 Prerequisites

### Required Tools
```bash
# Core tools
kubectl >= 1.28.0
docker >= 24.0.0
helm >= 3.12.0
git >= 2.40.0

# Optional but recommended
k9s          # Kubernetes dashboard
lens         # Kubernetes IDE
trivy        # Security scanning
```

### Infrastructure Requirements

#### Minimum Production Cluster
- **Nodes**: 6 nodes (3 compute, 2 database, 1 monitoring)
- **CPU**: 24 vCPUs total (4 per compute node)
- **Memory**: 96GB total (16GB per compute node)
- **Storage**: 1TB SSD total
- **Network**: 10Gbps with low latency

#### Recommended Production Cluster
- **Nodes**: 12 nodes (6 compute, 3 database, 2 cache, 1 monitoring)
- **CPU**: 72 vCPUs total (6 per compute node)
- **Memory**: 288GB total (24GB per compute node)
- **Storage**: 5TB SSD with 10,000 IOPS
- **Network**: 25Gbps with dedicated monitoring network

#### Node Specifications
```yaml
Compute Nodes:
  - 6 vCPUs, 24GB RAM
  - 500GB SSD storage
  - Labels: node-type=compute

Database Nodes:
  - 8 vCPUs, 32GB RAM
  - 1TB SSD storage, 10,000 IOPS
  - Labels: node-type=database

ML Nodes (Optional):
  - 8 vCPUs, 48GB RAM
  - 2x NVIDIA T4/V100 GPUs
  - 500GB SSD storage
  - Labels: node-type=ml, accelerator=nvidia-tesla-*

Cache Nodes:
  - 4 vCPUs, 16GB RAM
  - 200GB SSD storage
  - Labels: node-type=cache

Monitoring Node:
  - 4 vCPUs, 16GB RAM
  - 500GB SSD storage
  - Labels: node-type=monitoring
```

## 🚀 Quick Start

### 1. Clone and Configure
```bash
git clone <repository-url>
cd ai-conversation-system

# Copy environment template
cp .env.example .env.production

# Configure production settings
nano .env.production
```

### 2. Build Images
```bash
# Build for multi-platform
./scripts/build.sh all --platform linux/amd64,linux/arm64 --tag v1.0.0

# Build with GPU support
./scripts/build.sh all --gpu --tag v1.0.0

# Push to registry
./scripts/build.sh all --push --registry your-registry.com
```

### 3. Deploy to Kubernetes
```bash
# Full deployment
./scripts/deploy.sh deploy --environment production --image-tag v1.0.0

# With GPU workers
./scripts/deploy.sh deploy --gpu --environment production

# Check status
./scripts/deploy.sh status
```

## 📁 Project Structure

```
ai-conversation-system/
├── app/                          # Application code
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml           # Namespaces
│   ├── secrets.yaml             # Secret templates
│   ├── configmap-production.yaml # Production config
│   ├── deployment-production.yaml # Main deployments
│   ├── statefulset.yaml         # Databases
│   ├── service-production.yaml  # Services
│   └── pv-pvc.yaml             # Storage
├── scripts/                      # Deployment scripts
│   ├── deploy.sh                # Main deployment script
│   └── build.sh                 # Image build script
├── monitoring/                   # Monitoring configs
│   └── prometheus/              # Prometheus setup
├── docker-compose.production.yml # Production compose
├── docker-compose.gpu.yml       # GPU-enabled compose
├── Dockerfile                   # Main application
└── Dockerfile.gpu              # GPU-enabled build
```

## 🔧 Configuration

### Environment Variables

#### Database Configuration
```bash
DB_HOST=postgres-master-service
DB_PORT=5432
DB_NAME=ai_conversation
DB_USER=ai_conversation_user
DB_PASSWORD=<secure-password-32-chars>
DB_POOL_SIZE=30
DB_MAX_OVERFLOW=50
DB_REPLICATION_USER=replicator
DB_REPLICATION_PASSWORD=<replication-password>
```

#### Redis Configuration
```bash
REDIS_CLUSTER_ENABLED=true
REDIS_CLUSTER_NODES=redis-0.redis-headless:6379,redis-1.redis-headless:6379,redis-2.redis-headless:6379
REDIS_PASSWORD=<secure-redis-password>
REDIS_MAX_CONNECTIONS=100
```

#### Application Security
```bash
SECRET_KEY=<64-character-secret-key>
JWT_SECRET_KEY=<64-character-jwt-secret>
ENCRYPTION_KEY=<32-character-encryption-key>
```

#### Telegram Configuration
```bash
TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_WEBHOOK_URL=https://your-domain.com/webhook
TELEGRAM_WEBHOOK_SECRET=<webhook-secret>
```

#### Monitoring
```bash
SENTRY_DSN=<your-sentry-dsn>
SENTRY_ENVIRONMENT=production
METRICS_ENABLED=true
PROMETHEUS_URL=http://prometheus:9090
```

### Resource Limits and Scaling

#### Application Pods
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# Auto-scaling
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

#### Worker Pods
```yaml
# General workers
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# ML workers
resources:
  requests:
    memory: "3Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
  limits:
    memory: "6Gi"
    cpu: "3000m"
    nvidia.com/gpu: 1
```

## 🐳 Docker Deployment

### Production Docker Compose
```bash
# Start with production compose
docker-compose -f docker-compose.production.yml up -d

# Scale workers
docker-compose -f docker-compose.production.yml up -d --scale worker-general=4

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f app
```

### GPU-Enabled Deployment
```bash
# Prerequisites: NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Deploy GPU-enabled stack
docker-compose -f docker-compose.gpu.yml up -d

# Check GPU usage
docker exec -it ai-conversation-worker-ml-gpu-1 nvidia-smi
```

## ☸️ Kubernetes Deployment

### Namespace Setup
```bash
# Create namespaces
kubectl apply -f k8s/namespace.yaml

# Verify
kubectl get namespaces
```

### Secret Management
```bash
# Create secrets from .env file
kubectl create secret generic ai-conversation-secrets \
  --namespace=ai-conversation \
  --from-env-file=.env.production

# Or use the template
kubectl apply -f k8s/secrets.yaml

# Verify
kubectl get secrets -n ai-conversation
```

### Storage Configuration
```bash
# Deploy storage classes and PVCs
kubectl apply -f k8s/pv-pvc.yaml

# Check storage
kubectl get pvc -n ai-conversation
kubectl get storageclass
```

### Database Deployment
```bash
# Deploy PostgreSQL with replication
kubectl apply -f k8s/statefulset.yaml

# Wait for database to be ready
kubectl wait --for=condition=Ready pod/postgres-master-0 -n ai-conversation --timeout=600s

# Check database status
kubectl exec postgres-master-0 -n ai-conversation -- pg_isready
```

### Redis Cluster Setup
```bash
# Redis cluster is deployed with statefulset.yaml
kubectl get pods -l component=redis -n ai-conversation

# Initialize cluster (done automatically by init container)
kubectl logs redis-0 -n ai-conversation -c redis-init
```

### Application Deployment
```bash
# Deploy main application
kubectl apply -f k8s/deployment-production.yaml

# Wait for deployment
kubectl wait --for=condition=Available deployment/ai-conversation-app -n ai-conversation --timeout=600s

# Check pods
kubectl get pods -l app=ai-conversation-system -n ai-conversation
```

### Service and Ingress
```bash
# Deploy services
kubectl apply -f k8s/service-production.yaml

# Check services
kubectl get services -n ai-conversation

# Get external IP
kubectl get service ai-conversation-app -n ai-conversation
```

## 📊 Monitoring Setup

### Prometheus and Grafana
```bash
# Deploy monitoring namespace
kubectl apply -f k8s/namespace.yaml

# Deploy Prometheus
kubectl apply -f monitoring/prometheus/

# Deploy Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install grafana grafana/grafana \
  --namespace ai-conversation-monitoring \
  --set adminPassword=secure-admin-password \
  --set persistence.enabled=true \
  --set persistence.size=10Gi

# Get Grafana password
kubectl get secret grafana -n ai-conversation-monitoring -o jsonpath="{.data.admin-password}" | base64 --decode
```

### Access Monitoring
```bash
# Port forward Prometheus
kubectl port-forward service/prometheus-service 9090:9090 -n ai-conversation-monitoring

# Port forward Grafana
kubectl port-forward service/grafana 3000:80 -n ai-conversation-monitoring

# Access:
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/password from above)
```

## 🔍 Health Checks and Testing

### Application Health
```bash
# Check application health
kubectl exec deployment/ai-conversation-app -n ai-conversation -- curl -f http://localhost:8000/health

# Check database connection
kubectl exec deployment/ai-conversation-app -n ai-conversation -- python -c "from app.database import engine; print('DB OK')"

# Check Redis connection
kubectl exec deployment/ai-conversation-app -n ai-conversation -- python -c "from app.core.redis import redis_client; print(redis_client.ping())"
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load test
cd tests/load
locust -f locustfile.py --host https://your-domain.com
```

### Performance Monitoring
```bash
# Check resource usage
kubectl top pods -n ai-conversation
kubectl top nodes

# Check horizontal pod autoscaler
kubectl get hpa -n ai-conversation

# Check metrics
curl http://localhost:8001/metrics  # Port-forward metrics port
```

## 🔐 Security Configuration

### TLS/SSL Setup
```bash
# Create TLS secret
kubectl create secret tls ai-conversation-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=ai-conversation

# Update ingress or load balancer configuration
```

### Network Policies
```bash
# Apply network policies for security
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-conversation-netpol
  namespace: ai-conversation
spec:
  podSelector:
    matchLabels:
      app: ai-conversation-system
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ai-conversation
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: ai-conversation
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
EOF
```

### Pod Security Standards
```bash
# Apply pod security policy
kubectl label namespace ai-conversation pod-security.kubernetes.io/enforce=restricted
kubectl label namespace ai-conversation pod-security.kubernetes.io/audit=restricted
kubectl label namespace ai-conversation pod-security.kubernetes.io/warn=restricted
```

## 🚨 Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod events
kubectl describe pod <pod-name> -n ai-conversation

# Check logs
kubectl logs <pod-name> -n ai-conversation -f

# Check resource constraints
kubectl describe nodes
kubectl get resourcequotas -n ai-conversation
```

#### Database Connection Issues
```bash
# Check database pod
kubectl describe pod postgres-master-0 -n ai-conversation

# Check database logs
kubectl logs postgres-master-0 -n ai-conversation

# Test connection
kubectl exec postgres-master-0 -n ai-conversation -- psql -U ai_conversation_user -d ai_conversation -c "SELECT 1"
```

#### Redis Cluster Issues
```bash
# Check cluster status
kubectl exec redis-0 -n ai-conversation -- redis-cli cluster info

# Check cluster nodes
kubectl exec redis-0 -n ai-conversation -- redis-cli cluster nodes

# Reset cluster if needed
kubectl delete statefulset redis -n ai-conversation
kubectl delete pvc -l component=redis -n ai-conversation
kubectl apply -f k8s/statefulset.yaml
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n ai-conversation
kubectl top nodes

# Check HPA status
kubectl describe hpa -n ai-conversation

# Check metrics
kubectl port-forward service/ai-conversation-app 8001:8001 -n ai-conversation
curl http://localhost:8001/metrics | grep -E "(response_time|requests_total|memory|cpu)"
```

### Log Aggregation
```bash
# Centralized logging with ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n ai-conversation-monitoring
helm install kibana elastic/kibana -n ai-conversation-monitoring
helm install filebeat elastic/filebeat -n ai-conversation-monitoring
```

## 🔄 Backup and Recovery

### Database Backups
```bash
# Manual backup
kubectl exec postgres-master-0 -n ai-conversation -- pg_dump -U ai_conversation_user ai_conversation > backup.sql

# Automated backup with CronJob
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: ai-conversation
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - sh
            - -c
            - |
              pg_dump -h postgres-master-service -U \$DB_USER \$DB_NAME | gzip > /backup/backup-\$(date +%Y%m%d-%H%M%S).sql.gz
              # Upload to S3 or other storage
            env:
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: ai-conversation-secrets
                  key: DB_USER
            - name: DB_NAME
              value: ai_conversation
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: ai-conversation-secrets
                  key: DB_PASSWORD
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```

### Disaster Recovery
```bash
# Create snapshot of persistent volumes
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: postgres-snapshot-$(date +%Y%m%d)
  namespace: ai-conversation
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: postgres-master-storage-postgres-master-0
EOF

# Restore from snapshot
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-restored
  namespace: ai-conversation
spec:
  storageClassName: fast-ssd
  dataSource:
    name: postgres-snapshot-20231201
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF
```

## 🔧 Maintenance

### Updates and Rollbacks
```bash
# Update deployment
kubectl set image deployment/ai-conversation-app ai-conversation=ai-conversation:v1.1.0 -n ai-conversation

# Check rollout status
kubectl rollout status deployment/ai-conversation-app -n ai-conversation

# Rollback if needed
kubectl rollout undo deployment/ai-conversation-app -n ai-conversation

# Scale deployment
kubectl scale deployment ai-conversation-app --replicas=5 -n ai-conversation
```

### Certificate Management
```bash
# Update TLS certificates
kubectl create secret tls ai-conversation-tls-new \
  --cert=new-tls.crt \
  --key=new-tls.key \
  --namespace=ai-conversation

# Update ingress to use new certificate
kubectl patch ingress ai-conversation-ingress -n ai-conversation -p '{"spec":{"tls":[{"secretName":"ai-conversation-tls-new","hosts":["your-domain.com"]}]}}'
```

## 📈 Scaling Guidelines

### Horizontal Scaling
```bash
# Scale based on metrics
CPU > 70%: Add 1 replica every 60 seconds
Memory > 80%: Add 1 replica every 60 seconds
Requests/sec > 1000: Add 2 replicas

# Manual scaling
kubectl scale deployment ai-conversation-app --replicas=8 -n ai-conversation
```

### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment ai-conversation-app -n ai-conversation -p '{"spec":{"template":{"spec":{"containers":[{"name":"ai-conversation","resources":{"requests":{"memory":"4Gi","cpu":"2000m"},"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'
```

### Cluster Scaling
```bash
# Add nodes for scaling
# 1000 concurrent users ≈ 6-8 compute nodes
# 5000 concurrent users ≈ 12-16 compute nodes
# 10000 concurrent users ≈ 20-24 compute nodes

# Monitor metrics
kubectl get --raw /metrics | grep node_cpu
kubectl get --raw /metrics | grep node_memory
```

## 🎯 Production Checklist

### Pre-Deployment
- [ ] Infrastructure provisioned and configured
- [ ] DNS records configured
- [ ] SSL certificates obtained
- [ ] Secrets and configurations prepared
- [ ] Images built and pushed to registry
- [ ] Kubernetes manifests validated
- [ ] Monitoring stack configured

### Deployment
- [ ] Deploy in correct order (storage → databases → application)
- [ ] Verify all pods are running and healthy
- [ ] Check service endpoints and load balancers
- [ ] Run smoke tests
- [ ] Verify monitoring and alerting
- [ ] Test backup and recovery procedures

### Post-Deployment
- [ ] Monitor application metrics
- [ ] Verify auto-scaling behavior
- [ ] Test failure scenarios
- [ ] Document runbooks
- [ ] Set up on-call procedures
- [ ] Schedule regular maintenance

## 📞 Support

### Monitoring Dashboards
- **Application**: Grafana dashboard for request rates, response times, error rates
- **Infrastructure**: Node metrics, pod metrics, resource utilization
- **Business**: User interactions, conversation metrics, typing simulations

### Alerting
- **Critical**: Service down, database unreachable, high error rate
- **Warning**: High latency, memory usage, disk space low
- **Info**: Deployment events, scaling events

### Runbooks
- Service recovery procedures
- Database failover steps
- Scaling procedures
- Certificate renewal
- Backup verification

This deployment guide provides comprehensive coverage for deploying the AI Conversation System at scale. Follow the sections relevant to your deployment method and infrastructure setup.