# Telegram Bot Foundation - Production Ready Implementation

A comprehensive, production-ready Telegram bot implementation with advanced anti-ban measures, circuit breakers, real-time monitoring, and enterprise-grade scalability.

## 🚀 Features

### Core Bot Features
- **aiogram 3.x** - High-performance async Telegram bot framework
- **Advanced Rate Limiting** - Redis-based sliding window rate limiting with burst handling
- **Natural Typing Simulation** - Human-like typing patterns with realistic delays
- **Circuit Breaker Pattern** - Automatic failure detection and recovery
- **Comprehensive Error Handling** - Graceful error recovery with user-friendly messages
- **Session Management** - Redis-backed user session tracking with conversation context
- **Webhook Support** - Production-ready webhook handling with security validation
- **Health Monitoring** - Real-time health checks and system metrics
- **Anti-Detection System** - Advanced behavioral analysis and pattern randomization

### Anti-Ban Measures
- **Behavioral Pattern Analysis** - Risk assessment and user profiling
- **Natural Timing Variations** - Realistic response delays with jitter
- **Pattern Randomization** - Daily seed-based behavior variation
- **Rate Limit Intelligence** - Smart request throttling
- **Risk Mitigation** - Automatic intervention for suspicious activity
- **Detection Evasion** - Anti-fingerprinting measures

### Monitoring & Analytics
- **Prometheus Metrics** - Comprehensive performance and usage metrics
- **Real-time Dashboard** - System health and performance monitoring
- **Structured Logging** - JSON-formatted logs with correlation IDs
- **Performance Tracking** - Response times, throughput, and error rates
- **User Analytics** - Behavior patterns and interaction analysis

### Production Features
- **High Concurrency** - Handle 1000+ concurrent conversations
- **Horizontal Scaling** - Multi-instance deployment support
- **Zero-Downtime Deployment** - Rolling updates and health checks
- **Security Hardened** - IP filtering, signature verification, HTTPS
- **Database Integration** - PostgreSQL with connection pooling
- **Caching Layer** - Multi-tier Redis caching with LRU policies

## 📋 Requirements

### System Requirements
- Python 3.11+
- Redis 7.0+
- PostgreSQL 15+
- Docker & Docker Compose (for production)
- 4GB+ RAM (recommended for production)
- 2+ CPU cores

### Dependencies
All dependencies are specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🛠 Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
cd Reddit - bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Required environment variables:
- `TELEGRAM_BOT_TOKEN` - From @BotFather
- `TELEGRAM_WEBHOOK_URL` - Your webhook URL (HTTPS required)
- `DB_PASSWORD` - PostgreSQL password
- `REDIS_PASSWORD` - Redis password  
- `SECRET_KEY` - Application secret key
- `JWT_SECRET_KEY` - JWT signing key

### 3. Database Setup
```bash
# Run database migrations
alembic upgrade head

# Verify database connection
python -c "from app.database.connection import db_manager; import asyncio; asyncio.run(db_manager.health_check())"
```

### 4. Redis Setup
```bash
# Start Redis (if not using Docker)
redis-server

# Verify Redis connection
python -c "from app.core.redis import redis_manager; import asyncio; asyncio.run(redis_manager.health_check())"
```

## 🚀 Usage

### Development Mode

#### Using the CLI
```bash
# Start bot in polling mode
python -m app.telegram.cli start

# Check bot status
python -m app.telegram.cli status

# View metrics
python -m app.telegram.cli metrics

# Manage webhooks
python -m app.telegram.cli webhook info
python -m app.telegram.cli webhook test

# View rate limits
python -m app.telegram.cli rate-limits --user-id 123456

# Manage sessions
python -m app.telegram.cli sessions

# Run cleanup
python -m app.telegram.cli cleanup
```

#### Using the FastAPI Server
```bash
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health

# Bot metrics
curl http://localhost:8000/metrics
```

### Production Deployment

#### Using Docker Compose
```bash
# Deploy to production
python deploy_telegram_bot.py deploy

# Check deployment health
python deploy_telegram_bot.py health

# Scale services
python deploy_telegram_bot.py scale --replicas 5
```

#### Manual Docker Deployment
```bash
# Build images
docker build -t telegram-bot:latest .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale services
docker-compose -f docker-compose.prod.yml scale telegram-bot=3
```

## 📊 API Endpoints

### Bot Management
- `GET /api/v1/telegram/status` - Bot status and health
- `GET /api/v1/telegram/metrics` - Performance metrics
- `GET /api/v1/telegram/metrics/historical` - Historical data

### Webhook Management
- `GET /api/v1/telegram/webhook/info` - Webhook information
- `POST /api/v1/telegram/webhook/test` - Test webhook connectivity
- `POST /api/v1/telegram/webhook/restart` - Restart webhook

### Message Handling
- `POST /api/v1/telegram/send-message` - Send message through bot

### Session Management
- `GET /api/v1/telegram/sessions` - Active sessions overview
- `GET /api/v1/telegram/sessions/{user_id}` - User sessions
- `DELETE /api/v1/telegram/sessions/{session_id}` - Expire session

### Rate Limiting
- `GET /api/v1/telegram/rate-limits/{user_id}` - User rate limit status
- `POST /api/v1/telegram/rate-limits/reset` - Reset rate limits

### Circuit Breakers
- `GET /api/v1/telegram/circuit-breakers` - Circuit breaker status
- `POST /api/v1/telegram/circuit-breakers/{name}/reset` - Reset breaker

### Anti-Ban System
- `GET /api/v1/telegram/anti-ban/metrics` - Anti-detection metrics

### Maintenance
- `POST /api/v1/telegram/maintenance/cleanup` - Run cleanup tasks

## 🔧 Architecture

### Component Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Telegram Bot   │────│   Message       │
│   (REST API)    │    │   (aiogram)     │    │   Handlers      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Anti-Ban       │────│   Middleware    │────│  Rate Limiter   │
    │  Manager        │    │   Pipeline      │    │  (Redis)        │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Session        │────│  Circuit        │────│   Metrics       │
    │  Manager        │    │  Breakers       │    │  (Prometheus)   │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Middleware Pipeline
1. **Circuit Breaker** - Failure detection and recovery
2. **Logging** - Request/response tracking
3. **Metrics** - Performance data collection
4. **Authentication** - User verification
5. **Rate Limiting** - Request throttling
6. **Anti-Ban** - Behavioral analysis
7. **Session Management** - User context tracking

### Anti-Ban System
- **Risk Assessment** - Multi-factor analysis (timing, frequency, behavior)
- **Pattern Randomization** - Daily seed-based variation
- **Mitigation Strategies** - Delays, pattern changes, request blocking
- **Learning System** - Adaptive behavior based on effectiveness

### Rate Limiting
- **Sliding Window** - Redis-based precise rate limiting
- **Multiple Rules** - Per-user, per-chat, per-command limits
- **Burst Handling** - Allow temporary traffic spikes
- **Exponential Backoff** - Progressive delays for violations

## 📈 Monitoring

### Metrics Available
- **System Metrics** - CPU, memory, disk usage
- **Bot Metrics** - Messages, errors, response times
- **User Metrics** - Active users, sessions, interactions
- **Performance Metrics** - Throughput, latency, availability
- **Anti-Ban Metrics** - Risk scores, mitigations, effectiveness

### Health Checks
- **Database** - Connection and query performance
- **Redis** - Connection and response times
- **Bot API** - Telegram API connectivity
- **Webhook** - Endpoint accessibility and security
- **Services** - Component health and status

### Alerting
Configure alerts based on:
- High error rates (>5%)
- High response times (>2s)
- Circuit breaker opens
- Resource usage (>80%)
- Failed health checks

## 🔒 Security

### Anti-Ban Features
- **Behavioral Profiling** - User pattern analysis
- **Natural Timing** - Realistic response delays
- **Pattern Obfuscation** - Randomized behavior
- **Risk Mitigation** - Automatic countermeasures
- **Detection Evasion** - Anti-fingerprinting

### Network Security
- **IP Filtering** - Telegram server IP validation
- **Signature Verification** - Webhook authenticity
- **Rate Limiting** - DDoS protection
- **HTTPS Only** - Encrypted communication
- **Security Headers** - XSS, CSRF protection

### Data Protection
- **Encrypted Storage** - Sensitive data encryption
- **Session Security** - Secure session management
- **Audit Logging** - Comprehensive activity logs
- **Data Retention** - Automatic cleanup policies

## ⚙️ Configuration

### Environment Variables
See `.env.example` for all configuration options.

### Rate Limiting Rules
Configure in `app/telegram/rate_limiter.py`:
```python
# Example custom rate limit
await rate_limiter.add_rule(RateLimitRule(
    name="custom_command",
    limit=10,  # 10 requests
    window=60,  # per minute
    burst_limit=3,  # allow 3 burst
    priority=8
))
```

### Anti-Ban Settings
Customize behavior patterns in `app/telegram/anti_ban.py`:
```python
# Example behavior pattern
behavior_patterns[BehaviorPattern.CUSTOM] = {
    "response_delay_range": (1.0, 3.0),
    "typing_speed_multiplier": (0.8, 1.2),
    "pause_frequency": 0.1,
    "active_hours": list(range(9, 18)),
}
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_rate_limiter.py
pytest tests/test_anti_ban.py
pytest tests/test_handlers.py
```

### Integration Tests
```bash
# Test bot integration
pytest tests/integration/

# Test webhook handling
pytest tests/integration/test_webhook.py

# Test rate limiting
pytest tests/integration/test_rate_limits.py
```

### Load Testing
```bash
# Test bot under load
python tests/load_test.py --users 1000 --duration 300

# Test webhook performance
python tests/webhook_load_test.py --concurrent 100
```

## 🐛 Troubleshooting

### Common Issues

#### Bot Not Responding
1. Check bot token: `python -c "from app.telegram.bot import get_bot; import asyncio; bot = asyncio.run(get_bot())"`
2. Verify webhook: `curl -X GET https://api.telegram.org/bot<TOKEN>/getWebhookInfo`
3. Check logs: `docker-compose logs telegram-bot`

#### High Error Rates
1. Check circuit breakers: `GET /api/v1/telegram/circuit-breakers`
2. Review error logs: `grep ERROR /app/logs/telegram-bot.log`
3. Monitor resources: `GET /api/v1/telegram/metrics`

#### Rate Limiting Issues
1. Check user limits: `GET /api/v1/telegram/rate-limits/{user_id}`
2. Review rate limit rules: `python -m app.telegram.cli rate-limits`
3. Reset if needed: `POST /api/v1/telegram/rate-limits/reset`

#### Session Problems
1. Check Redis connection: `redis-cli ping`
2. View active sessions: `GET /api/v1/telegram/sessions`
3. Run cleanup: `POST /api/v1/telegram/maintenance/cleanup`

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with detailed logs
python -m app.telegram.cli start
```

### Performance Issues
1. Monitor metrics: `GET /api/v1/telegram/metrics`
2. Check resource usage: `docker stats`
3. Scale services: `python deploy_telegram_bot.py scale --replicas 5`
4. Review slow queries: Check database logs
5. Optimize Redis: Adjust `maxmemory-policy`

## 📚 Documentation

### API Documentation
- FastAPI automatic docs: `http://localhost:8000/docs`
- ReDoc format: `http://localhost:8000/redoc`

### Code Documentation
```bash
# Generate code docs
pip install sphinx
sphinx-build -b html docs/ docs/_build/
```

### Architecture Diagrams
See `docs/architecture/` for detailed system diagrams and flow charts.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests
4. Run test suite: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open pull request

### Development Guidelines
- Follow PEP 8 style guide
- Add comprehensive tests
- Update documentation
- Use type hints
- Write descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [aiogram](https://github.com/aiogram/aiogram) - Telegram Bot framework
- [FastAPI](https://github.com/tiangolo/fastapi) - Web framework
- [Redis](https://redis.io/) - In-memory data structure store
- [PostgreSQL](https://www.postgresql.org/) - Database system
- [Prometheus](https://prometheus.io/) - Monitoring system

---

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review API documentation
- Monitor system health dashboard

Built with ❤️ for enterprise-grade Telegram bot deployments.