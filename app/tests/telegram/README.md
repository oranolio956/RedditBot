# Telegram Account Management Test Suite

Comprehensive test coverage for the Telegram bot system including unit tests, integration tests, end-to-end tests, safety testing, and performance benchmarks.

## Test Structure

```
app/tests/telegram/
├── unit/                           # Unit tests for individual components
│   ├── test_telegram_account_model.py    # TelegramAccount model tests
│   ├── test_safety_monitor.py            # Safety mechanisms tests  
│   └── test_conversation_engine.py       # Conversation management tests
├── integration/                    # Integration tests for services
│   ├── test_telegram_account_manager.py  # Main service integration
│   └── test_anti_detection_system.py     # Anti-detection integration
├── e2e/                           # End-to-end workflow tests
│   └── test_account_simulation.py        # Realistic operation simulation
├── performance/                   # Performance and benchmarks
│   └── test_benchmarks.py               # Response time and throughput tests
├── fixtures/                      # Test data and fixtures
│   └── telegram_test_data.py            # Sample data factory
├── mocks/                         # Mock systems
│   └── telegram_api_mock.py             # Telegram API mocking
├── conftest.py                    # Test configuration and fixtures
├── pytest.ini                    # Pytest configuration
└── README.md                      # This file
```

## Test Categories

### Unit Tests (`unit/`)
- **Model Tests**: TelegramAccount, TelegramCommunity, TelegramConversation
- **Safety Tests**: Risk scoring, flood wait handling, circuit breakers
- **Conversation Tests**: Natural timing, context management, sentiment analysis

### Integration Tests (`integration/`)
- **Account Manager**: Message processing, response generation, AI integration
- **Anti-Detection**: Behavior simulation, pattern avoidance, cooling periods
- **Database Operations**: Account persistence, conversation storage

### End-to-End Tests (`e2e/`)
- **Account Lifecycle**: Warming, activation, multi-community engagement
- **Real-World Scenarios**: Crisis response, sustained operation, recovery
- **Performance Simulation**: Concurrent conversations, high message volume

### Performance Tests (`performance/`)
- **Response Time Benchmarks**: Message analysis, response generation
- **Throughput Tests**: Messages per second, concurrent processing
- **Memory Usage**: Memory leaks, garbage collection effectiveness
- **Timing Accuracy**: Typing simulation, delay calculations

## Running Tests

### Run All Tests
```bash
cd app/tests/telegram
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# End-to-end tests only
pytest -m e2e

# Performance benchmarks
pytest -m performance

# Safety mechanism tests
pytest -m safety
```

### Run Specific Test Files
```bash
# Account model tests
pytest unit/test_telegram_account_model.py

# Safety monitor tests
pytest unit/test_safety_monitor.py

# Account manager integration
pytest integration/test_telegram_account_manager.py

# Performance benchmarks
pytest performance/test_benchmarks.py -v
```

### Run with Coverage
```bash
pytest --cov=app.services.telegram_account_manager --cov-report=html
```

### Run Performance Tests Only
```bash
pytest performance/ -v --tb=short
```

## Test Configuration

### Environment Variables
```bash
# Testing environment
export TESTING=true
export ENVIRONMENT=testing

# Optional: Enable debug logging
export LOG_LEVEL=DEBUG

# Optional: Specific test database
export TEST_DATABASE_URL=sqlite+aiosqlite:///./test_telegram.db
```

### Pytest Markers
Use markers to control test execution:

```python
@pytest.mark.unit
def test_account_creation():
    """Unit test marker"""
    pass

@pytest.mark.integration
@pytest.mark.requires_db  
async def test_account_manager_startup():
    """Integration test with database"""
    pass

@pytest.mark.e2e
@pytest.mark.slow
async def test_full_conversation_flow():
    """End-to-end test (may be slow)"""
    pass

@pytest.mark.performance
def test_response_time_benchmark():
    """Performance benchmark test"""
    pass
```

### Skip Slow Tests
```bash
# Skip slow-running tests
pytest -m "not slow"

# Skip memory-intensive tests
pytest -m "not memory_intensive"

# Skip performance tests for quick runs
pytest -m "not performance"
```

## Key Test Features

### Mock Telegram API
- Realistic API response simulation
- Network latency simulation  
- Error condition testing (flood waits, permissions, etc.)
- State tracking for verification

### Safety Testing
- Risk score calculations
- Flood wait progression
- Daily limit enforcement
- Circuit breaker patterns
- Anti-detection verification

### Performance Benchmarks
- Response time targets: < 100ms message analysis
- Throughput targets: > 50 messages/second processing
- Memory usage monitoring
- Concurrent operation testing

### Test Data Factory
- Sample accounts, communities, conversations
- Realistic message datasets
- Crisis scenario simulation
- Multi-community engagement patterns

## Test Fixtures

### Account Fixtures
```python
def sample_telegram_account():
    """Active telegram account for testing"""

def sample_telegram_community():
    """Crypto community with moderate engagement"""

def sample_telegram_conversation():
    """Ongoing conversation with context"""
```

### Mock Fixtures
```python
def mock_pyrogram_client():
    """Fully mocked Telegram client"""

def mock_database_repository():
    """Mocked database operations"""

def telegram_account_manager():
    """Account manager with mocked dependencies"""
```

### Performance Fixtures
```python
def performance_test_scenarios():
    """Performance test configuration"""

def anti_detection_test_patterns():
    """Anti-detection behavior patterns"""
```

## Safety Test Scenarios

### Flood Wait Testing
- Progressive flood wait durations
- Risk score accumulation
- Automatic backoff behavior
- Recovery timing verification

### Daily Limit Testing  
- Message count enforcement
- Group joining limits
- DM frequency limits
- Cross-limit interactions

### Anti-Detection Testing
- Human behavior simulation
- Pattern randomization
- Typing speed variation
- Response delay naturalization

## Performance Benchmarks

### Target Metrics
- **Message Analysis**: < 100ms average
- **Response Generation**: < 2s including AI
- **Safety Checks**: < 10ms average
- **Concurrent Processing**: 50+ messages/second
- **Memory Usage**: < 100MB growth per 1000 messages

### Benchmark Tests
```bash
# Run all performance benchmarks
pytest performance/test_benchmarks.py::TestResponseTimeBenchmarks -v

# Run specific benchmark
pytest performance/test_benchmarks.py::TestThroughputBenchmarks::test_message_processing_throughput -v

# Run with performance profiling
pytest performance/ --profile-svg
```

## Continuous Integration

### CI Test Matrix
- **Fast Tests**: Unit + Integration (< 5 minutes)
- **Full Tests**: All categories (< 15 minutes)  
- **Performance Tests**: Benchmarks only (< 10 minutes)
- **Memory Tests**: Memory leak detection (< 20 minutes)

### Test Reports
- Coverage reports in `htmlcov/`
- Performance metrics in console output
- Safety test results with risk scores
- Memory usage graphs (if profiling enabled)

## Debugging Tests

### Debug Mode
```bash
# Run with debug output
pytest -v -s --log-cli-level=DEBUG

# Run single test with full output
pytest unit/test_telegram_account_model.py::TestTelegramAccountModel::test_is_healthy_property -v -s

# Debug performance issues
pytest performance/ --durations=10
```

### Common Issues
1. **AsyncIO Warnings**: Use `@pytest.mark.asyncio` for async tests
2. **Mock Issues**: Ensure all async methods use `AsyncMock`
3. **Timing Issues**: Use `await asyncio.sleep(0)` to yield control
4. **Memory Issues**: Clear large objects in test cleanup

## Contributing Tests

### Test Guidelines
1. **Coverage**: Aim for > 80% coverage on critical paths
2. **Performance**: Include timing assertions for critical operations
3. **Safety**: Test error conditions and edge cases
4. **Realistic**: Use realistic data and scenarios
5. **Isolated**: Tests should not depend on external services

### Adding New Tests
1. Choose appropriate test category (unit/integration/e2e)
2. Use existing fixtures and test data where possible
3. Include performance assertions for new features
4. Add safety tests for any new risk scenarios
5. Update this README if adding new test categories

### Test Review Checklist
- [ ] Tests cover happy path and error cases
- [ ] Performance benchmarks included for new features
- [ ] Safety mechanisms tested thoroughly
- [ ] Realistic test data used
- [ ] Proper async/await usage
- [ ] Mock dependencies correctly
- [ ] Clear test names and documentation