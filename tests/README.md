# ITSM Triage API Test Suite

Comprehensive test suite for the ITSM Triage API v4.1 application.

## Test Coverage

This test suite provides **72 tests** covering:

- **Configuration Management** (15 tests)
  - Taxonomy validation
  - Dependency map structure
  - Priority matrix calculations
  - Category validation

- **Constants & Configuration** (13 tests)
  - Input validation limits
  - Threshold getters
  - Environment variable overrides
  - Quality levels and scoring profiles

- **AI Service** (10 tests)
  - Embedding generation (mocked)
  - Core issue extraction (mocked)
  - Text sanitization
  - Hallucination detection
  - Response validation

- **Search Service** (13 tests)
  - Vector search with temporal filtering (mocked)
  - Hybrid search (mocked)
  - Result formatting
  - Statistical queries

- **API Endpoints** (21 tests)
  - Health check
  - Authentication and authorization
  - Triage endpoint with validation
  - Search endpoint
  - Index statistics
  - Error handling (404, 405, 413, 429, 500)

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Or install individual packages:

```bash
pip install pytest pytest-cov pytest-mock pytest-flask responses faker
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with coverage report:
```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

### Run specific test file:
```bash
pytest tests/test_config_manager.py -v
```

### Run specific test class:
```bash
pytest tests/test_api.py::TestTriageEndpoint -v
```

### Run specific test:
```bash
pytest tests/test_api.py::TestTriageEndpoint::test_triage_valid_request -v
```

### Run tests with markers:
```bash
pytest tests/ -m unit
pytest tests/ -m integration
```

### Stop on first failure:
```bash
pytest tests/ -x
```

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_config_manager.py   # Configuration and taxonomy tests
├── test_constants.py        # Constants and limits tests
├── test_ai_service.py       # AI service tests (mocked)
├── test_search_service.py   # Search service tests (mocked)
├── test_api.py              # Flask API integration tests
└── README.md                # This file
```

## Key Fixtures

### `test_env_vars` (session scope)
Sets up test environment variables for Azure OpenAI and Search credentials.

### `mock_azure_openai_client`
Mocked Azure OpenAI client for testing AI operations without API calls.

### `mock_azure_search_client`
Mocked Azure Search client for testing search operations without API calls.

### `app_with_mocks`
Flask application instance with all external services mocked.

### `client`
Flask test client for making HTTP requests to endpoints.

### `auth_headers`
Authentication headers for API requests.

### `sample_triage_request`
Sample valid triage request payload.

### `sample_search_results`
Sample search results from Azure Cognitive Search.

## Test Categories

### Unit Tests
Tests for individual components in isolation:
- Configuration manager
- Constants
- AI service (with mocked Azure OpenAI)
- Search service (with mocked Azure Search)

### Integration Tests
Tests for API endpoints with real Flask app:
- Health check
- Triage endpoint
- Search endpoint
- Authentication
- Error handling

## Mocking Strategy

The test suite uses comprehensive mocking to avoid external dependencies:

1. **Azure OpenAI Client**: Mocked to return predefined embeddings and chat responses
2. **Azure Search Client**: Mocked to return predefined search results
3. **Environment Variables**: Test-specific values set in fixtures

This ensures tests:
- Run quickly (no API calls)
- Are deterministic (same results every time)
- Don't require Azure credentials
- Can run in CI/CD pipelines

## Common Issues

### Import Errors
If you see import errors, ensure you're running tests from the project root:
```bash
cd /path/to/Sunrise-ITSM-AI-Triage_NEW-index-DEMO-v2
pytest tests/
```

### Environment Variable Conflicts
Tests set their own environment variables. If you have conflicting variables in your environment, the test fixtures will temporarily override them.

### Missing Dependencies
If tests fail due to missing packages:
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

## CI/CD Integration

To run tests in CI/CD pipelines:

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run tests with coverage
pytest tests/ --cov=. --cov-report=xml --cov-report=term-missing

# Generate coverage badge
coverage-badge -o coverage.svg -f
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test
```python
def test_my_feature(client, auth_headers):
    """Test my new feature"""
    response = client.post(
        '/api/v2/my-endpoint',
        json={'data': 'value'},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'expected_field' in data
```

### Using Mocks
```python
from unittest.mock import patch

def test_with_mock(ai_service):
    """Test with mocked external call"""
    with patch('services.ai_service.AIService.some_method') as mock_method:
        mock_method.return_value = {'result': 'value'}
        result = ai_service.some_method()
        assert result['result'] == 'value'
```

## Test Results

Current test status: **✓ All 72 tests passing**

Last run: October 2025

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all existing tests still pass
3. Aim for 80%+ code coverage
4. Add integration tests for new API endpoints
5. Mock external dependencies appropriately

## License

Same license as the main application.

