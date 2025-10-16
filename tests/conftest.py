"""
Pytest configuration and fixtures for ITSM Triage API tests

Provides:
- Mocked Azure OpenAI client
- Mocked Azure Search client
- Flask test client
- Common test data fixtures
"""
import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add parent directory to path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='session')
def test_env_vars():
    """Set up test environment variables"""
    test_vars = {
        'FLASK_ENV': 'development',
        'FLASK_DEBUG': 'False',
        'API_KEY': 'test_api_key_for_testing_only',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test_openai_key',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': 'test-embedding',
        'AZURE_OPENAI_CHAT_DEPLOYMENT': 'test-chat',
        'AZURE_SEARCH_ENDPOINT': 'https://test.search.windows.net',
        'AZURE_SEARCH_API_KEY': 'test_search_key',
        'AZURE_SEARCH_VECTOR_INDEX': 'test-index',
        'LOG_LEVEL': 'ERROR',  # Reduce noise during tests
        'INCIDENT_THRESHOLD': '0.65',
        'PROBLEM_THRESHOLD': '0.74',
        'KNOWLEDGE_THRESHOLD': '0.74'
    }
    
    # Store original values
    original_values = {}
    for key, value in test_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_vars
    
    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_embedding():
    """Mock embedding vector (1536 dimensions for text-embedding-3-small)"""
    return [0.1] * 1536


@pytest.fixture
def mock_azure_openai_client():
    """Mock Azure OpenAI client"""
    mock_client = Mock()
    
    # Mock embeddings.create
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock chat.completions.create
    mock_chat_response = Mock()
    mock_chat_response.choices = [Mock(message=Mock(content='{"test": "response"}'))]
    mock_chat_response.usage = Mock(total_tokens=100, prompt_tokens=75, completion_tokens=25)
    mock_client.chat.completions.create.return_value = mock_chat_response
    
    return mock_client


@pytest.fixture
def mock_azure_search_client():
    """Mock Azure Search client"""
    mock_client = Mock()
    
    # Mock search results - return dictionary that supports both attribute and item access
    mock_result_dict = {
        'id': 'test-1',
        '@search.score': 0.85,
        'incident_id': 'SUN123456',
        'clean_summary': 'Test incident summary',
        'clean_description': 'Test incident description',
        'itil_resolution': 'Test resolution',
        'ticket_type': 'Incident',
        'product': 'Software',
        'issue_category': 'Email',
        'priority': '3. Medium',
        'confidence_score': 0.9,
        'data_quality': 'gold',
        'resolution_date': '2024-01-15T10:00:00Z',
        'source_record': 'INCIDENT'
    }
    
    # Create a class that supports both dict and attribute access
    class MockSearchResult(dict):
        """Mock search result that supports both dict and attribute access"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
        
        def get(self, key, default=None):
            return self.__dict__.get(key, default)
    
    # Mock search method to return iterable results
    def mock_search(*args, **kwargs):
        return [MockSearchResult(mock_result_dict)]
    
    mock_client.search = mock_search
    mock_client.get_document_count.return_value = 1000
    mock_client.get_facets.return_value = {'product': [{'value': 'Software', 'count': 100}]}
    
    return mock_client


@pytest.fixture
def sample_triage_request():
    """Sample valid triage request"""
    return {
        'description': 'User is unable to access email. Getting error message when trying to login.',
        'summary': 'Email access issue',
        'source': 'Phone',
        'temperature': 0.0,
        'top_k': 5,
        'use_semantic': True
    }


@pytest.fixture
def sample_search_results():
    """Sample search results from Azure Cognitive Search"""
    return [
        {
            'id': 'test-1',
            '@search.score': 0.85,
            'incident_id': 'SUN123456',
            'clean_summary': 'Email login failure',
            'clean_description': 'User unable to login to email',
            'itil_resolution': 'Reset password and verify account not locked',
            'ticket_type': 'Incident',
            'product': 'Communications',
            'issue_category': 'Email',
            'priority': '3. Medium',
            'confidence_score': 0.9,
            'data_quality': 'gold',
            'resolution_date': '2024-01-15T10:00:00Z',
            'days_ago': 5,
            'source_record': 'INCIDENT'
        },
        {
            'id': 'test-2',
            '@search.score': 0.75,
            'incident_id': 'PRB789012',
            'clean_summary': 'Email authentication problem',
            'clean_description': 'Multiple users experiencing email authentication failures',
            'itil_resolution': 'Known issue with authentication service. Restart auth service.',
            'ticket_type': 'Problem',
            'product': 'Communications',
            'issue_category': 'Email',
            'priority': '2. High',
            'confidence_score': 0.95,
            'data_quality': 'gold',
            'resolution_date': '2024-01-10T14:30:00Z',
            'days_ago': 10,
            'source_record': 'PROBLEM'
        }
    ]


@pytest.fixture
def app_with_mocks(test_env_vars, mock_azure_openai_client, mock_azure_search_client):
    """Create Flask app with mocked external services"""
    with patch('services.ai_service.AzureOpenAI', return_value=mock_azure_openai_client), \
         patch('services.search_service.SearchClient', return_value=mock_azure_search_client):
        from app_v2 import create_app
        app = create_app()
        app.config['TESTING'] = True
        yield app


@pytest.fixture
def client(app_with_mocks):
    """Flask test client"""
    return app_with_mocks.test_client()


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {'X-API-Key': 'test_api_key_for_testing_only'}

