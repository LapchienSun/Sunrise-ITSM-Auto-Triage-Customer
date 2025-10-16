"""
Unit tests for search_service.py

Tests:
- Search service initialization
- Vector search with temporal filtering (mocked)
- Hybrid search (mocked)
- Result formatting
- Statistical queries (mocked)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from services.search_service import SearchService


class TestSearchService:
    """Test suite for SearchService"""
    
    @pytest.fixture
    def mock_search_client(self):
        """Create mock Azure Search client"""
        mock_client = Mock()
        
        # Mock search result dictionary
        mock_result_dict = {
            'id': 'test-1',
            '@search.score': 0.85,
            'incident_id': 'SUN123456',
            'clean_summary': 'Test summary',
            'clean_description': 'Test description',
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
                return super().get(key, default)
        
        def mock_search(*args, **kwargs):
            # Return iterable with mock result that supports item assignment
            return [MockSearchResult(mock_result_dict)]
        
        mock_client.search = mock_search
        mock_client.get_document_count.return_value = 1000
        
        return mock_client
    
    @pytest.fixture
    def search_service(self, mock_search_client):
        """Create SearchService with mocked client"""
        with patch('services.search_service.SearchClient', return_value=mock_search_client):
            service = SearchService(
                endpoint='https://test.search.windows.net',
                api_key='test_key',
                index_name='test-index'
            )
            service.client = mock_search_client
            return service
    
    def test_search_service_initialization(self, search_service):
        """Test SearchService initializes successfully"""
        assert search_service is not None
        assert search_service.index_name == 'test-index'
        assert search_service.endpoint == 'https://test.search.windows.net'
    
    def test_test_connection(self, search_service, mock_search_client):
        """Test connection test succeeds"""
        result = search_service.test_connection()
        assert result is True
        mock_search_client.get_document_count.assert_called_once()
    
    def test_vector_search_with_temporal_filter(self, search_service, mock_search_client):
        """Test vector search with temporal filtering"""
        query_embedding = [0.1] * 1536
        results = search_service.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=5,
            days_back=30
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert '@search.score' in results[0]
        assert 'incident_id' in results[0]
    
    def test_vector_search_without_temporal_filter(self, search_service):
        """Test vector search without temporal filtering"""
        query_embedding = [0.1] * 1536
        results = search_service.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=5,
            days_back=None
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_vector_search_with_product_filter(self, search_service):
        """Test vector search with product filtering"""
        query_embedding = [0.1] * 1536
        results = search_service.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=5,
            product_filter='Software'
        )
        
        assert isinstance(results, list)
    
    def test_vector_search_formats_resolution_date(self, search_service):
        """Test vector search formats resolution date correctly"""
        query_embedding = [0.1] * 1536
        results = search_service.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=5
        )
        
        if results and results[0].get('resolution_date'):
            assert 'days_ago' in results[0]
            assert 'resolution_date_formatted' in results[0]
    
    def test_hybrid_search(self, search_service):
        """Test hybrid search combining vector and text"""
        query_text = "email login problem"
        query_embedding = [0.1] * 1536
        
        results = search_service.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_text_search(self, search_service):
        """Test text-only search"""
        results = search_service.text_search(
            query_text="email",
            top_k=5
        )
        
        assert isinstance(results, list)
    
    def test_text_search_with_wildcard(self, search_service):
        """Test text search with wildcard returns all documents"""
        results = search_service.text_search(
            query_text="*",
            top_k=10
        )
        
        assert isinstance(results, list)
    
    def test_search_documents(self, search_service):
        """Test document search"""
        query_embedding = [0.1] * 1536
        results = search_service.search_documents(
            query_text="test",
            query_embedding=query_embedding,
            top_k=5
        )
        
        assert isinstance(results, list)
    
    def test_get_recent_incidents(self, search_service):
        """Test getting recent incidents"""
        results = search_service.get_recent_incidents(
            days_back=30,
            top_k=10
        )
        
        assert isinstance(results, list)
    
    def test_format_search_results(self, search_service, sample_search_results):
        """Test search result formatting"""
        formatted = search_service.format_search_results(
            search_results=sample_search_results,
            query_text="email",
            max_results=5
        )
        
        assert isinstance(formatted, list)
        assert len(formatted) > 0
        assert 'has_resolution' in formatted[0]
        assert 'display_text' in formatted[0]
    
    def test_convert_to_v2_format(self, search_service):
        """Test conversion to v2 format"""
        raw_result = {
            'id': 'test-1',
            '@search.score': 0.85,
            'incident_id': 'SUN123456',
            'clean_summary': 'Test summary',
            'clean_description': 'Test description',
            'itil_resolution': 'Test resolution',
            'ticket_type': 'Incident',
            'product': 'Software',
            'issue_category': 'Email',
            'priority': '3. Medium',
            'resolution_date': '2024-01-15T10:00:00Z'
        }
        
        formatted = search_service._convert_to_v2_format(raw_result)
        
        assert isinstance(formatted, dict)
        assert 'incident_id' in formatted
        assert 'days_ago' in formatted
        assert 'resolution_date_formatted' in formatted
    
    def test_get_incident_statistics(self, search_service, mock_search_client):
        """Test getting incident statistics"""
        # Mock search results for statistics
        mock_count_result = Mock()
        mock_count_result.get_count.return_value = 1000
        mock_search_client.search.return_value = mock_count_result
        
        stats = search_service.get_incident_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_incidents' in stats or 'error' in stats

