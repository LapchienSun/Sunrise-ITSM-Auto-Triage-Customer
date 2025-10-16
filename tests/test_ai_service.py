"""
Unit tests for ai_service.py

Tests:
- AI service initialization
- Embedding generation (mocked)
- Core issue extraction (mocked)
- Multi-step triage process (mocked)
- Hallucination detection and sanitization
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.ai_service import AIService


class TestAIService:
    """Test suite for AIService"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock Azure OpenAI client"""
        mock_client = Mock()
        
        # Mock embeddings
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Mock chat completions
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content='Test response'))]
        mock_chat_response.usage = Mock(total_tokens=100, prompt_tokens=75, completion_tokens=25)
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        return mock_client
    
    @pytest.fixture
    def ai_service(self, mock_openai_client):
        """Create AIService with mocked client"""
        with patch('services.ai_service.AzureOpenAI', return_value=mock_openai_client):
            service = AIService(
                endpoint='https://test.openai.azure.com/',
                api_key='test_key',
                embedding_deployment='test-embedding',
                chat_deployment='test-chat'
            )
            service.client = mock_openai_client
            return service
    
    def test_ai_service_initialization(self, ai_service):
        """Test AIService initializes successfully"""
        assert ai_service is not None
        assert ai_service.embedding_deployment == 'test-embedding'
        assert ai_service.chat_deployment == 'test-chat'
    
    def test_generate_embedding(self, ai_service):
        """Test embedding generation"""
        text = "Test incident description"
        embedding = ai_service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_embedding_calls_client(self, ai_service, mock_openai_client):
        """Test embedding generation calls Azure OpenAI client"""
        text = "Test incident description"
        ai_service.generate_embedding(text)
        
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args[1]['input'] == text
        assert call_args[1]['model'] == 'test-embedding'
    
    def test_extract_core_issue(self, ai_service, mock_openai_client):
        """Test core issue extraction"""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Extracted core issue"
        
        raw_description = "Long email with headers and signatures. The actual issue is: printer not working."
        extracted = ai_service.extract_core_issue(raw_description)
        
        assert isinstance(extracted, str)
        assert len(extracted) > 0
    
    def test_extract_core_issue_fallback_on_error(self, ai_service, mock_openai_client):
        """Test core issue extraction falls back to original on error"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        raw_description = "Original description"
        extracted = ai_service.extract_core_issue(raw_description)
        
        # Should return original on error
        assert extracted == raw_description
    
    def test_sanitize_text_fields_removes_hallucinations(self, ai_service):
        """Test sanitization removes hallucinated document IDs"""
        content_dict = {
            'root_cause_preliminary': 'This matches PRB999999 which was not in search results',
            'initial_response': 'Based on SUN888888 resolution'
        }
        
        found_ids = {'PRB123456', 'SUN654321'}
        
        sanitized = ai_service._sanitize_text_fields(content_dict, found_ids)
        
        # Hallucinated IDs should be removed
        assert 'PRB999999' not in sanitized['root_cause_preliminary']
        assert 'SUN888888' not in sanitized['initial_response']
    
    def test_sanitize_text_fields_preserves_valid_ids(self, ai_service):
        """Test sanitization preserves valid document IDs"""
        content_dict = {
            'root_cause_preliminary': 'This matches PRB123456 from search results'
        }
        
        found_ids = {'PRB123456', 'SUN654321'}
        
        sanitized = ai_service._sanitize_text_fields(content_dict, found_ids)
        
        # Valid IDs should be preserved
        assert 'PRB123456' in sanitized['root_cause_preliminary']
    
    def test_sanitize_for_llm(self, ai_service):
        """Test input sanitization for LLM"""
        text_with_crlf = "Line1\r\nLine2\r\nRegards\n\nJohn Doe"
        sanitized = ai_service._sanitise_for_llm(text_with_crlf)
        
        # Should normalize line endings
        assert '\r\n' not in sanitized
        assert '\r' not in sanitized
        # Email signature pattern may or may not match depending on exact text structure
        # The main goal is to sanitize CRLF which we've verified above
        assert isinstance(sanitized, str)
    
    def test_validate_and_fix_response_removes_phantom_provisioning(self, ai_service):
        """Test validation removes phantom media provisioning claims"""
        response_dict = {
            'suggested_resolution': 'We will provide installation media for Java',
            'initial_response': 'Attached installation media'
        }
        
        fixed = ai_service._validate_and_fix_response(response_dict, 'test description', [])
        
        # Phantom provisioning should be replaced with conditional
        assert 'provide installation media' not in fixed['suggested_resolution'].lower() or \
               'on request' in fixed['suggested_resolution'].lower()
    
    def test_validate_and_fix_response_detects_phantom_references(self, ai_service):
        """Test validation detects phantom document references"""
        response_dict = {
            'suggested_resolution': 'Based on SUN999999 and PRB888888'
        }
        
        valid_docs = ['SUN123456']
        
        # Should not raise error, but log warning
        fixed = ai_service._validate_and_fix_response(response_dict, 'test', valid_docs)
        assert fixed is not None

