"""
Unit tests for constants.py

Tests:
- Constant value ranges
- Threshold getters
- Environment variable overrides
"""
import pytest
import os
from config import constants


class TestConstants:
    """Test suite for constants module"""
    
    def test_input_validation_limits(self):
        """Test input validation limits are reasonable"""
        assert constants.MIN_DESCRIPTION_LENGTH > 0
        assert constants.MAX_DESCRIPTION_LENGTH > constants.MIN_DESCRIPTION_LENGTH
        assert constants.MAX_SUMMARY_LENGTH > 0
        assert constants.MAX_SOURCE_LENGTH > 0
        assert constants.MAX_REQUEST_SIZE_BYTES > 0
    
    def test_search_parameters(self):
        """Test search parameter limits"""
        assert constants.MIN_TOP_K >= 1
        assert constants.MAX_TOP_K >= constants.MIN_TOP_K
        assert constants.DEFAULT_TOP_K >= constants.MIN_TOP_K
        assert constants.DEFAULT_TOP_K <= constants.MAX_TOP_K
    
    def test_temperature_limits(self):
        """Test temperature validation limits"""
        assert constants.MIN_TEMPERATURE == 0.0
        assert constants.MAX_TEMPERATURE == 1.0
        assert constants.MIN_TEMPERATURE <= constants.MAX_TEMPERATURE
    
    def test_confidence_thresholds(self):
        """Test confidence threshold values"""
        assert 0.0 <= constants.HIGH_CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= constants.DEFAULT_INCIDENT_THRESHOLD <= 1.0
        assert 0.0 <= constants.DEFAULT_PROBLEM_THRESHOLD <= 1.0
        assert 0.0 <= constants.DEFAULT_KNOWLEDGE_THRESHOLD <= 1.0
    
    def test_get_incident_threshold_default(self):
        """Test get_incident_threshold returns default value"""
        # Remove env var if it exists
        os.environ.pop('INCIDENT_THRESHOLD', None)
        threshold = constants.get_incident_threshold()
        assert threshold == constants.DEFAULT_INCIDENT_THRESHOLD
    
    def test_get_incident_threshold_from_env(self):
        """Test get_incident_threshold reads from environment"""
        os.environ['INCIDENT_THRESHOLD'] = '0.75'
        threshold = constants.get_incident_threshold()
        assert threshold == 0.75
        os.environ.pop('INCIDENT_THRESHOLD', None)
    
    def test_get_problem_threshold_default(self):
        """Test get_problem_threshold returns default value"""
        os.environ.pop('PROBLEM_THRESHOLD', None)
        threshold = constants.get_problem_threshold()
        assert threshold == constants.DEFAULT_PROBLEM_THRESHOLD
    
    def test_get_knowledge_threshold_default(self):
        """Test get_knowledge_threshold returns default value"""
        os.environ.pop('KNOWLEDGE_THRESHOLD', None)
        threshold = constants.get_knowledge_threshold()
        assert threshold == constants.DEFAULT_KNOWLEDGE_THRESHOLD
    
    def test_quality_levels(self):
        """Test quality level constants"""
        assert constants.QUALITY_BRONZE == 'bronze'
        assert constants.QUALITY_SILVER == 'silver'
        assert constants.QUALITY_GOLD == 'gold'
        
        assert constants.QUALITY_BRONZE in constants.VALID_QUALITY_LEVELS
        assert constants.QUALITY_SILVER in constants.VALID_QUALITY_LEVELS
        assert constants.QUALITY_GOLD in constants.VALID_QUALITY_LEVELS
    
    def test_scoring_profiles(self):
        """Test scoring profile constants"""
        assert constants.SCORING_PROFILE_DEFAULT in constants.VALID_SCORING_PROFILES
        assert constants.SCORING_PROFILE_RECENT in constants.VALID_SCORING_PROFILES
        assert constants.SCORING_PROFILE_QUALITY in constants.VALID_SCORING_PROFILES
    
    def test_threadpool_configuration(self):
        """Test threadpool configuration"""
        assert constants.DEFAULT_MAX_WORKERS > 0
        assert constants.DEFAULT_MAX_WORKERS <= 10  # Reasonable limit
    
    def test_rate_limits(self):
        """Test rate limit strings are properly formatted"""
        assert 'per' in constants.DEFAULT_RATE_LIMIT
        assert 'per' in constants.TRIAGE_RATE_LIMIT
        assert 'per' in constants.SEARCH_RATE_LIMIT
        assert 'per' in constants.STATS_RATE_LIMIT
    
    def test_recent_incidents_days(self):
        """Test recent incidents cutoff"""
        assert constants.RECENT_INCIDENTS_DAYS > 0
        assert constants.RECENT_INCIDENTS_DAYS <= 365

