"""
Unit tests for config_manager.py

Tests:
- Configuration initialization
- Taxonomy validation
- Dependency map structure
- Product/issue filtering by type
- Priority matrix calculations
"""
import pytest
from config.config_manager import ConfigManager, DEPENDENCY_MAP


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initializes successfully"""
        config = ConfigManager()
        assert config is not None
        assert config.categories is not None
        assert config.priority_matrix is not None
    
    def test_dependency_map_structure(self):
        """Test DEPENDENCY_MAP has correct structure"""
        assert isinstance(DEPENDENCY_MAP, dict)
        assert 'Incident' in DEPENDENCY_MAP
        assert 'Service Request' in DEPENDENCY_MAP
        
        # Check each type has products
        for type_name, products_dict in DEPENDENCY_MAP.items():
            assert isinstance(products_dict, dict)
            assert len(products_dict) > 0
            
            # Check each product has issues
            for product_name, issues_list in products_dict.items():
                assert isinstance(issues_list, list)
                assert len(issues_list) > 0
    
    def test_categories_populated(self):
        """Test categories are properly populated"""
        config = ConfigManager()
        
        assert len(config.categories.type) == 2
        assert 'Incident' in config.categories.type
        assert 'Service Request' in config.categories.type
        
        assert len(config.categories.product) > 0
        assert len(config.categories.issue) > 0
        assert len(config.categories.environment) == 2
        assert len(config.categories.priority) == 4
    
    def test_get_products_for_type_incident(self):
        """Test getting products for Incident type"""
        config = ConfigManager()
        products = config.get_products_for_type('Incident')
        
        assert isinstance(products, list)
        assert len(products) > 0
        assert 'Hardware' in products
        assert 'Software' in products
        assert 'Network' in products
    
    def test_get_products_for_type_service_request(self):
        """Test getting products for Service Request type"""
        config = ConfigManager()
        products = config.get_products_for_type('Service Request')
        
        assert isinstance(products, list)
        assert len(products) > 0
        assert 'Account Administration' in products
        assert 'Information' in products
    
    def test_get_products_for_invalid_type(self):
        """Test getting products for invalid type returns empty list"""
        config = ConfigManager()
        products = config.get_products_for_type('InvalidType')
        
        assert isinstance(products, list)
        assert len(products) == 0
    
    def test_get_issues_for_type_product(self):
        """Test getting issues for specific type/product combination"""
        config = ConfigManager()
        issues = config.get_issues_for_type_product('Incident', 'Software')
        
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert 'Email' in issues
        assert 'Operating System' in issues
    
    def test_get_issues_for_invalid_combination(self):
        """Test getting issues for invalid type/product returns empty list"""
        config = ConfigManager()
        issues = config.get_issues_for_type_product('InvalidType', 'InvalidProduct')
        
        assert isinstance(issues, list)
        assert len(issues) == 0
    
    def test_priority_matrix_rules(self):
        """Test priority matrix has all required rules"""
        config = ConfigManager()
        
        assert len(config.priority_matrix.rules) == 9  # 3x3 matrix
        
        # Test specific rules
        priorities = {
            ('High', 'High'): '1. Critical',
            ('High', 'Medium'): '2. High',
            ('High', 'Low'): '3. Medium',
            ('Medium', 'Medium'): '3. Medium',
            ('Low', 'Low'): '4. Low'
        }
        
        for rule in config.priority_matrix.rules:
            key = (rule.urgency, rule.impact)
            if key in priorities:
                assert rule.priority == priorities[key]
    
    def test_get_priority(self):
        """Test priority calculation from urgency and impact"""
        config = ConfigManager()
        
        # Test known combinations
        assert config.get_priority('High', 'High') == '1. Critical'
        assert config.get_priority('High', 'Medium') == '2. High'
        assert config.get_priority('Medium', 'Medium') == '3. Medium'
        assert config.get_priority('Low', 'Low') == '4. Low'
    
    def test_get_priority_invalid_combination(self):
        """Test priority calculation with invalid inputs returns default"""
        config = ConfigManager()
        
        # Invalid combination should return default (3. Medium)
        result = config.get_priority('Invalid', 'Invalid')
        assert result == '3. Medium'
    
    def test_validate_category_value(self):
        """Test category value validation"""
        config = ConfigManager()
        
        # Valid values
        assert config.validate_category_value('type', 'Incident') is True
        assert config.validate_category_value('environment', 'Live') is True
        
        # Invalid values
        assert config.validate_category_value('type', 'InvalidType') is False
        assert config.validate_category_value('invalid_category', 'anything') is False
    
    def test_get_category_options(self):
        """Test getting category options"""
        config = ConfigManager()
        
        type_options = config.get_category_options('type')
        assert isinstance(type_options, list)
        assert 'Incident' in type_options
        
        invalid_options = config.get_category_options('invalid_category')
        assert isinstance(invalid_options, list)
        assert len(invalid_options) == 0
    
    def test_get_categories_dict(self):
        """Test getting all categories as dictionary"""
        config = ConfigManager()
        categories_dict = config.get_categories()
        
        assert isinstance(categories_dict, dict)
        assert 'type' in categories_dict
        assert 'product' in categories_dict
        assert 'issue' in categories_dict
        assert 'priority_matrix' in categories_dict
        
        # Check priority matrix structure
        assert 'urgency_levels' in categories_dict['priority_matrix']
        assert 'impact_levels' in categories_dict['priority_matrix']
        assert 'rules' in categories_dict['priority_matrix']
    
    def test_source_types(self):
        """Test source record types are defined"""
        config = ConfigManager()
        source_types = config.get_source_types()
        
        assert isinstance(source_types, list)
        assert 'INCIDENT' in source_types
        assert 'KNOWLEDGE' in source_types
        assert 'PROBLEM' in source_types

