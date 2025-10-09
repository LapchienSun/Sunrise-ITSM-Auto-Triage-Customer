"""
Embedded Configuration Manager - DEMO Version
Updated for DEMO taxonomy requirements
"""
from dataclasses import dataclass
from typing import List, Dict
import logging
from config import prompts

logger = logging.getLogger(__name__)

# Source record types including PROBLEMS
SOURCE_RECORD_TYPES = ['INCIDENT', 'KNOWLEDGE', 'PROBLEM']

@dataclass
class CategoryConfig:
    """Configuration for ITSM categories"""
    type: List[str]
    product: List[str]
    issue: List[str]
    environment: List[str]
    priority: List[str]


@dataclass
class PriorityRule:
    """Priority assignment rule"""
    urgency: str
    impact: str
    priority: str


@dataclass
class PriorityMatrix:
    """Priority matrix configuration"""
    rules: List[PriorityRule]
    urgency_levels: Dict[str, str]
    impact_levels: Dict[str, str]


# DEMO Dependency map - updated for DEMO taxonomy
DEPENDENCY_MAP = {
    "Incident": {
        "Communications": [
            "Desk Telephone", "Email", "Fax", "Mobile Phone", "Other", 
            "Pager", "Unidentified Change", "Voicemail"
        ],
        "Documentation": [
            "Contract", "Knowledge Article", "Manual", "Other", "Process", 
            "SLA", "Unidentified Change"
        ],
        "Facilities": [
            "Building", "Car Park Space", "Floor", "Land", "Room", "Site"
        ],
        "Hardware": [
            "CRT Monitor", "Desktop PC", "Keyboard", "Laptop", "Memory", 
            "Modem", "Mouse", "Other", "PDA", "Printer", "Server", 
            "TFT Monitor", "Unidentified Change"
        ],
        "Network": [
            "Drive Access", "File Access", "Firewall", "Hardware", "Hub", 
            "Leased Line", "Other", "Router", "Unidentified Change", 
            "Virus", "VPN"
        ],
        "Other": ["Other"],
        "Software": [
            "CRM", "Database", "Email", "Internet", "Microsoft Office", 
            "Operating System", "Other"
        ]
    },
    "Service Request": {
        "Account Administration": [
            "Create Account", "Other", "Reset Account", "Reset Password", 
            "Suspend Account", "Update Account"
        ],
        "Catalogue Request": ["New Request"],
        "Equipment Move": ["Desktop PC", "Other", "Printer", "Server"],
        "Implementation": ["Hardware", "Other", "Software"],
        "Information": ["Contact", "Information Request"],
        "Leaver": ["Leaving Organisation"],
        "Microsoft Teams": ["New Team Request"],
        "Other": ["Other"],
        "Software (Clinical)": ["Winscribe"],
        "Software (Non Clinical)": ["Microsoft Teams"],
        "Upgrade": ["Hardware", "Other", "Software"]
    }
}

class ConfigManager:
    """
    Manages embedded configuration for ITSM triage system
    
    Provides ITIL-compliant taxonomy for generic IT support including:
    - Incident and Service Request types
    - Products (hardware, software, network, etc.)
    - Issues (specific problems within each product)
    - Priority matrix (urgency × impact)
    """
    
    def __init__(self):
        """Initialize configuration with DEMO taxonomy from DEPENDENCY_MAP"""
        
        # Validate DEPENDENCY_MAP structure
        self._validate_dependency_map()
        
        # Dynamically extract products and issues from DEPENDENCY_MAP
        # This ensures consistency - single source of truth
        all_products = set()
        all_issues = set()
        
        for type_name, products_dict in DEPENDENCY_MAP.items():
            all_products.update(products_dict.keys())
            for issues_list in products_dict.values():
                all_issues.update(issues_list)
        
        # Initialize categories for DEMO taxonomy
        self._categories = CategoryConfig(
            type=["Incident", "Service Request"],
            product=sorted(list(all_products)),  # Alphabetically sorted
            issue=sorted(list(all_issues)),      # Alphabetically sorted
            environment=["Live", "Test"],
            priority=["1. Critical", "2. High", "3. Medium", "4. Low"]
        )
        
        logger.info(f"ConfigManager initialized with {len(all_products)} products and {len(all_issues)} issues")
        logger.debug(f"Products: {', '.join(sorted(list(all_products))[:5])}...")
        logger.debug(f"Issues: {', '.join(sorted(list(all_issues))[:5])}...")
        
        # Initialize priority matrix directly (unchanged from original)
        self._priority_matrix = PriorityMatrix(
            rules=[
                PriorityRule("High", "High", "1. Critical"),
                PriorityRule("High", "Medium", "2. High"),
                PriorityRule("High", "Low", "3. Medium"),
                PriorityRule("Medium", "High", "2. High"),
                PriorityRule("Medium", "Medium", "3. Medium"),
                PriorityRule("Medium", "Low", "4. Low"),
                PriorityRule("Low", "High", "3. Medium"),
                PriorityRule("Low", "Medium", "4. Low"),
                PriorityRule("Low", "Low", "4. Low")
            ],
            urgency_levels={
                "High": "Resolution needed immediately - critical business functions are stopped or severely degraded",
                "Medium": "Resolution needed soon - business functions are impaired but workarounds exist",
                "Low": "Resolution can be scheduled - minor inconvenience with minimal business impact"
            },
            
            impact_levels={
                "High": "Widespread disruption - affects multiple departments, business units, or critical services",
                "Medium": "Significant disruption - affects multiple users or a single department/team",
                "Low": "Minimal disruption - affects only a single user or non-critical function"
            }
        )
        
    
    def _validate_dependency_map(self) -> None:
        """
        Validate DEPENDENCY_MAP structure for correctness
        Raises ValueError if structure is invalid
        """
        if not isinstance(DEPENDENCY_MAP, dict):
            raise ValueError("DEPENDENCY_MAP must be a dictionary")
        
        for type_name, products_dict in DEPENDENCY_MAP.items():
            if not isinstance(type_name, str):
                raise ValueError(f"Type name must be string, got {type(type_name)}")
            
            if not isinstance(products_dict, dict):
                raise ValueError(f"Products for type '{type_name}' must be a dictionary")
            
            for product_name, issues_list in products_dict.items():
                if not isinstance(product_name, str):
                    raise ValueError(f"Product name must be string, got {type(product_name)}")
                
                if not isinstance(issues_list, list):
                    raise ValueError(f"Issues for product '{product_name}' must be a list")
                
                if not issues_list:
                    logger.warning(f"Product '{product_name}' has empty issues list")
                
                for issue in issues_list:
                    if not isinstance(issue, str):
                        raise ValueError(f"Issue must be string, got {type(issue)}")
        
        logger.debug("DEPENDENCY_MAP structure validation passed")
    
    def get_products_for_type(self, type_value: str) -> List[str]:
        """
        Get valid products for a given type
        
        Args:
            type_value: The type to get products for (e.g., "Incident", "Service Request")
            
        Returns:
            List of valid product names for this type, or empty list if type not found
        """
        return list(DEPENDENCY_MAP.get(type_value, {}).keys())
    
    def get_issues_for_type_product(self, type_value: str, product_value: str) -> List[str]:
        """
        Get valid issues for a given type and product combination
        
        Args:
            type_value: The type (e.g., "Incident", "Service Request")
            product_value: The product (e.g., "Hardware", "Network")
            
        Returns:
            List of valid issue names for this type/product combination, or empty list if not found
        """
        return DEPENDENCY_MAP.get(type_value, {}).get(product_value, [])
    
    @property
    def categories(self) -> CategoryConfig:
        """Get categories configuration"""
        return self._categories
    
    @property
    def priority_matrix(self) -> PriorityMatrix:
        """Get priority matrix configuration"""
        return self._priority_matrix
    
    @property
    def _category_map(self) -> Dict[str, List[str]]:
        """
        Internal property: Category type to values mapping
        Centralizes the mapping to avoid duplication
        """
        return {
            'type': self.categories.type,
            'product': self.categories.product,
            'issue': self.categories.issue,
            'environment': self.categories.environment,
            'priority': self.categories.priority
        }
    
    def get_prompt(self) -> str:
        """Generate system prompt for AI - uses prompts module"""
        return prompts.SYSTEM_PROMPT
    
    def build_user_prompt(self, summary: str, description: str, call_source: str, 
                         retrieved_records: List[Dict], categories: CategoryConfig, 
                         priority_matrix: PriorityMatrix) -> str:
        """Build the user prompt for the AI - delegated to prompts module"""
        return prompts.build_user_prompt(
            summary=summary,
            description=description,
            call_source=call_source,
            retrieved_records=retrieved_records,
            categories=categories,
            priority_matrix=priority_matrix
        )
    
    def validate_category_value(self, category_type: str, value: str) -> bool:
        """
        Validate if a value is valid for a given category type
        
        Args:
            category_type: The category to validate against ('type', 'product', 'issue', etc.)
            value: The value to validate
            
        Returns:
            True if value is valid for the category type, False otherwise
        """
        if category_type not in self._category_map:
            return False
            
        return value in self._category_map[category_type]
    
    def get_category_options(self, category_type: str) -> List[str]:
        """
        Get available options for a category type
        
        Args:
            category_type: The category to get options for ('type', 'product', 'issue', etc.)
            
        Returns:
            List of valid values for the category type, or empty list if invalid category
        """
        return self._category_map.get(category_type, [])
    
    def get_source_types(self) -> List[str]:
        """Get valid source record types"""
        return SOURCE_RECORD_TYPES
    
    def get_priority(self, urgency: str, impact: str) -> str:
        """
        Calculate priority based on urgency and impact using ITIL v4 matrix
        
        Args:
            urgency: Urgency level ("High", "Medium", or "Low")
            impact: Impact level ("High", "Medium", or "Low")
            
        Returns:
            Priority level (e.g., "1. Critical", "2. High", "3. Medium", "4. Low")
            Defaults to "3. Medium" if no matching rule found
        """
        for rule in self.priority_matrix.rules:
            if rule.urgency == urgency and rule.impact == impact:
                return rule.priority
        
        # Default to medium priority if no rule matches
        logger.warning(f"No priority rule found for urgency='{urgency}', impact='{impact}'. Using default.")
        return "3. Medium"

    def get_categories(self) -> Dict:
        """Get all categories as a dictionary - for API endpoint"""
        return {
            "type": self.categories.type,
            "product": self.categories.product,
            "issue": self.categories.issue,
            "environment": self.categories.environment,
            "priority": self.categories.priority,
            "priority_matrix": {
                "urgency_levels": self.priority_matrix.urgency_levels,
                "impact_levels": self.priority_matrix.impact_levels,
                "rules": [
                    {
                        "urgency": rule.urgency,
                        "impact": rule.impact,
                        "priority": rule.priority
                    }
                    for rule in self.priority_matrix.rules
                ]
            }
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager