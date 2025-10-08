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
    """Manages embedded configuration - DEMO version"""
    
    def __init__(self):
        # Initialize categories for DEMO taxonomy
        self._categories = CategoryConfig(
            type=["Incident", "Service Request"],
            product=[  # DEMO products - alphabetised
                "Account Administration", "Catalogue Request", "Communications", 
                "Documentation", "Equipment Move", "Facilities", "Hardware", 
                "Implementation", "Information", "Leaver", "Microsoft Teams", 
                "Network", "Other", "Software", "Software (Clinical)", 
                "Software (Non Clinical)", "Upgrade"
            ],
            issue=[  # DEMO issues - alphabetised
                "Building", "CRT Monitor", "Car Park Space", "Contact", 
                "Contract", "Create Account", "CRM", "Database", 
                "Desk Telephone", "Desktop PC", "Drive Access", "Email", 
                "Fax", "File Access", "Firewall", "Floor", "Hardware", 
                "Hub", "Information Request", "Internet", "Keyboard", 
                "Knowledge Article", "Land", "Laptop", "Leased Line", 
                "Leaving Organisation", "Manual", "Memory", "Microsoft Office", 
                "Microsoft Teams", "Mobile Phone", "Modem", "Mouse", 
                "New Request", "New Team Request", "Operating System", 
                "Other", "PDA", "Pager", "Printer", "Process", 
                "Reset Account", "Reset Password", "Room", "Router", 
                "SLA", "Server", "Site", "Software", "Suspend Account", 
                "TFT Monitor", "Unidentified Change", "Update Account", 
                "VPN", "Virus", "Voicemail", "Winscribe"
            ],
            environment=["Live", "Test"],
            priority=["1. Critical", "2. High", "3. Medium", "4. Low"]
        )
        
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
        
        logger.info("ConfigManager initialized with DEMO taxonomy configuration")
    
    def get_products_for_type(self, type_value: str) -> List[str]:
        """Get valid products for a given type"""
        return list(DEPENDENCY_MAP.get(type_value, {}).keys())
    
    def get_issues_for_type_product(self, type_value: str, product_value: str) -> List[str]:
        """Get valid issues for a given type and product combination"""
        return DEPENDENCY_MAP.get(type_value, {}).get(product_value, [])
    
    @property
    def categories(self) -> CategoryConfig:
        """Get categories configuration"""
        return self._categories
    
    @property
    def priority_matrix(self) -> PriorityMatrix:
        """Get priority matrix configuration"""
        return self._priority_matrix
    
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
        """Validate if a value is valid for a given category type"""
        category_map = {
            'type': self.categories.type,
            'product': self.categories.product,
            'issue': self.categories.issue,
            'environment': self.categories.environment,
            'priority': self.categories.priority
        }
        
        if category_type not in category_map:
            return False
            
        return value in category_map[category_type]
    
    def get_category_options(self, category_type: str) -> List[str]:
        """Get available options for a category type"""
        category_map = {
            'type': self.categories.type,
            'product': self.categories.product,
            'issue': self.categories.issue,
            'environment': self.categories.environment,
            'priority': self.categories.priority
        }
        
        return category_map.get(category_type, [])
    
    def get_source_types(self) -> List[str]:
        """Get valid source record types"""
        return SOURCE_RECORD_TYPES
    
    def get_priority(self, urgency: str, impact: str) -> str:
        """Calculate priority based on urgency and impact"""
        for rule in self.priority_matrix.rules:
            if rule.urgency == urgency and rule.impact == impact:
                return rule.priority
        return "3. Medium"  # Default priority

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