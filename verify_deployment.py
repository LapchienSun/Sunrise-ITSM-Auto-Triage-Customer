#!/usr/bin/env python3
"""
Deployment Verification Script
Validates configuration and connectivity before production deployment
"""
import os
import sys
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def check_required_vars():
    """Check all required environment variables"""
    print_header("1. ENVIRONMENT VARIABLES")
    
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI endpoint URL",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "Embedding model deployment name",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "Chat model deployment name",
        "AZURE_SEARCH_ENDPOINT": "Azure Cognitive Search endpoint",
        "AZURE_SEARCH_API_KEY": "Azure Search API key",
        "AZURE_SEARCH_VECTOR_INDEX": "Vector index name"
    }
    
    optional_vars = {
        "API_KEY": "API authentication key",
        "FLASK_ENV": "Flask environment (development/production)",
        "LOG_LEVEL": "Logging level",
        "PORT": "Application port"
    }
    
    all_good = True
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "SECRET" in var:
                display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            else:
                display_value = value
            print_success(f"{var}: {display_value}")
        else:
            print_error(f"{var}: NOT SET ({description})")
            all_good = False
    
    # Check optional but recommended variables
    print(f"\n{Colors.BOLD}Optional Configuration:{Colors.ENDC}")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            else:
                display_value = value
            print_success(f"{var}: {display_value}")
        else:
            print_warning(f"{var}: Not set (using default) - {description}")
    
    return all_good

def check_security():
    """Check security configuration"""
    print_header("2. SECURITY VALIDATION")
    
    all_good = True
    
    # Check Flask environment
    flask_env = os.getenv("FLASK_ENV", "production").lower()
    if flask_env == "production":
        print_success(f"Flask environment: {flask_env}")
    else:
        print_warning(f"Flask environment: {flask_env} (not production)")
    
    # Check Flask debug mode
    flask_debug = os.getenv("FLASK_DEBUG", "False").lower()
    if flask_debug == "false":
        print_success("Flask debug mode: disabled")
    else:
        print_error("Flask debug mode: ENABLED (not recommended for production!)")
        all_good = False
    
    # Check API key strength
    api_key = os.getenv("API_KEY")
    if api_key:
        if len(api_key) >= 32:
            print_success(f"API key strength: {len(api_key)} characters (good)")
        else:
            print_warning(f"API key strength: {len(api_key)} characters (recommend 32+)")
    else:
        if flask_env == "production":
            print_error("API key: NOT SET (required for production)")
            all_good = False
        else:
            print_warning("API key: Not set (OK for development with localhost)")
    
    return all_good

def check_azure_connectivity():
    """Test connectivity to Azure services"""
    print_header("3. AZURE CONNECTIVITY")
    
    all_good = True
    
    # Test Azure OpenAI endpoint
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if openai_endpoint:
        try:
            # Just check if endpoint is reachable (don't make actual API call)
            import socket
            from urllib.parse import urlparse
            
            parsed = urlparse(openai_endpoint)
            hostname = parsed.hostname
            
            # DNS resolution check
            socket.gethostbyname(hostname)
            print_success(f"Azure OpenAI endpoint reachable: {hostname}")
        except Exception as e:
            print_error(f"Cannot reach Azure OpenAI endpoint: {str(e)}")
            all_good = False
    else:
        print_error("Azure OpenAI endpoint not configured")
        all_good = False
    
    # Test Azure Search endpoint
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    if search_endpoint:
        try:
            from urllib.parse import urlparse
            import socket
            
            parsed = urlparse(search_endpoint)
            hostname = parsed.hostname
            
            # DNS resolution check
            socket.gethostbyname(hostname)
            print_success(f"Azure Search endpoint reachable: {hostname}")
        except Exception as e:
            print_error(f"Cannot reach Azure Search endpoint: {str(e)}")
            all_good = False
    else:
        print_error("Azure Search endpoint not configured")
        all_good = False
    
    return all_good

def check_dependencies():
    """Check Python dependencies"""
    print_header("4. PYTHON DEPENDENCIES")
    
    required_packages = [
        'flask',
        'flask_cors',
        'flask_limiter',
        'openai',
        'azure-search-documents',
        'python-dotenv',
        'httpx'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_success(f"{package}: installed")
        except ImportError:
            print_error(f"{package}: NOT INSTALLED")
            all_good = False
    
    return all_good

def test_application_startup():
    """Test if application can start"""
    print_header("5. APPLICATION STARTUP TEST")
    
    try:
        print("Attempting to create Flask application...")
        from app_v2 import create_app
        
        app = create_app()
        print_success("Application created successfully")
        return True
    except Exception as e:
        print_error(f"Application failed to start: {str(e)}")
        return False

def main():
    """Main verification routine"""
    print(f"\n{Colors.BOLD}ITSM Triage API - Deployment Verification{Colors.ENDC}")
    print(f"{Colors.BOLD}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    
    results = {
        "Environment Variables": check_required_vars(),
        "Security Configuration": check_security(),
        "Azure Connectivity": check_azure_connectivity(),
        "Python Dependencies": check_dependencies(),
        "Application Startup": test_application_startup()
    }
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        if passed:
            print_success(f"{check}: PASSED")
        else:
            print_error(f"{check}: FAILED")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED - Ready for deployment{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED - Fix issues before deploying{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

