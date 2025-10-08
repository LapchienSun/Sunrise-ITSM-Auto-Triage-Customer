"""
ITSM Triage API v4.0 - Production Hardened Release
Security: Authentication, rate limiting, input validation
Performance: 3-4x faster with optimized processing
"""
from config.config_manager import get_config_manager
from config import constants
from services.ai_service import AIService
from services.search_service import SearchService
from flask import Flask, request, jsonify, render_template, Blueprint, current_app
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import os
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json
import atexit

# Load environment variables IMMEDIATELY
load_dotenv()

# Configure logging with unbuffered output
import sys

# Print immediately to verify output is working
print("=" * 80, flush=True)
print("STARTING ITSM TRIAGE API - Loading Configuration...", flush=True)
print("=" * 80, flush=True)

log_level = os.getenv("LOG_LEVEL", "INFO")

# Force flush after each log
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Configure root logger with console output only
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[FlushingStreamHandler(sys.stdout)],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# Test that logging is working
print(f"\nLogging configured at level: {log_level}", flush=True)
logger.info("Logger initialized successfully - you should see this message!")
print("", flush=True)

# ===========================
# AUTHENTICATION
# ===========================

def require_api_key(f):
    """API key authentication decorator with secure localhost bypass"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = os.getenv("API_KEY")
        is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
        is_localhost = request.remote_addr in ['127.0.0.1', 'localhost', '::1']
        
        # Only bypass authentication in explicit development mode AND from localhost
        if is_development and is_localhost:
            logger.debug(f"Development mode: bypassing API key check for localhost ({request.remote_addr})")
            return f(*args, **kwargs)
        
        # Production: Always require API_KEY to be configured
        if not api_key:
            logger.error("SECURITY: API_KEY not configured in environment variables")
            return jsonify({
                "error": "Service misconfigured",
                "message": "Authentication system not properly configured"
            }), 500
        
        # Check for provided API key
        provided_key = request.headers.get("X-API-Key")
        
        if not provided_key:
            logger.warning(f"SECURITY: Missing API key attempt from {request.remote_addr}")
            return jsonify({
                "error": "Authentication required",
                "message": "Please provide API key in X-API-Key header"
            }), 401
        
        if provided_key != api_key:
            logger.warning(f"SECURITY: Invalid API key attempt from {request.remote_addr}")
            return jsonify({
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }), 401
        
        # Valid API key - allow request
        logger.debug(f"Valid API key provided from {request.remote_addr}")
        return f(*args, **kwargs)

    return decorated_function

# ===========================
# REQUEST/RESPONSE MODELS
# ===========================

@dataclass
class TriageRequest:
    """Validated triage request model with comprehensive input validation"""
    description: str
    source: Optional[str] = None
    summary: str = ""
    temperature: float = 0.0
    top_k: int = 5
    use_semantic: bool = True
    scoring_profile: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'TriageRequest':
        """
        Create TriageRequest from dictionary with comprehensive validation
        
        Raises:
            ValueError: If validation fails with specific error message
        """
        # Validate data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        
        # ===========================
        # DESCRIPTION VALIDATION
        # ===========================
        description = data.get("description")
        
        # Required field check
        if not description:
            raise ValueError("Description is required")
        
        # Type check
        if not isinstance(description, str):
            raise ValueError("Description must be a string")
        
        # Strip whitespace
        description = description.strip()
        
        # Length validation
        if len(description) < constants.MIN_DESCRIPTION_LENGTH:
            raise ValueError(f"Description must be at least {constants.MIN_DESCRIPTION_LENGTH} characters")
        
        if len(description) > constants.MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"Description must not exceed {constants.MAX_DESCRIPTION_LENGTH:,} characters")
        
        # ===========================
        # SUMMARY VALIDATION
        # ===========================
        summary = data.get("summary", "")
        
        if summary:
            if not isinstance(summary, str):
                raise ValueError("Summary must be a string")
            
            summary = summary.strip()
            
            if len(summary) > constants.MAX_SUMMARY_LENGTH:
                raise ValueError(f"Summary must not exceed {constants.MAX_SUMMARY_LENGTH} characters")
        else:
            summary = ""
        
        # ===========================
        # SOURCE VALIDATION
        # ===========================
        source = data.get("source")
        
        if source:
            if not isinstance(source, str):
                raise ValueError("Source must be a string")
            
            source = source.strip()
            
            if len(source) > constants.MAX_SOURCE_LENGTH:
                raise ValueError(f"Source must not exceed {constants.MAX_SOURCE_LENGTH} characters")
        
        # ===========================
        # TEMPERATURE VALIDATION
        # ===========================
        try:
            temperature = float(data.get("temperature", constants.MIN_TEMPERATURE))
        except (TypeError, ValueError):
            raise ValueError("Temperature must be a number")
        
        if not constants.MIN_TEMPERATURE <= temperature <= constants.MAX_TEMPERATURE:
            raise ValueError(f"Temperature must be between {constants.MIN_TEMPERATURE} and {constants.MAX_TEMPERATURE}")
        
        # ===========================
        # TOP_K VALIDATION
        # ===========================
        try:
            top_k = int(data.get("top_k", constants.DEFAULT_TOP_K))
        except (TypeError, ValueError):
            raise ValueError("top_k must be an integer")
        
        if not constants.MIN_TOP_K <= top_k <= constants.MAX_TOP_K:
            raise ValueError(f"top_k must be between {constants.MIN_TOP_K} and {constants.MAX_TOP_K}")
        
        # ===========================
        # BOOLEAN VALIDATIONS
        # ===========================
        use_semantic = data.get("use_semantic", True)
        if not isinstance(use_semantic, bool):
            raise ValueError("use_semantic must be a boolean")
        
        # ===========================
        # SCORING PROFILE VALIDATION
        # ===========================
        scoring_profile = data.get("scoring_profile")
        
        if scoring_profile:
            if not isinstance(scoring_profile, str):
                raise ValueError("scoring_profile must be a string")
            
            # Validate against known profiles
            if scoring_profile not in constants.VALID_SCORING_PROFILES:
                logger.warning(f"Unknown scoring_profile: {scoring_profile}")
        
        return cls(
            description=description,
            source=source,
            summary=summary,
            temperature=temperature,
            top_k=top_k,
            use_semantic=use_semantic,
            scoring_profile=scoring_profile
        )

# ===========================
# HELPER FUNCTIONS
# ===========================

def process_search_results(semantic_results, reference_results):
    """
    Process search results to handle INCIDENT, PROBLEM, and KNOWLEDGE records
    Sort by relevance score only - let semantic similarity determine priority
    """
    all_results = semantic_results + reference_results
    
    # Remove duplicates by incident_id
    seen_ids = set()
    consolidated_records = []
    
    for result in all_results:
        incident_id = result.get('incident_id')
        if incident_id and incident_id not in seen_ids:
            seen_ids.add(incident_id)
            consolidated_records.append(result)
    
    # Sort by search score only - let relevance determine priority
    consolidated_records.sort(key=lambda x: x.get('@search.score', 0), reverse=True)
    
    return consolidated_records

def extract_key_insight_from_records(consolidated_records):
    """
    Extract key insight from top result, prioritising KNOWLEDGE records
    """
    key_insight_text = None
    key_insight_source_id = None
    
    if consolidated_records:
        top_result = consolidated_records[0]
        source_record = top_result.get('source_record', 'INCIDENT')
        
        resolution_text = top_result.get('itil_resolution')
        key_insight_source_id = top_result.get('incident_id')
        
        if source_record == 'KNOWLEDGE':
            record_type_label = f"Knowledge Article {top_result.get('ticket_type', 'KB')}"
        elif source_record == 'PROBLEM':
            record_type_label = f"Problem {top_result.get('ticket_type', 'Record')}"
        else:
            record_type_label = f"{top_result.get('ticket_type', 'Incident')}"
        
        if resolution_text and key_insight_source_id:
            key_insight_text = resolution_text
            logger.info(f"KEY INSIGHT EXTRACTED:")
            logger.info(f"  Source: {key_insight_source_id} ({record_type_label})")
            logger.info(f"  Text: {key_insight_text[:300] if key_insight_text else 'None'}...")
            logger.info(f"Using resolution from {record_type_label} {key_insight_source_id} as key insight")
    
    return key_insight_text, key_insight_source_id

def extract_referenced_documents(consolidated_records, max_docs=5):
    """
    Extract document references from consolidated records
    All records use incident_id field regardless of source_record type
    """
    referenced_doc_ids = []
    for record in consolidated_records[:max_docs]:
        doc_id = record.get('incident_id', '')  # Same field for both INCIDENT and PROBLEM records
        if doc_id:
            referenced_doc_ids.append(doc_id)
    
    return referenced_doc_ids

# ===========================
# APPLICATION FACTORY
# ===========================

def create_app():
    """Application factory"""
    app = Flask(__name__)
    CORS(app)

    # Configure rate limiting
    limiter_storage = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
    default_limits = os.getenv("RATE_LIMIT_DEFAULT", constants.DEFAULT_RATE_LIMIT)
    
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[default_limits],
        storage_uri=limiter_storage,
        strategy="fixed-window",  # or "moving-window" for more accuracy
        headers_enabled=True  # Send rate limit info in response headers
    )
    
    # Store limiter in app for access in routes
    app.limiter = limiter
    
    logger.info(f"[OK] Rate limiting configured: {default_limits}")
    if limiter_storage != "memory://":
        logger.info(f"     Using storage: {limiter_storage}")

    # ===========================
    # CRITICAL: Validate required environment variables at startup
    # ===========================
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "AZURE_SEARCH_API_KEY": os.getenv("AZURE_SEARCH_API_KEY"),
        "AZURE_SEARCH_VECTOR_INDEX": os.getenv("AZURE_SEARCH_VECTOR_INDEX")
    }

    missing_vars = [var for var, val in required_vars.items() if not val]
    if missing_vars:
        error_msg = f"FATAL: Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        logger.error("Application cannot start without proper configuration")
        logger.error("Please ensure all required variables are set in .env file or environment")
        logger.error("See .env.example for required configuration")
        raise RuntimeError(error_msg)
    
    logger.info("[OK] All required environment variables configured")
    
    # Validate API_KEY if not in development mode
    if os.getenv("FLASK_ENV", "production").lower() != "development":
        api_key = os.getenv("API_KEY")
        if not api_key:
            error_msg = "FATAL: API_KEY not configured for production deployment"
            logger.error(error_msg)
            logger.error("Set FLASK_ENV=development for local testing without API key")
            raise RuntimeError(error_msg)
        if len(api_key) < 32:
            logger.warning(f"WARNING: API_KEY is weak (only {len(api_key)} characters). Recommend 32+ characters for production.")
        logger.info("[OK] API_KEY configured")

    app.config_manager = get_config_manager()

    try:
        app.ai_service = AIService(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
        )
        logger.info("[OK] AI Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize AI Service: {str(e)}")
        raise RuntimeError(f"Cannot start application without AI Service: {str(e)}")

    try:
        vector_index_name = os.getenv("AZURE_SEARCH_VECTOR_INDEX", "itsm-incidents-vector-v2")
        app.search_service = SearchService(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=vector_index_name
        )
        logger.info(f"[OK] Search Service initialized ({vector_index_name})")
    except Exception as e:
        logger.error(f"Failed to initialize Search Service: {str(e)}")
        raise RuntimeError(f"Cannot start application without Search Service: {str(e)}")

    # Initialize ThreadPoolExecutor with configurable workers
    max_workers = int(os.getenv("THREADPOOL_MAX_WORKERS", str(constants.DEFAULT_MAX_WORKERS)))
    app.executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"[OK] ThreadPoolExecutor initialized (workers={max_workers})")
    
    def cleanup_executor():
        """Cleanup function to shutdown ThreadPoolExecutor gracefully"""
        if hasattr(app, 'executor'):
            logger.info("Shutting down ThreadPoolExecutor...")
            app.executor.shutdown(wait=True, cancel_futures=False)
            logger.info("[OK] ThreadPoolExecutor shutdown complete")
    
    # Register cleanup on app teardown
    atexit.register(cleanup_executor)
    
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up resources on app context teardown"""
        if exception:
            logger.error(f"App context teardown with exception: {str(exception)}")

    register_routes(app)
    register_error_handlers(app)

    return app

# ===========================
# ROUTES
# ===========================

def register_routes(app):
    """Register all routes with the app"""

    @app.route("/health", methods=["GET"])
    def health():
        """Enhanced health check with vector index info"""
        try:
            stats = app.search_service.get_incident_statistics()
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "ITSM Triage API v4.0 - Production Ready",
                "index_stats": stats
            })
        except Exception as e:
            return jsonify({
                "status": "degraded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "ITSM Triage API v4.0 - Production Ready",
                "error": str(e)
            }), 500

    @app.route("/api/v2/triage", methods=["POST"])
    @app.limiter.limit(os.getenv("RATE_LIMIT_TRIAGE", constants.TRIAGE_RATE_LIMIT))
    @require_api_key
    def triage_incident():
        """Main triage endpoint enhanced for Vector Index"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Check request size
            content_length = request.content_length
            if content_length and content_length > constants.MAX_REQUEST_SIZE_BYTES:
                return jsonify({
                    'error': 'Request too large',
                    'message': f'Request body must not exceed {constants.MAX_REQUEST_SIZE_BYTES // 1_000_000}MB'
                }), 413
            
            # Parse JSON with ITSM-friendly recovery
            # Note: ITSM systems may send customer text "as-is" without proper escaping
            raw_data = None
            
            try:
                raw_data = request.get_json()
            except Exception as json_error:
                logger.warning(f"Initial JSON parse failed from {request.remote_addr}: {str(json_error)}")
                
                # ITSM Integration: Attempt to recover from common JSON formatting issues
                # This is necessary because we cannot control how the ITSM system encodes customer text
                try:
                    raw_text = request.get_data(as_text=True)
                    
                    # Safe JSON recovery using regex-based field extraction
                    import re
                    
                    # Extract field values safely without placeholder substitution
                    recovered_data = {}
                    
                    # Pattern to match JSON string values (handles escaped quotes)
                    # This is more robust than string replacement
                    field_patterns = {
                        'description': r'"description"\s*:\s*"((?:[^"\\]|\\.)*)(?<!\\)"',
                        'summary': r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)(?<!\\)"',
                        'source': r'"source"\s*:\s*"((?:[^"\\]|\\.)*)(?<!\\)"',
                        'temperature': r'"temperature"\s*:\s*([0-9.]+)',
                        'top_k': r'"top_k"\s*:\s*([0-9]+)',
                        'use_semantic': r'"use_semantic"\s*:\s*(true|false)',
                        'scoring_profile': r'"scoring_profile"\s*:\s*"((?:[^"\\]|\\.)*)(?<!\\)"'
                    }
                    
                    for field, pattern in field_patterns.items():
                        match = re.search(pattern, raw_text, re.IGNORECASE)
                        if match:
                            value = match.group(1)
                            
                            # Unescape only valid JSON escape sequences
                            if field in ['description', 'summary', 'source', 'scoring_profile']:
                                # Handle escaped quotes and backslashes properly
                                value = value.replace('\\"', '"')
                                value = value.replace('\\\\', '\\')
                                value = value.replace('\\/', '/')
                                # Leave other sequences as-is (they're likely literal backslashes from Windows paths)
                                recovered_data[field] = value
                            elif field == 'temperature':
                                recovered_data[field] = float(value)
                            elif field == 'top_k':
                                recovered_data[field] = int(value)
                            elif field == 'use_semantic':
                                recovered_data[field] = value.lower() == 'true'
                    
                    if recovered_data:
                        raw_data = recovered_data
                        logger.info(f"Successfully recovered JSON data from ITSM system (likely unescaped Windows paths)")
                        logger.info(f"Recovered fields: {list(recovered_data.keys())}")
                    else:
                        raise ValueError("Could not extract any valid fields from request")
                        
                except Exception as recovery_error:
                    logger.error(f"JSON recovery failed: {str(recovery_error)}")
                    return jsonify({
                        'error': 'Invalid request format',
                        'message': 'Unable to parse request body as JSON',
                        'details': str(json_error),
                        'help': 'Ensure the ITSM system is sending valid JSON format. Common issue: Windows paths with unescaped backslashes.'
                    }), 400

            # Validate raw_data was successfully parsed
            if raw_data is None:
                return jsonify({
                    'error': 'Empty request body',
                    'message': 'Request body is required'
                }), 400
                
            logger.info(f"RAW REQUEST DATA: {json.dumps(raw_data, indent=2)}")
            
            # Validate request with comprehensive checks
            try:
                triage_request = TriageRequest.from_dict(raw_data)
            except ValueError as ve:
                logger.warning(f"Validation error: {str(ve)}")
                return jsonify({
                    'error': 'Validation error',
                    'message': str(ve)
                }), 400
            
            # ADDED: Extract core issue before creating embeddings for better similarity matching
            try:
                extracted_description = app.ai_service.extract_core_issue(triage_request.description)
                logger.info(f"Description extraction - Original: {len(triage_request.description)} chars, Extracted: {len(extracted_description)} chars")
                
                # Use extracted description for embeddings and query text
                query_text = f"{triage_request.summary} {extracted_description}".strip()
            except Exception as e:
                logger.warning(f"Description extraction failed: {str(e)}, using original description")
                # Fallback to original if extraction fails
                query_text = f"{triage_request.summary} {triage_request.description}".strip()
            
            call_source = triage_request.source or "API"
            
            logger.info(f"Processing triage request - Source: {call_source}")

            # Generate embedding for vector search using extracted/cleaned query text
            query_embedding = app.ai_service.generate_embedding(query_text)

            # SIMPLIFIED SEARCH: Vector Index v2 contains full incidents, not chunks
            # No need for complex problem record consolidation
            
            # 1. Perform semantic search on the vector index
            logger.info("Performing semantic search on Vector Index v2")
            semantic_results = app.search_service.vector_search_with_temporal_filter(
                query_embedding=query_embedding,
                top_k=triage_request.top_k,
                days_back=None,  # No temporal filtering
                min_confidence=constants.MIN_SEMANTIC_CONFIDENCE,
                data_quality=None  # Allow all quality levels
            )

            # 2. Search for high-quality reference documents 
            logger.info("Searching for reference documents")
            reference_results = app.search_service.vector_search_with_temporal_filter(
                query_embedding=query_embedding,
                top_k=constants.MAX_SIMILAR_RECORDS,
                days_back=None,
                min_confidence=constants.MIN_REFERENCE_CONFIDENCE,
                data_quality=constants.QUALITY_GOLD  # High quality only for references
            )

            # 3. Combine results using updated processing for INCIDENT and PROBLEM records
            consolidated_records = process_search_results(semantic_results, reference_results)
            
            # Take top results
            consolidated_records = consolidated_records[:triage_request.top_k]
            logger.info("CONSOLIDATED RECORDS FOR AI:")
            for i, record in enumerate(consolidated_records):
                source_record = record.get('source_record', 'INCIDENT')
                record_id = record.get('incident_id')
                logger.info(f"  Record {i+1}: {record_id} ({source_record}) - Quality: {record.get('data_quality')}")
                
                # All records use the same resolution field
                resolution = record.get('itil_resolution')
                if resolution:
                    logger.info(f"    Resolution: {resolution[:200]}...")
                
                if record.get('clean_summary'):
                    logger.info(f"    Summary: {record.get('clean_summary')[:100]}...")
                    
            logger.info(f"Consolidated results count: {len(consolidated_records)}")
            for i, result in enumerate(consolidated_records):
                source_record = result.get('source_record', 'INCIDENT')
                record_id = result.get('incident_id')
                logger.info(f"  Result {i+1}: {record_id} ({source_record}) "
                        f"(Score: {result.get('@search.score', 0):.3f}, "
                        f"Quality: {result.get('data_quality', 'unknown')}, "
                        f"Age: {result.get('days_ago', 'unknown')} days)")

            # 4. Extract key insight from top result using updated logic
            key_insight_text, key_insight_source_id = extract_key_insight_from_records(consolidated_records)

            # 5. Generate AI response using ORIGINAL description for full context
            all_found_ids = set()
            for r in semantic_results + reference_results:
                record_id = r.get('incident_id')
                if record_id:
                    all_found_ids.add(record_id)
            
            ai_response = app.ai_service.generate_optimized_multistep_triage(
                summary=triage_request.summary,
                description=triage_request.description,  # Use ORIGINAL description for AI synthesis
                call_source=call_source,
                retrieved_records=consolidated_records,
                config_manager=app.config_manager,
                temperature=triage_request.temperature,
                found_document_ids=all_found_ids,
                key_insight_text=key_insight_text,
                key_insight_source_id=key_insight_source_id
            )

            # Extract the final 5 document references used for triage
            referenced_doc_ids = extract_referenced_documents(consolidated_records, max_docs=5)

            # Add referenced_documents to the AI response as array of individual strings
            ai_response["referenced_documents"] = referenced_doc_ids

            # Add enhanced metadata
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            ai_response["metadata"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_length": len(query_text),
                "results_found": len(consolidated_records),
                "api_version": "4.0-production-hardened",
                "processing_time_seconds": round(processing_time, 2),
                "referenced_document_count": len(referenced_doc_ids),
                "search_config": {
                    "scoring_profile": triage_request.scoring_profile,
                    "semantic_enabled": triage_request.use_semantic,
                    "top_k": triage_request.top_k,
                    "index": os.getenv("AZURE_SEARCH_VECTOR_INDEX", "itsm-incidents-vector-v2"),
                    "index_type": "Vector v2 - Full Records with Problems and Knowledge"
                },
                "result_quality": {
                    "avg_confidence": sum(r.get('confidence_score', 0) for r in consolidated_records) / len(consolidated_records) if consolidated_records else 0,
                    "quality_distribution": {
                        quality: sum(1 for r in consolidated_records if r.get('data_quality') == quality)
                        for quality in ['gold', 'silver', 'bronze']
                    },
                    "avg_age_days": sum(r.get('days_ago', 0) for r in consolidated_records if r.get('days_ago') is not None) / len([r for r in consolidated_records if r.get('days_ago') is not None]) if consolidated_records else None,
                    "record_type_distribution": {
                        "INCIDENT": sum(1 for r in consolidated_records if r.get('source_record', 'INCIDENT') == 'INCIDENT'),
                        "PROBLEM": sum(1 for r in consolidated_records if r.get('source_record') == 'PROBLEM'),
                        "KNOWLEDGE": sum(1 for r in consolidated_records if r.get('source_record') == 'KNOWLEDGE')  # Added KNOWLEDGE tracking
                    }
                }
            }

            logger.info(f"Triage completed in {processing_time:.2f}s using Vector Index")
            return jsonify(ai_response)

        except Exception as e:
            logger.error(f"Error in triage_incident: {str(e)}", exc_info=True)
            # Don't expose internal error details in production
            is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
            error_message = str(e) if is_development else "An unexpected error occurred during triage"
            return jsonify({"error": "Internal server error", "message": error_message}), 500

    @app.route("/api/v2/search-recent", methods=["POST"])
    @app.limiter.limit(os.getenv("RATE_LIMIT_SEARCH", constants.SEARCH_RATE_LIMIT))
    @require_api_key
    def search_recent_incidents():
        """NEW: Search recent high-quality incidents and problems"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body is required"}), 400
            
            # Validate query
            query = data.get("query", "")
            if not query:
                return jsonify({"error": "Query is required"}), 400
            
            if not isinstance(query, str):
                return jsonify({"error": "Query must be a string"}), 400
            
            if len(query) > constants.MAX_SUMMARY_LENGTH:
                return jsonify({"error": f"Query must not exceed {constants.MAX_SUMMARY_LENGTH} characters"}), 400
            
            # Validate days_back
            try:
                days_back = int(data.get("days_back", constants.RECENT_INCIDENTS_DAYS))
            except (TypeError, ValueError):
                return jsonify({"error": "days_back must be an integer"}), 400
            
            if not constants.MIN_DAYS_BACK <= days_back <= constants.MAX_DAYS_BACK:
                return jsonify({"error": f"days_back must be between {constants.MIN_DAYS_BACK} and {constants.MAX_DAYS_BACK}"}), 400
            
            # Validate top_k
            try:
                top_k = int(data.get("top_k", 10))
            except (TypeError, ValueError):
                return jsonify({"error": "top_k must be an integer"}), 400
            
            if not constants.MIN_SEARCH_TOP_K <= top_k <= constants.MAX_SEARCH_TOP_K:
                return jsonify({"error": f"top_k must be between {constants.MIN_SEARCH_TOP_K} and {constants.MAX_SEARCH_TOP_K}"}), 400
            
            # Generate embedding
            query_embedding = app.ai_service.generate_embedding(query)
            
            # Search with temporal filter
            results = app.search_service.vector_search_with_temporal_filter(
                query_embedding=query_embedding,
                top_k=top_k,
                days_back=days_back,
                min_confidence=constants.MIN_SEMANTIC_CONFIDENCE,
                data_quality=constants.QUALITY_GOLD
            )
            
            formatted_results = []
            for r in results:
                source_record = r.get('source_record', 'INCIDENT')
                record_id = r.get('incident_id')  # Same field for both record types
                
                formatted_results.append({
                    "record_id": record_id,
                    "source_record": source_record,
                    "ticket_type": r.get('ticket_type'),
                    "summary": r.get('clean_summary'),
                    "product": r.get('product'),
                    "priority": r.get('priority'),
                    "confidence_score": r.get('confidence_score'),
                    "days_ago": r.get('days_ago'),
                    "resolution_date": r.get('resolution_date_formatted'),
                    "search_score": round(r.get("@search.score"), 3)
                })
            
            return jsonify({
                "query": query,
                "days_back": days_back,
                "results_count": len(formatted_results),
                "results": formatted_results
            })
            
        except Exception as e:
            logger.error(f"Error in search_recent_incidents: {str(e)}", exc_info=True)
            # Don't expose internal error details in production
            is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
            error_message = str(e) if is_development else "An unexpected error occurred during search"
            return jsonify({"error": "Internal server error", "message": error_message}), 500

    @app.route("/api/v2/index-stats", methods=["GET"])
    @app.limiter.limit(os.getenv("RATE_LIMIT_STATS", constants.STATS_RATE_LIMIT))
    @require_api_key
    def get_index_statistics():
        """NEW: Get vector index statistics"""
        try:
            stats = app.search_service.get_incident_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            # Don't expose internal error details in production
            is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
            error_message = str(e) if is_development else "Failed to retrieve index statistics"
            return jsonify({"error": "Internal server error", "message": error_message}), 500

# ===========================
# ERROR HANDLERS
# ===========================

def register_error_handlers(app):
    """Register error handlers"""
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not Found", "message": "Resource not found"}), 404
    
    @app.errorhandler(413)
    def request_too_large(error):
        logger.warning(f"REQUEST TOO LARGE: {request.remote_addr} sent oversized request to {request.path}")
        return jsonify({
            "error": "Request Entity Too Large",
            "message": f"Request body exceeds maximum size of {constants.MAX_REQUEST_SIZE_BYTES // 1_000_000}MB"
        }), 413

    @app.errorhandler(429)
    def ratelimit_handler(error):
        logger.warning(f"RATE LIMIT: {request.remote_addr} exceeded rate limit on {request.path}")
        return jsonify({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded. Please slow down your requests.",
            "retry_after": error.description
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}", exc_info=True)
        # Don't expose internal error details in production
        is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
        error_message = str(error) if is_development else "An unexpected error occurred"
        return jsonify({"error": "Internal Server Error", "message": error_message}), 500
    
    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """Catch-all handler for any unhandled exceptions"""
        logger.error(f"UNHANDLED EXCEPTION: {str(error)}", exc_info=True)
        # Don't expose internal error details in production
        is_development = os.getenv("FLASK_ENV", "production").lower() == "development"
        error_message = str(error) if is_development else "An unexpected error occurred"
        return jsonify({"error": "Internal Server Error", "message": error_message}), 500

# ===========================
# MAIN ENTRY POINT
# ===========================

# Create app instance
app = create_app()

if __name__ == "__main__":
    print("\n" + "="*80, flush=True)
    print("ITSM TRIAGE API v4.0 - PRODUCTION READY", flush=True)
    print("="*80, flush=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ITSM Triage API v4.0 - Production Hardened Starting...")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Vector Index: {os.getenv('AZURE_SEARCH_VECTOR_INDEX', 'itsm-incidents-vector-v2')}")
    logger.info(f"{'='*60}\n")

    port = int(os.getenv("PORT", 8000))
    is_debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"Starting server on http://0.0.0.0:{port}", flush=True)
    print(f"Debug mode: {is_debug_mode}", flush=True)
    print(f"API endpoint: http://localhost:{port}/api/v2/triage", flush=True)
    print("="*80 + "\n", flush=True)
    
    logger.info(f"Starting server on http://0.0.0.0:{port}")
    logger.info(f"Debug mode: {is_debug_mode}")
    logger.info(f"API endpoint: http://localhost:{port}/api/v2/triage")
    logger.info(f"NEW: Recent search: http://localhost:{port}/api/v2/search-recent")
    logger.info(f"NEW: Index stats: http://localhost:{port}/api/v2/index-stats")
    
    # Disable Flask's default logger to avoid conflicts
    import logging as flask_logging
    flask_logging.getLogger('werkzeug').setLevel(flask_logging.WARNING)
    
    app.run(host="0.0.0.0", port=port, debug=is_debug_mode, use_reloader=False)