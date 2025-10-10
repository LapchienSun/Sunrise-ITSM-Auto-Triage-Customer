"""
Configuration Constants for ITSM Triage API
Centralized location for all magic numbers and thresholds
"""

# ===========================
# INPUT VALIDATION LIMITS
# ===========================

# Description validation
MIN_DESCRIPTION_LENGTH = 10
MAX_DESCRIPTION_LENGTH = 10_000

# Summary validation
MAX_SUMMARY_LENGTH = 500

# Source validation
MAX_SOURCE_LENGTH = 100

# Request size limits
MAX_REQUEST_SIZE_BYTES = 1_000_000  # 1 MB

# Search parameters
MIN_TOP_K = 1
MAX_TOP_K = 20

# Search parameters for recent search endpoint
MIN_SEARCH_TOP_K = 1
MAX_SEARCH_TOP_K = 50

# Temperature validation
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0

# Days back validation
MIN_DAYS_BACK = 1
MAX_DAYS_BACK = 365

# ===========================
# SEARCH CONFIDENCE THRESHOLDS
# ===========================

# High confidence match threshold - used for resolution fidelity
# This is based on VECTOR SIMILARITY (@search.score), not synthesis confidence
HIGH_CONFIDENCE_THRESHOLD = 0.70

# DEPRECATED: These were incorrectly filtering on confidence_score (synthesis quality)
# confidence_score reflects how well data was synthesized during indexing, NOT match relevance
# Vector similarity (@search.score) is the only relevant metric for matching
# MIN_SEMANTIC_CONFIDENCE = 0.70  # DEPRECATED - do not use
# MIN_REFERENCE_CONFIDENCE = 0.80  # DEPRECATED - do not use

# ===========================
# SEARCH RESULT LIMITS
# ===========================

# Maximum number of similar records to process
MAX_SIMILAR_RECORDS = 5

# Maximum number of referenced documents in response
MAX_REFERENCED_DOCUMENTS = 5

# Default number of search results
DEFAULT_TOP_K = 5

# Number of records to show in prompts
MAX_PROMPT_RECORDS = 5

# ===========================
# AI PROCESSING LIMITS
# ===========================

# Maximum tokens for core issue extraction
MAX_EXTRACTION_TOKENS = 400

# Extraction timeout (seconds)
EXTRACTION_TIMEOUT_SECONDS = 10

# Maximum tokens for resolution content
MAX_RESOLUTION_TOKENS = 800

# ===========================
# THREADPOOL CONFIGURATION
# ===========================

# Default maximum workers for ThreadPoolExecutor
DEFAULT_MAX_WORKERS = 3

# ===========================
# RATE LIMITING DEFAULTS
# ===========================

# Default rate limit (can be overridden by environment variables)
DEFAULT_RATE_LIMIT = "200 per hour, 50 per minute"

# Triage endpoint rate limit (requests per minute)
TRIAGE_RATE_LIMIT = "10 per minute"

# Search endpoint rate limit (requests per minute)
SEARCH_RATE_LIMIT = "30 per minute"

# Stats endpoint rate limit (requests per minute)
STATS_RATE_LIMIT = "60 per minute"

# ===========================
# QUALITY LEVELS
# ===========================

QUALITY_BRONZE = "bronze"
QUALITY_SILVER = "silver"
QUALITY_GOLD = "gold"

# Valid quality levels
VALID_QUALITY_LEVELS = [QUALITY_BRONZE, QUALITY_SILVER, QUALITY_GOLD]

# ===========================
# SCORING PROFILES
# ===========================

SCORING_PROFILE_DEFAULT = "default"
SCORING_PROFILE_RECENT = "recent"
SCORING_PROFILE_QUALITY = "quality"

# Valid scoring profiles
VALID_SCORING_PROFILES = [
    SCORING_PROFILE_DEFAULT,
    SCORING_PROFILE_RECENT,
    SCORING_PROFILE_QUALITY
]

# ===========================
# VECTOR SEARCH PARAMETERS
# ===========================

# Default days back for temporal filtering (None = no filter)
DEFAULT_DAYS_BACK = None

# Recent incidents cutoff (for statistics)
RECENT_INCIDENTS_DAYS = 30

# ===========================
# LOGGING
# ===========================

# Preview length for logged text
LOG_TEXT_PREVIEW_LENGTH = 200

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

