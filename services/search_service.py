"""
SearchService for Vector Index v2 - Full Incident Search with Temporal Filtering

This module provides comprehensive Azure Cognitive Search operations for the ITSM triage system:
- Vector search with embeddings for semantic similarity
- Hybrid search combining vector and text search
- Temporal filtering for recent incidents
- Quality filtering (bronze, silver, gold)
- Statistical analysis and trending
- Support for multiple record types (INCIDENT, PROBLEM, KNOWLEDGE)

Version: 4.1 - Vector Index v2 Schema
"""
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from config import constants

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for handling Azure Cognitive Search operations with Vector Index v2.
    
    Provides unified interface for searching incident records using:
    - Pure vector search for semantic similarity
    - Hybrid search (vector + text) for comprehensive matching
    - Text-only search for keyword matching
    - Temporal and quality filters
    - Statistical aggregations
    """
    
    def __init__(self, endpoint: str, api_key: str, index_name: str):
        """
        Initialize search service with Azure credentials.
        
        Args:
            endpoint: Azure Search service endpoint URL
            api_key: Azure Search API key for authentication
            index_name: Name of the search index to query
        """
        self.endpoint = endpoint
        self.index_name = index_name
        
        # Create search client
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
        
        logger.info(f"SearchService initialized for vector index: {index_name}")
        logger.info("Using Vector Index v2 schema with full incident documents")
    
    def test_connection(self):
        """
        Test connection to Azure Search and retrieve document count.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            Exception: If connection fails
        """
        try:
            result = self.client.get_document_count()
            logger.info(f"Connected to vector index. Document count: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vector index: {str(e)}")
            raise
    
    def vector_search_with_temporal_filter(self, query_embedding: List[float], 
                                         top_k: int = 10,
                                         days_back: Optional[int] = None,
                                         min_confidence: Optional[float] = None,
                                         data_quality: Optional[str] = None,
                                         product_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search with temporal and quality filtering.
        
        This is the core search method that supports multiple filter types:
        - Temporal: Filter by resolution date (recent incidents prioritized)
        - Confidence: Filter by document confidence score
        - Quality: Filter by data quality tier (bronze/silver/gold)
        - Product: Filter by product category
        
        Args:
            query_embedding: Vector embedding of the query text
            top_k: Maximum number of results to return (default: 10)
            days_back: Optional days to look back for temporal filtering
            min_confidence: Optional minimum confidence score threshold
            data_quality: Optional quality tier filter (bronze/silver/gold)
            product_filter: Optional product category filter
            
        Returns:
            List of formatted search results with scores, metadata, and age information
            
        Raises:
            Exception: If search operation fails
        """
        try:
            # Build filter expression
            filter_parts = []
            
            # Temporal filter - recent incidents first
            if days_back:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back))
                filter_parts.append(f"resolution_date ge {cutoff_date.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                logger.info(f"🔍 Vector search (last {days_back} days)")
            else:
                logger.info(f"🔍 Vector search (all dates)")
            
            # NOTE: min_confidence parameter intentionally not used for filtering
            # confidence_score in indexed data reflects synthesis quality, not match relevance
            # Vector similarity (@search.score) is the only relevant metric for matching
            if min_confidence:
                logger.warning(f"  ⚠️ min_confidence parameter ignored - synthesis confidence not relevant for matching")
            
            if data_quality:
                filter_parts.append(f"data_quality eq '{data_quality}'")
            
            if product_filter:
                filter_parts.append(f"product eq '{product_filter}'")
                logger.info(f"  Filtering to product: {product_filter}")
            
            filter_expression = " and ".join(filter_parts) if filter_parts else None
            
            # Build vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector",
                exhaustive=False
            )
            
            search_params = {
                "vector_queries": [vector_query],
                "top": top_k,
                "select": [
                    "id", "incident_id", "clean_summary", "clean_description",
                    "itil_resolution", "ticket_type", "product", "issue_category",
                    "priority", "confidence_score", "data_quality", "resolution_date",
                    "source_record"
                ]
            }
            
            if filter_expression:
                search_params["filter"] = filter_expression
            
            results = self.client.search(**search_params)
            
            formatted_results = []
            for result in results:
                # Format resolution date for display
                resolution_date = result.get("resolution_date")
                if resolution_date:
                    try:
                        date_obj = datetime.fromisoformat(resolution_date.replace('Z', '+00:00'))
                        days_ago = (datetime.now(timezone.utc) - date_obj).days
                        result["days_ago"] = days_ago
                        result["resolution_date_formatted"] = date_obj.strftime('%Y-%m-%d')
                    except Exception as e:
                        logger.warning(f"Could not parse resolution_date: {e}")
                        result["days_ago"] = None
                        result["resolution_date_formatted"] = "Unknown"
                
                formatted_results.append({
                    "id": result.get("id"),
                    "@search.score": result.get("@search.score"),
                    "incident_id": result.get("incident_id"),
                    "clean_summary": result.get("clean_summary"),
                    "clean_description": result.get("clean_description"),
                    "itil_resolution": result.get("itil_resolution"),
                    "ticket_type": result.get("ticket_type"),
                    "product": result.get("product"),
                    "issue_category": result.get("issue_category"),
                    "priority": result.get("priority"),
                    "confidence_score": result.get("confidence_score"),
                    "data_quality": result.get("data_quality"),
                    "resolution_date": result.get("resolution_date"),
                    "resolution_date_formatted": result.get("resolution_date_formatted"),
                    "days_ago": result.get("days_ago"),
                    "source_record": result.get("source_record")
                })
            
            logger.info(f"✅ Vector search returned {len(formatted_results)} results")
            if formatted_results:
                top_result = formatted_results[0]
                logger.info(
                    f"  Top result: {top_result['incident_id']} "
                    f"(Search score: {top_result['@search.score']:.3f}, "
                    f"Age: {top_result.get('days_ago', 'Unknown')} days)"
                )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in vector_search_with_temporal_filter: {str(e)}", exc_info=True)
            raise

    def hybrid_search(self, query_text: str, query_embedding: List[float], 
                     top_k: int = 10, use_semantic: bool = True,
                     filter_expression: str = None, 
                     scoring_profile: str = None) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search combining vector + text with temporal intelligence.
        
        Performs a three-step search process:
        1. Vector search for semantic similarity (70% weight)
        2. Text search for keyword matching (30% weight)
        3. Combines results with quality boosting (gold: 1.3x, silver: 1.1x)
        
        Args:
            query_text: Text query for keyword matching
            query_embedding: Vector embedding for semantic search
            top_k: Maximum number of results to return (default: 10)
            use_semantic: Whether to use semantic search capabilities (default: True)
            filter_expression: Optional OData filter expression
            scoring_profile: Optional Azure Search scoring profile name
            
        Returns:
            List of ranked search results with combined scores and metadata
            
        Note:
            Falls back to vector-only search if hybrid search fails
        """
        try:
            logger.info("🚀 Starting Vector v2 hybrid search")
            
            # Step 1: Vector search
            vector_results = self.vector_search_with_temporal_filter(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more candidates
                days_back=None,  # No temporal filtering
                min_confidence=None  # Don't filter on synthesis confidence
            )
            
            # Step 2: Text search for keyword matching
            text_params = {
                "search_text": query_text,
                "top": top_k * 2,
                "include_total_count": True,
                "select": [
                    "id", "incident_id", "clean_summary", "clean_description",
                    "itil_resolution", "ticket_type", "product", "issue_category",
                    "priority", "confidence_score", "data_quality", "resolution_date",
                    "source_record"
                ]
            }
            
            if filter_expression:
                text_params["filter"] = filter_expression
            
            if use_semantic:
                # Vector Index v2 doesn't have semantic configuration yet
                # Use full-text search instead until semantic config is added
                text_params["query_type"] = "full"
                text_params["search_mode"] = "all"
                # Remove semantic-specific parameters
                # text_params["semantic_configuration_name"] = "default" 
                # text_params["query_caption"] = "extractive"
                # text_params["query_caption_highlight_enabled"] = True
            
            text_results = list(self.client.search(**text_params))
            
            # Step 3: Combine and rank results
            result_map = {}
            
            # Process vector results
            for i, result in enumerate(vector_results):
                doc_id = result.get("id")
                vector_score = result.get("@search.score", 0)
                
                # REMOVED: Quality boost multiplier - let raw scores stand as they are
                # Previously applied 1.3x for gold, 1.1x for silver
                # quality_boost = 1.0
                # if result.get("data_quality") == "gold":
                #     quality_boost = 1.3
                # elif result.get("data_quality") == "silver":
                #     quality_boost = 1.1
                # combined_score = vector_score * quality_boost
                
                # Use raw vector score without boosting
                combined_score = vector_score
                
                result_map[doc_id] = {
                    **result,
                    "vector_score": vector_score,
                    "quality_boost": 1.0,  # No longer applying boost, kept for backward compatibility
                    "@search.score": combined_score,
                    "text_score": 0,
                    "search_type": "vector"
                }
            
            # Process text results and enhance existing entries
            for i, result in enumerate(text_results):
                doc_id = result.get("id")
                text_score = result.get("@search.score", 0)
                
                if doc_id in result_map:
                    # Enhance existing vector result
                    existing = result_map[doc_id]
                    # Combine scores with vector preference (no quality boost applied)
                    combined_score = (existing["vector_score"] * 0.7) + (text_score * 0.3)
                    # REMOVED: Quality boost multiplier
                    # combined_score *= existing["quality_boost"]
                    
                    result_map[doc_id]["@search.score"] = combined_score
                    result_map[doc_id]["text_score"] = text_score
                    result_map[doc_id]["search_type"] = "hybrid"
                    
                    # Add semantic captions if available
                    if "@search.captions" in result:
                        result_map[doc_id]["@search.captions"] = result["@search.captions"]
                else:
                    # Add text-only result with conversion to new schema
                    formatted_result = self._convert_to_v2_format(result)
                    result_map[doc_id] = {
                        **formatted_result,
                        "vector_score": 0,
                        "text_score": text_score,
                        "@search.score": text_score * 0.8,  # Slight penalty for text-only
                        "quality_boost": 1.0,
                        "search_type": "text",
                        "@search.captions": result.get("@search.captions", [])
                    }
            
            # Sort by combined score and return top results
            sorted_results = sorted(
                result_map.values(), 
                key=lambda x: x["@search.score"], 
                reverse=True
            )[:top_k]
            
            logger.info(f"✅ Hybrid search returned {len(sorted_results)} results")
            logger.info("=== TOP 3 RESULTS ===")
            for i, result in enumerate(sorted_results[:3], 1):
                logger.info(f"{i}. {result['incident_id']} - "
                          f"Score: {result['@search.score']:.3f} "
                          f"(Type: {result['search_type']}, "
                          f"Age: {result.get('days_ago', 'Unknown')} days)")
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error in hybrid_search: {str(e)}", exc_info=True)
            # Fallback to vector search only
            return self.vector_search_with_temporal_filter(query_embedding, top_k)

    def search_documents(self, query_text: str, query_embedding: List[float], 
                        top_k: int = 10, filter_expression: str = None,
                        scoring_profile: str = None, use_semantic: bool = True) -> List[Dict[str, Any]]:
        """
        Search for document-like incidents.
        
        Searches for incidents with good resolutions, effectively treating
        resolved incidents as knowledge base articles.
        
        Args:
            query_text: Text query (not used in this implementation)
            query_embedding: Vector embedding for semantic search
            top_k: Maximum number of results to return (default: 10)
            filter_expression: Optional filter (not used)
            scoring_profile: Optional scoring profile (not used)
            use_semantic: Whether to use semantic search (not used)
            
        Returns:
            List of incidents based on vector similarity
        """
        logger.info("📚 Searching documents (all incident resolutions)")
        
        # Search for incidents with good resolutions
        return self.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=top_k,
            min_confidence=None,  # Don't filter on synthesis confidence
            data_quality=None  # Don't filter on data quality - focus on vector similarity
        )

    def text_search(self, query_text: str, top_k: int = 10,
                   filter_expression: str = None, 
                   scoring_profile: str = None,
                   order_by: str = None) -> List[Dict[str, Any]]:
        """
        Perform traditional text search without vector component.
        
        Uses full-text search capabilities for keyword-based matching.
        Supports wildcard queries ("*") for listing all documents.
        
        Args:
            query_text: Search query text (use "*" for all documents)
            top_k: Maximum number of results to return (default: 10)
            filter_expression: Optional OData filter expression
            scoring_profile: Optional Azure Search scoring profile name
            order_by: Optional field(s) to order results by
            
        Returns:
            List of formatted search results matching the text query
            
        Raises:
            Exception: If search operation fails
        """
        try:
            search_params = {
                "search_text": query_text if query_text != "*" else "",
                "top": top_k,
                "include_total_count": True,
                "query_type": "simple" if query_text == "*" else "full",
                "select": [
                    "id", "incident_id", "clean_summary", "clean_description",
                    "itil_resolution", "ticket_type", "product", "issue_category",
                    "priority", "confidence_score", "data_quality", "resolution_date",
                    "source_record"
                ]
            }
            
            if filter_expression:
                search_params["filter"] = filter_expression
            
            if order_by:
                search_params["order_by"] = order_by
            
            results = self.client.search(**search_params)
            
            formatted_results = []
            for result in results:
                formatted_result = self._convert_to_v2_format(result)
                formatted_results.append(formatted_result)
            
            logger.info(f"Text search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in text_search: {str(e)}", exc_info=True)
            raise

    def _convert_to_v2_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert search result to consistent Vector v2 format.
        
        Normalizes search results from different query types into a standard format,
        handles resolution date formatting, and calculates document age.
        
        Args:
            result: Raw search result from Azure Search
            
        Returns:
            Formatted dictionary with consistent field structure and additional
            computed fields (days_ago, resolution_date_formatted)
        """
        # Handle resolution date formatting
        resolution_date = result.get("resolution_date")
        days_ago = None
        resolution_date_formatted = "Unknown"
        
        if resolution_date:
            try:
                date_obj = datetime.fromisoformat(resolution_date.replace('Z', '+00:00'))
                days_ago = (datetime.now(timezone.utc) - date_obj).days
                resolution_date_formatted = date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass
        
        return {
            "id": result.get("id"),
            "@search.score": result.get("@search.score"),
            "incident_id": result.get("incident_id"),
            "clean_summary": result.get("clean_summary"),
            "clean_description": result.get("clean_description"),
            "itil_resolution": result.get("itil_resolution"),
            "ticket_type": result.get("ticket_type"),  # Use new field
            "product": result.get("product"),
            "issue_category": result.get("issue_category"),
            "priority": result.get("priority"),
            "confidence_score": result.get("confidence_score"),
            "data_quality": result.get("data_quality"),
            "resolution_date": resolution_date,
            "resolution_date_formatted": resolution_date_formatted,
            "days_ago": days_ago,
            # REMOVED: Legacy field mappings that were causing confusion
            # These mappings made prompts think they were still using old schema
            "content": f"{result.get('clean_summary', '')} {result.get('clean_description', '')} {result.get('itil_resolution', '')}".strip(),
            "tags": [],  # Not available in v2
            "record_order": 1,  # Single document per incident
            "@search.highlights": result.get("@search.highlights", {}),
            "@search.captions": result.get("@search.captions", [])
        }

    def enhanced_hybrid_search(self, query_text: str, query_embedding: List[float], 
                              top_k: int = 10, use_semantic: bool = True,
                              filter_expression: str = None) -> List[Dict[str, Any]]:
        """
        Alias for hybrid_search to maintain backwards compatibility.
        
        Args:
            query_text: Text query for keyword matching
            query_embedding: Vector embedding for semantic search
            top_k: Maximum number of results to return (default: 10)
            use_semantic: Whether to use semantic search (default: True)
            filter_expression: Optional OData filter expression
            
        Returns:
            List of ranked search results (delegates to hybrid_search)
        """
        return self.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            use_semantic=use_semantic,
            filter_expression=filter_expression
        )

    def vector_search(self, query_embedding: List[float], top_k: int = 10,
                     filter_expression: str = None) -> List[Dict[str, Any]]:
        """
        Pure vector search without temporal filtering.
        
        Performs semantic similarity search using only the vector embedding.
        No temporal, quality, or product filters applied.
        
        Args:
            query_embedding: Vector embedding of the query text
            top_k: Maximum number of results to return (default: 10)
            filter_expression: Optional filter (not used, delegates to unfiltered search)
            
        Returns:
            List of semantically similar results ranked by vector similarity
        """
        return self.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=top_k,
            days_back=None  # No temporal filter for pure vector search
        )

    def format_search_results(self, search_results: List[Dict[str, Any]], 
                             query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Format search results for display with additional metadata.
        
        Adds display flags and context to search results for UI rendering.
        Simplified for Vector v2 schema (all records are incidents).
        
        Args:
            search_results: Raw search results from any search method
            query_text: Original query text (not currently used)
            max_results: Maximum number of results to format (default: 5)
            
        Returns:
            List of formatted results with has_resolution, display_text flags
        """
        formatted_results = []
        
        for result in search_results[:max_results]:
            # All results are incidents with resolutions in v2
            result['has_resolution'] = bool(result.get('itil_resolution'))
            result['is_known_problem'] = False  # No separate problem records in v2
            
            # Add display context
            if result['has_resolution']:
                result['display_text'] = result.get('itil_resolution', '')
            else:
                result['display_text'] = f"""
                Incident: {result.get('clean_summary', '')}
                Description: {result.get('clean_description', '')}
                Status: No resolution available
                """
                
            formatted_results.append(result)
        
        logger.info(f"Formatted {len(formatted_results)} v2 incident results")
        return formatted_results

    def get_recent_incidents(self, days_back: int = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent incidents for trending analysis.
        
        Retrieves incidents from a recent time window,
        ordered by resolution date (newest first).
        
        Args:
            days_back: Number of days to look back (default: from constants)
            top_k: Maximum number of results to return (default: 10)
            
        Returns:
            List of recent incidents ordered by date descending
        """
        if days_back is None:
            days_back = constants.RECENT_INCIDENTS_DAYS
            
        logger.info(f"📈 Getting recent incidents from last {days_back} days")
        
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back))
        filter_expr = f"resolution_date ge {cutoff_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        
        return self.text_search(
            query_text="*",
            top_k=top_k,
            filter_expression=filter_expr,
            order_by="resolution_date desc"
        )

    def get_incident_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector index.
        
        Retrieves:
        - Total document count
        - Recent incidents count (last 30 days default)
        - Product distribution (top 10 products)
        - Index metadata
        
        Returns:
            Dictionary with statistical information and facet distributions
        """
        try:
            # Total count
            total_results = self.client.search("*", include_total_count=True, top=0)
            total_count = total_results.get_count()
            
            # Recent incidents
            recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=constants.RECENT_INCIDENTS_DAYS))
            recent_filter = f"resolution_date ge {recent_cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            recent_results = self.client.search(
                "*", filter=recent_filter, include_total_count=True, top=0
            )
            recent_count = recent_results.get_count()
            
            # Get facets for product distribution
            facet_results = self.client.search(
                "*", 
                facets=["product", "priority"],
                top=0
            )
            
            facets = facet_results.get_facets()
            
            return {
                "total_incidents": total_count,
                f"recent_incidents_{constants.RECENT_INCIDENTS_DAYS}_days": recent_count,
                "product_distribution": {
                    facet['value']: facet['count'] 
                    for facet in facets.get("product", [])[:10]  # Top 10
                },
                "index_name": self.index_name,
                "index_type": "Vector v2 - Full Incidents with Temporal Filtering"
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}