"""
SearchService for Vector Index v2 - Full Incident Search with Temporal Filtering
"""
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import math
from config import constants

logger = logging.getLogger(__name__)


class SearchService:
    """Service for handling Azure Cognitive Search operations with Vector Index v2"""
    
    def __init__(self, endpoint: str, api_key: str, index_name: str):
        """Initialize search service with Azure credentials"""
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
        """Test connection to Azure Search"""
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
        Perform vector search with temporal and quality filtering
        """
        try:
            logger.info(f"🔍 Vector search with temporal filtering")
            
            # Build filter expression
            filter_parts = []
            
            # Temporal filter - recent incidents first
            if days_back:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back))
                filter_parts.append(f"resolution_date ge {cutoff_date.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                logger.info(f"  📅 Filtering to last {days_back} days (since {cutoff_date.strftime('%Y-%m-%d')})")
            
            # Quality filter
            if min_confidence:
                filter_parts.append(f"confidence_score ge {min_confidence}")
                logger.info(f"  ⭐ Filtering to confidence >= {min_confidence}")
            
            if data_quality:
                filter_parts.append(f"data_quality eq '{data_quality}'")
                logger.info(f"  🥇 Filtering to '{data_quality}' quality incidents")
            
            if product_filter:
                filter_parts.append(f"product eq '{product_filter}'")
                logger.info(f"  🏷️ Filtering to product: {product_filter}")
            
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
                logger.info(f"  🔧 Applied filter: {filter_expression}")
            
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
                logger.info(f"  Top result: {top_result['incident_id']} (Azure score: {top_result['@search.score']:.3f}, Doc confidence: {top_result.get('confidence_score', 'N/A')}, Quality: {top_result['data_quality']}, Age: {top_result.get('days_ago', 'Unknown')} days)")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in vector_search_with_temporal_filter: {str(e)}", exc_info=True)
            raise

    def hybrid_search(self, query_text: str, query_embedding: List[float], 
                     top_k: int = 10, use_semantic: bool = True,
                     filter_expression: str = None, 
                     scoring_profile: str = None) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search combining vector + text with temporal intelligence
        """
        try:
            logger.info("🚀 Starting Vector v2 hybrid search")
            
            # Step 1: Vector search
            vector_results = self.vector_search_with_temporal_filter(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more candidates
                days_back=None,  # No temporal filtering
                min_confidence=constants.MIN_SEMANTIC_CONFIDENCE  # Only high-confidence incidents
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
                
                # Quality boost
                quality_boost = 1.0
                if result.get("data_quality") == "gold":
                    quality_boost = 1.3
                elif result.get("data_quality") == "silver":
                    quality_boost = 1.1
                
                combined_score = vector_score * quality_boost
                
                result_map[doc_id] = {
                    **result,
                    "vector_score": vector_score,
                    "quality_boost": quality_boost,
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
                    # Combine scores with vector preference
                    combined_score = (existing["vector_score"] * 0.7) + (text_score * 0.3)
                    combined_score *= existing["quality_boost"]
                    
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
                          f"Quality: {result['data_quality']}, "
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
        Search for documents - now searches all incidents as KB functionality is integrated
        """
        logger.info("📚 Searching documents (all incident resolutions)")
        
        # Search for high-quality incidents with good resolutions
        return self.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=top_k,
            min_confidence=constants.MIN_REFERENCE_CONFIDENCE,  # High-confidence incidents only
            data_quality=constants.QUALITY_GOLD  # Gold quality incidents only
        )

    def text_search(self, query_text: str, top_k: int = 10,
                   filter_expression: str = None, 
                   scoring_profile: str = None,
                   order_by: str = None) -> List[Dict[str, Any]]:
        """
        Perform traditional text search without vector component
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
        Convert search result to consistent format - updated for Vector v2 schema
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
        Alias for hybrid_search to maintain compatibility
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
        Pure vector search - delegates to temporal filter method
        """
        return self.vector_search_with_temporal_filter(
            query_embedding=query_embedding,
            top_k=top_k,
            days_back=None  # No temporal filter for pure vector search
        )

    def format_search_results(self, search_results: List[Dict[str, Any]], 
                             query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Format search results - simplified for v2 (no special PROBLEM record handling)
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
        Get recent high-quality incidents for trending analysis
        """
        if days_back is None:
            days_back = constants.RECENT_INCIDENTS_DAYS
            
        logger.info(f"📈 Getting recent incidents from last {days_back} days")
        
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back))
        filter_expr = f"resolution_date ge {cutoff_date.strftime('%Y-%m-%dT%H:%M:%SZ')} and data_quality eq '{constants.QUALITY_GOLD}'"
        
        return self.text_search(
            query_text="*",
            top_k=top_k,
            filter_expression=filter_expr,
            order_by="resolution_date desc"
        )

    def get_incident_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the vector index
        """
        try:
            # Total count
            total_results = self.client.search("*", include_total_count=True, top=0)
            total_count = total_results.get_count()
            
            # Recent incidents
            recent_filter = f"resolution_date ge {(datetime.now(timezone.utc) - timedelta(days=constants.RECENT_INCIDENTS_DAYS)).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            recent_results = self.client.search("*", filter=recent_filter, include_total_count=True, top=0)
            recent_count = recent_results.get_count()
            
            # Quality distribution
            quality_results = self.client.search(
                "*", 
                facets=["data_quality", "product", "priority"],
                top=0
            )
            
            facets = quality_results.get_facets()
            
            return {
                "total_incidents": total_count,
                f"recent_incidents_{constants.RECENT_INCIDENTS_DAYS}_days": recent_count,
                "quality_distribution": {
                    facet['value']: facet['count'] 
                    for facet in facets.get("data_quality", [])
                },
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