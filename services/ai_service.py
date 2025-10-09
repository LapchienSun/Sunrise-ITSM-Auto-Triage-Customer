"""
AI Service - Handles Azure OpenAI operations with optimized multi-step triage

This module provides the core AI functionality for the ITSM triage system, including:
- Azure OpenAI client initialization with proxy support
- Multi-step triage processing (type → product → issue → analysis → resolution)
- Embedding generation for semantic search
- Core issue extraction from noisy descriptions
- Hallucination prevention and validation
- Fallback handling for failed operations

Version: 4.1
"""
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import os
import time
import re
import httpx
from config import constants

logger = logging.getLogger(__name__)


class AIService:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        embedding_deployment: str,
        chat_deployment: str,
        api_version: str = "2024-10-21"
    ):
        """
        Initialize AI Service with Azure OpenAI client
        Uses httpx proxy configuration instead of manipulating environment variables
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            embedding_deployment: Name of the embedding model deployment
            chat_deployment: Name of the chat completion model deployment
            api_version: Azure OpenAI API version (default: "2024-10-21")
            
        Raises:
            Exception: If Azure OpenAI client initialization fails
        """
        try:
            # Configure proxy support via httpx if proxy settings exist
            http_client = None
            http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            
            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies["http://"] = http_proxy
                if https_proxy:
                    proxies["https://"] = https_proxy
                
                http_client = httpx.Client(proxies=proxies)
                logger.info(f"Configured httpx client with proxy support")
            
            # Initialize Azure OpenAI client with optional proxy support
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                http_client=http_client
            )
            logger.info("Successfully initialized AzureOpenAI client")

        except Exception as e:
            logger.error(f"Failed to initialize AzureOpenAI client: {e}")
            raise

        self.embedding_deployment = embedding_deployment
        self.chat_deployment = chat_deployment
        logger.info(
            f"AI Service initialised with deployments: embedding={embedding_deployment}, chat={chat_deployment}")


    def extract_core_issue(self, raw_description: str) -> str:
        """
        Extract just the core technical issue from noisy description.
        
        Removes email headers, signatures, HTML tags, and other noise while preserving
        the essential technical details, error messages, and user problem description.
        
        Args:
            raw_description: Raw user input that may contain noise
            
        Returns:
            Cleaned description containing only the core technical issue
            
        Note:
            Uses zero temperature for consistency. On failure, returns original text.
        """
        
        extraction_prompt = f"""You are extracting the core technical issue from a support request. Your task is ONLY to remove noise whilst preserving the exact user problem.

    CRITICAL RULES - DO NOT VIOLATE:
    1. NEVER add information not in the original text
    2. NEVER interpret, enhance, or explain the user's issue
    3. NEVER change technical terms, system names, or error messages
    4. NEVER add context or assumptions about what the user "probably means"
    5. If unclear what the core issue is, preserve more rather than less

    REMOVE (noise to strip):
    - Email headers, signatures, contact details
    - HTML tags, security warnings, legal disclaimers
    - Greetings ("Hi", "Thanks"), closings ("Regards")
    - Company boilerplate, privacy notices
    - URL defence mangling

    PRESERVE (essential content):
    - The exact user problem or question
    - All technical details, error messages, system names
    - Specific symptoms, behaviours, configurations mentioned
    - Any technical context needed to understand the issue

    EXTRACTION APPROACH:
    - Focus on what the user is asking for or reporting as broken
    - Keep their exact wording for technical descriptions
    - If multiple issues exist, extract the primary one that matches any provided summary

    Original text:
    {raw_description}

    Extract core issue (preserve exact technical language):"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{
                    "role": "user", 
                    "content": extraction_prompt
                }],
                temperature=constants.MIN_TEMPERATURE,  # Zero temperature for maximum consistency
                max_tokens=constants.MAX_EXTRACTION_TOKENS,
                timeout=constants.EXTRACTION_TIMEOUT_SECONDS
            )
            
            extracted = response.choices[0].message.content.strip()
            
            # Basic validation - ensure we didn't lose critical content
            if len(extracted) < 20 or len(extracted) > len(raw_description):
                logger.warning(f"Extraction may be invalid: {len(extracted)} chars from {len(raw_description)} original")
            
            logger.info(f"🔍 Extracted core issue: {extracted[:100]}...")
            return extracted
            
        except Exception as e:
            logger.error(f"Error in extract_core_issue: {str(e)}")
            return raw_description  # Fail safe - use original

    def _sanitise_for_llm(self, text: str) -> str:
        """
        Sanitise user input to prevent safety classifier false positives.
        
        Removes problematic line endings and neutralizes potential prompt injection
        patterns, particularly email signatures that may trigger false positives.
        
        Args:
            text: User input text to sanitise
            
        Returns:
            Sanitised text safe for LLM processing
        """
        if not text:
            return text

        # Replace problematic line endings that might trigger classifiers
        cleaned = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove or neutralise potential prompt injection patterns
        # Email signatures at end of text are common false positive triggers
        cleaned = re.sub(
            r'\n\n(Regards|Best regards|Thanks|Cheers)\n\n\w+\s*$',
            '\n\n[Email signature removed]',
            cleaned,
            flags=re.IGNORECASE
        )

        return cleaned

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for semantic search using Azure OpenAI.
        
        Args:
            text: Input text to generate embeddings for
            
        Returns:
            List of float values representing the text embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.info(f"\n🧠 GENERATING EMBEDDING")
            logger.info(f"Text Length: {len(text)} chars")
            logger.info(f"Text Preview: {text[:200]}...")

            start_time = datetime.now()
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            end_time = datetime.now()

            embedding = response.data[0].embedding
            logger.info(f"✅ Embedding Generated Successfully")
            logger.info(f"├─ Dimensions: {len(embedding)}")
            logger.info(
                f"├─ Generation Time: {(end_time - start_time).total_seconds():.2f}s")
            logger.info(
                f"└─ Non-zero values: {sum(1 for x in embedding if abs(x) > 0.001)}")

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_optimized_multistep_triage(
        self,
        summary: str,
        description: str,
        call_source: str,
        retrieved_records: List[Dict[str, Any]],
        config_manager: Any,
        temperature: float = 0.0,
        found_document_ids: set = None,
        key_insight_text: Optional[str] = None,
        key_insight_source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute optimized multi-step triage process using Azure OpenAI.
        
        This is the main triage orchestration method that executes 5 steps:
        1. Type Classification (Incident vs Service Request) + Environment
        2. Product Selection (from type-specific product list)
        3. Issue Selection + Priority Calculation + Clarifying Questions
        4. Root Cause Analysis + Scope Assessment + Initial Response
        5. Resolution Content Generation + HTML Formatting
        
        Args:
            summary: Brief incident summary
            description: Detailed incident description
            call_source: Source of request (Phone, Email, Portal, etc.)
            retrieved_records: List of similar records from semantic search
            config_manager: Configuration manager with taxonomy and rules
            temperature: LLM temperature (default: 0.0 for consistency)
            found_document_ids: Set of valid document IDs from search results
            key_insight_text: Optional key technical detail from top match
            key_insight_source_id: Optional ID of record containing key insight
            
        Returns:
            Complete triage result dictionary with classification, analysis,
            resolution, and token usage information
            
        Raises:
            Falls back to generate_triage_response on any exception
        """
        try:
            from config import prompts

            logger.info(f"Starting triage: {len(description)} chars, {len(retrieved_records)} similar records")
            overall_start = time.time()

            # Step 1: Type Classification
            step1_start = time.time()
            type_prompt = prompts.build_type_prompt(
                summary, description, call_source, retrieved_records[:3],
                config_manager.categories.type
            )
            
            type_response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an ITSM expert. Return only JSON."},
                    {"role": "user", "content": type_prompt}
                ],
                temperature=temperature,
                top_p=1,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"}
            )
            type_result = json.loads(type_response.choices[0].message.content)
            type_selected = type_result.get("type", "Incident")
            environment_selected = type_result.get("environment", "Live")
            step1_time = time.time() - step1_start
            
            logger.info(f"[Step 1] {type_selected}/{environment_selected} ({step1_time:.2f}s)")

            # Step 2: Product Selection
            step2_start = time.time()
            valid_products = config_manager.get_products_for_type(type_selected)

            # Normal AI-based product selection
            if not valid_products:
                logger.warning(f"WARNING: No filtered products for {type_selected}, using full list")
                valid_products = config_manager.categories.product
            
            product_prompt = prompts.build_product_prompt(
                summary, description, type_selected, retrieved_records[:5], valid_products
            )
            product_response = self.client.chat.completions.create(
                model=self.chat_deployment, 
                messages=[
                    {"role": "system", "content": "You are an ITSM expert. Return only JSON."}, 
                    {"role": "user", "content": product_prompt}
                ], 
                temperature=temperature, 
                top_p=1, 
                n=1, 
                frequency_penalty=0, 
                presence_penalty=0, 
                response_format={"type": "json_object"}
            )
            
            product_result = json.loads(product_response.choices[0].message.content)
            product_selected = product_result.get("product", "Other")
            
            # Validate product is in the allowed list for this type
            expected_products = config_manager.get_products_for_type(type_selected)
            if expected_products and product_selected not in expected_products:
                logger.warning(f"Auto-corrected product '{product_selected}' to valid option")
                product_selected = expected_products[0] if expected_products else "Other"
            
            step2_time = time.time() - step2_start
            logger.info(f"[Step 2] {product_selected} ({step2_time:.2f}s)")
    
            # Step 3: Issue, Priority & Questions
            step3_start = time.time()
            valid_issues = config_manager.get_issues_for_type_product(type_selected, product_selected)

            # Issue Precedence Rule
            issue_precedence_override = None
            if re.search(r"^(Does|Is)\b.*(compatible|work with)", description, re.IGNORECASE):
                if "Information" in valid_issues:
                    issue_precedence_override = "Information"
                    logger.info("Compatibility question detected, suggesting 'Information' issue")

            if not valid_issues:
                logger.warning(f"WARNING: No filtered issues for {type_selected}/{product_selected}")
                valid_issues = config_manager.categories.issue

            issue_prompt = prompts.build_issue_priority_prompt_only(
                summary, description, type_selected, product_selected, 
                retrieved_records, valid_issues, config_manager.priority_matrix, 
                environment_selected, issue_precedence_override
            )

            issue_response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an ITSM expert. Generate detailed output. Return only JSON."},
                    {"role": "user", "content": issue_prompt}
                ],
                temperature=temperature, top_p=1, n=1, frequency_penalty=0, presence_penalty=0,
                response_format={"type": "json_object"}
            )
            issue_result = json.loads(issue_response.choices[0].message.content)
            step3_time = time.time() - step3_start
            
            logger.info(f"[Step 3] {issue_result.get('issue')}, {issue_result.get('priority')} ({step3_time:.2f}s)")

            # Step 4: Analysis & Initial Response
            step4_start = time.time()
            
            if key_insight_source_id:
                logger.info(f"Using key insight from {key_insight_source_id}")
            
            sanitised_description = self._sanitise_for_llm(description)
            analysis_prompt = prompts.build_analysis_prompt(
                summary, sanitised_description, type_selected, product_selected,
                issue_result['issue'], issue_result['priority'],
                environment_selected, retrieved_records,
                key_insight_text=key_insight_text,
                key_insight_source_id=key_insight_source_id
            )
            analysis_response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an ITSM expert. Provide detailed analysis. Return only JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=temperature,
                top_p=1,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"}
            )

            analysis_result = json.loads(analysis_response.choices[0].message.content)
            analysis_result = self._sanitize_text_fields(analysis_result, found_document_ids)
            step4_time = time.time() - step4_start
            
            logger.info(f"[Step 4] Analysis complete ({step4_time:.2f}s)")

            # Step 5a: Resolution Content Generation
            step5a_start = time.time()

            # Check if we have a high-confidence match for resolution fidelity
            high_confidence_match = None
            for record in retrieved_records[:3]:
                score = record.get('@search.score', 0)
                if score > constants.HIGH_CONFIDENCE_THRESHOLD:
                    high_confidence_match = record
                    logger.info(f"High-confidence match: {record.get('incident_id')} (score: {score:.3f})")
                    break

            incident_details = {
                'type': type_selected,
                'product': product_selected,
                'issue': issue_result['issue'],
                'priority': issue_result['priority'],
                'urgency': issue_result['urgency'],
                'impact': issue_result['impact'],
                'root_cause_preliminary': analysis_result['root_cause_preliminary'],
                'scope_of_impact': analysis_result['scope_of_impact'],
                'initial_response': analysis_result['initial_response'],
                'high_confidence_match': high_confidence_match
            }

            resolution_content_prompt = prompts.build_resolution_content_prompt(
                incident_details, retrieved_records[:5]
            )

            resolution_content_response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate resolution content. Be concise but comprehensive. Return only JSON."
                    },
                    {"role": "user", "content": resolution_content_prompt}
                ],
                temperature=temperature,
                top_p=1,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                max_tokens=constants.MAX_RESOLUTION_TOKENS,
                response_format={"type": "json_object"}
            )

            resolution_content = json.loads(resolution_content_response.choices[0].message.content)
            resolution_content = self._sanitize_text_fields(resolution_content, found_document_ids)
            step5a_time = time.time() - step5a_start
            
            logger.info(f"[Step 5a] {len(resolution_content.get('implementation_steps', []))} steps generated ({step5a_time:.2f}s)")

            # Step 5b: Format HTML
            step5b_start = time.time()
            incident_details['clarifying_questions'] = issue_result.get('clarifying_questions', [])
            resolution_formatted = prompts.format_resolution_html(incident_details, resolution_content)
            step5b_time = time.time() - step5b_start

            # Calculate total tokens - handle override case where product_response doesn't exist
            product_tokens = product_response.usage.total_tokens if 'product_response' in locals() else 0

            total_tokens = type_response.usage.total_tokens + product_tokens + \
                issue_response.usage.total_tokens + \
                analysis_response.usage.total_tokens + \
                resolution_content_response.usage.total_tokens

            # Map scope_of_impact_category to proper ITSM impact values
            scope_category = analysis_result.get('scope_of_impact_category', 'Single User')

            if scope_category == 'Single User':
                impact_value = 'Individual'
            elif scope_category in ['Multiple Users', 'Department-Wide']:
                impact_value = 'Group'
            elif scope_category in ['Organisation-Wide', 'Enterprise', 'Site']:
                impact_value = 'Site'
            else:
                # Fallback based on keywords
                if 'user' in scope_category.lower() and 'single' in scope_category.lower():
                    impact_value = 'Individual'
                elif 'department' in scope_category.lower() or 'group' in scope_category.lower():
                    impact_value = 'Group'
                elif 'organisation' in scope_category.lower() or 'enterprise' in scope_category.lower():
                    impact_value = 'Site'
                else:
                    impact_value = 'Individual'  # Final fallback

            logger.info(f"ITSM Impact mapping: '{scope_category}' -> '{impact_value}'")

            complete_result = {
                "type": type_selected,
                "product": product_selected,
                "issue": issue_result.get('issue', 'Other'),
                "urgency": issue_result.get('urgency', 'Medium'),
                "impact": impact_value,
                "priority": issue_result.get('priority', '3. Medium'),
                "environment": environment_selected,
                "escalation_level": "Normal",
                "scope_of_impact": analysis_result.get('scope_of_impact'),
                "scope_of_impact_category": analysis_result.get('scope_of_impact_category'),
                "initial_response": analysis_result.get('initial_response'),
                "summary_rating": analysis_result.get('summary_rating', 'not_provided'),
                "description_rating": analysis_result.get('description_rating', 'good'),
                "alternative_description": analysis_result.get('alternative_description', ''),
                "suggested_resolution": resolution_formatted['suggested_resolution'],
                "source": call_source,
                "token_usage": {
                    "step1_type": type_response.usage.total_tokens,
                    "step2_product": product_tokens,
                    "step3_issue_priority": issue_response.usage.total_tokens,
                    "step4_analysis": analysis_response.usage.total_tokens,
                    "step5_resolution": resolution_content_response.usage.total_tokens,
                    "total_tokens": total_tokens,
                    "input_tokens": int(total_tokens * 0.75),
                    "output_tokens": int(total_tokens * 0.25)
                }
            }

            # Apply post-generation validation
            complete_result = self._validate_and_fix_response(
                complete_result,
                description,
                list(found_document_ids) if found_document_ids else []
            )

            if found_document_ids:
                full_ai_text = " ".join(filter(None, [
                    complete_result.get('root_cause_preliminary'),
                    complete_result.get('initial_response'),
                    complete_result.get('suggested_resolution')
                ]))

                id_pattern = re.compile(r'\b((?:PRB|SUN|KB)\d+)\b')
                mentioned_ids = set(id_pattern.findall(full_ai_text))

                validated_and_formatted_docs = []
                for doc_id in mentioned_ids:
                    if doc_id in found_document_ids:
                        if doc_id.startswith('PRB'):
                            validated_and_formatted_docs.append(
                                f"Problem {doc_id}")
                        elif doc_id.startswith('SUN'):
                            validated_and_formatted_docs.append(
                                f"Incident {doc_id}")
                        elif doc_id.startswith('KB'):
                            validated_and_formatted_docs.append(
                                f"Knowledgebase {doc_id}")
                            
            # Fix Related Records to always use highest-scoring document
            if found_document_ids and key_insight_source_id:
                # Get the actual ticket type from the retrieved records
                primary_reference = f"Record {key_insight_source_id}"  # Default fallback
                
                for record in retrieved_records[:3]:
                    if record.get('incident_id') == key_insight_source_id:
                        ticket_type = record.get('ticket_type', '')
                        record_id = record.get('incident_id', key_insight_source_id)
                        
                        if key_insight_source_id.startswith('PRB'):
                            primary_reference = f"Problem {record_id}"
                        elif key_insight_source_id.startswith('KB'):
                            primary_reference = f"Knowledgebase {record_id}"
                        elif key_insight_source_id.startswith('SUN'):
                            primary_reference = f"{ticket_type} {record_id}" if ticket_type else f"Incident {record_id}"
                        break

                # Replace any existing Related Records with the top match
                complete_result['suggested_resolution'] = re.sub(
                    r'<p><b>Related Records:</b>.*?</p>',
                    f'<p><b>Related Records:</b> {primary_reference}</p>',
                    complete_result['suggested_resolution'],
                    flags=re.DOTALL
                )
                
                # If no Related Records section exists, add it before the closing
                if '<b>Related Records:</b>' not in complete_result['suggested_resolution']:
                    complete_result['suggested_resolution'] = complete_result['suggested_resolution'].rstrip()
                    complete_result['suggested_resolution'] += f"<p><b>Related Records:</b> {primary_reference}</p>"
                    
            total_time = time.time() - overall_start
            
            logger.info(f"[COMPLETE] Triage: {type_selected}/{product_selected}/{issue_result.get('issue')} - {issue_result.get('priority')} ({total_time:.2f}s, {total_tokens:,} tokens)")

            return complete_result

        except Exception as e:
            logger.error(
                f"Error in optimised multi-step triage: {str(e)}", exc_info=True)
            logger.warning("Falling back to standard triage method")
            return self.generate_triage_response(summary, description, call_source, retrieved_records, config_manager, temperature)

    def _sanitize_text_fields(self, content_dict: dict, found_document_ids: set) -> dict:
        """
        Remove hallucinated document references from AI-generated content.
        
        Scans text fields for document ID patterns (PRB, SUN, KB) and removes
        any references to IDs that were not in the original search results.
        
        Args:
            content_dict: Dictionary containing AI-generated text fields
            found_document_ids: Set of valid document IDs from search results
            
        Returns:
            Sanitized dictionary with hallucinated references removed
        """
        if not found_document_ids:
            return content_dict

        # Updated pattern to include KB records
        id_pattern = re.compile(r'\b((?:PRB|SUN|KB)\d+)\b')

        fields_to_sanitize = [
            "root_cause_preliminary", "initial_response", "alternative_description",
            "issue_summary", "resolution_summary", "suggested_resolution", "implementation_plan"
        ]

        for field in fields_to_sanitize:
            if field in content_dict and isinstance(content_dict[field], str):
                text = content_dict[field]
                potential_ids = id_pattern.findall(text)
                hallucinated_ids = [
                    pid for pid in potential_ids if pid not in found_document_ids]

                if not hallucinated_ids:
                    continue

                logger.warning(
                    f"Hallucination Detected in text field '{field}'! "
                    f"AI referenced {list(set(hallucinated_ids))}, which were not in search results. Sanitising text."
                )

                for hallucinated_id in set(hallucinated_ids):
                    phrases_to_remove = [
                        f"referencing Knowledge Article {hallucinated_id}", 
                        f"referencing KB {hallucinated_id}",
                        f"as documented in Knowledge Article {hallucinated_id}",
                        f"according to Knowledge Article {hallucinated_id}",
                        f"Knowledge Article {hallucinated_id}",
                        f"KB {hallucinated_id}",
                        f"referencing Known Problem {hallucinated_id}", 
                        f"referencing Problem {hallucinated_id}",
                        f"referencing {hallucinated_id}", 
                        f"aligns with Known Problem {hallucinated_id}",
                        f"matches the known problem documented in {hallucinated_id}",
                        f"as documented in {hallucinated_id}", 
                        f"documented in {hallucinated_id}",
                        f"Problem {hallucinated_id}", 
                        f"Known Problem {hallucinated_id}",
                        hallucinated_id
                    ]
                    for phrase in phrases_to_remove:
                        text = re.sub(re.escape(phrase), '',
                                    text, flags=re.IGNORECASE)

                text = re.sub(r'\s{2,}', ' ', text)
                text = re.sub(r'\s+\.', '.', text)
                text = text.strip()

                content_dict[field] = text

        return content_dict

    def _validate_and_fix_response(self, response_dict, description, referenced_documents):
        """
        Post-generation validator to catch common AI reasoning errors.
        
        Applies guardrails to fix known AI issues such as:
        - Phantom media provisioning claims
        - Invalid document references
        - Issue misclassifications
        
        Args:
            response_dict: AI-generated response dictionary
            description: Original incident description
            referenced_documents: List of valid document IDs from search
            
        Returns:
            Validated and corrected response dictionary
        """
        fixes_applied = []

        # Guardrail 1: Issue misclassification for compatibility questions (already handled by precedence rule)

        # Guardrail 2: Remove phantom media provisioning claims
        phantom_provision_patterns = [
            r"provide.*installation media",
            r"supplied.*installation media",
            r"attached.*installation media",
            r"give.*installation media",
            r"provide.*Java.*version",
            r"provide.*with.*Java.*11\.0\.23",
            r"obtaining.*Java.*version",
            r"share installer",
            r"provide guidance on obtaining"
        ]

        fields_to_check = ["implementation_plan",
                        "initial_response", "suggested_resolution"]

        for field in fields_to_check:
            if field in response_dict and response_dict[field]:
                original_text = response_dict[field]
                fixed_text = original_text

                for pattern in phantom_provision_patterns:
                    if re.search(pattern, fixed_text, re.IGNORECASE):
                        # Replace with conditional offering
                        fixed_text = re.sub(
                            pattern,
                            "can provide installation media on request",
                            fixed_text,
                            flags=re.IGNORECASE
                        )
                        fixes_applied.append(
                            f"Fixed phantom provisioning in {field}")

                response_dict[field] = fixed_text

        # Guardrail 3: Enhanced phantom reference check for all record types including KB records
        all_pattern = r"(SUN\d{6}|PRB\d{6}|KB\d{6})"
        all_text = " ".join([str(v) for v in response_dict.values() if isinstance(v, str)])
        found_refs = set(re.findall(all_pattern, all_text))
        valid_refs = set(referenced_documents) if referenced_documents else set()

        phantom_refs = found_refs - valid_refs
        if phantom_refs:
            fixes_applied.append(f"Warning: Phantom references found: {phantom_refs}")
            
            # Categorise phantom references by type
            phantom_kb = [ref for ref in phantom_refs if ref.startswith('KB')]
            phantom_sun = [ref for ref in phantom_refs if ref.startswith('SUN')]
            phantom_prb = [ref for ref in phantom_refs if ref.startswith('PRB')]
            
            if phantom_kb:
                logger.warning(f"Phantom Knowledge Article references: {phantom_kb}")
            if phantom_sun:
                logger.warning(f"Phantom Incident references: {phantom_sun}")  
            if phantom_prb:
                logger.warning(f"Phantom Problem references: {phantom_prb}")

        if fixes_applied:
            logger.info(f"Post-generation fixes applied: {'; '.join(fixes_applied)}")

        return response_dict

    def generate_triage_response(self, summary: str, description: str, call_source: str,
                                 retrieved_records: List[Dict[str, Any]], config_manager: Any,
                                 temperature: float = 0.2) -> Dict[str, Any]:
        """
        Fallback triage response when optimised method fails.
        
        Provides a basic, safe categorization as a Service Request with
        low priority to ensure the system remains operational even when
        the main triage method encounters errors.
        
        Args:
            summary: Brief incident summary
            description: Detailed incident description
            call_source: Source of the request
            retrieved_records: List of similar records (not used in fallback)
            config_manager: Configuration manager (not used in fallback)
            temperature: LLM temperature (not used in fallback)
            
        Returns:
            Basic triage response dictionary with minimal token usage
        """
        logger.info("🔄 Using fallback triage response")

        # Basic categorisation - generic fallback for IT support
        basic_response = {
            "type": "Service Request",
            "product": "Information",
            "issue": "Information Request",
            "urgency": "Low",
            "impact": "Individual",
            "priority": "4. Low",
            "environment": "Live",
            "escalation_level": "Normal",
            "scope_of_impact": "Single user information request",
            "scope_of_impact_category": "Single User",
            "initial_response": "Provide information and guidance to address the user's request",
            "summary_rating": "good",
            "description_rating": "good",
            "alternative_description": "",
            "suggested_resolution": "<p><b>Initial Response:</b> Provide information and guidance to address the user's request</p>",
            "source": call_source,
            "token_usage": {
                "total_tokens": 100,
                "input_tokens": 75,
                "output_tokens": 25,
                "step1_type": 25,
                "step2_product": 25,
                "step3_issue_priority": 25,
                "step4_analysis": 25,
                "step5_resolution": 0
            }
        }

        return basic_response
