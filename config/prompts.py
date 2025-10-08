import logging
from typing import List, Dict, Optional
import re
from config import constants

"""
ITSM AI Triage Tool - Optimized Prompts Module
VERSION 4.1 - Enhanced Summary Separation
- Only uses user-provided data (no vector search results)
- Maintains all existing functionality in optimized Steps 1-5
"""

SYSTEM_PROMPT = """You are an expert IT service desk analyst with deep knowledge of ITSM best practices, applying ITIL v4 principles for incident management. All free-text must use UK British English spelling, grammar, and terminology.

Provide comprehensive, actionable guidance for IT service desk analysts handling a wide range of IT support issues including hardware, software, network, communications, and account administration requests."""

# Step 1: Type Classification (Fast - ~1s)
def build_type_prompt(summary: str, description: str, call_source: str,
                      retrieved_records: list, type_options: list) -> str:
    """
    Build prompt for determining Type and Environment
    """
    record_hints = ""
    for i, record in enumerate(retrieved_records[:3], 1):
        record_summary = record.get('summary', '')[:100]
        record_product = record.get('product', '')
        record_issue = record.get('issue', '')
        record_hints += f"Record {i}: {record_summary}\n"
        if record.get('source_record'):
            record_hints += f"  Type: {record.get('source_record')}"
        if record_product and record_issue:
            record_hints += f" | Classification: {record_product} -> {record_issue}\n"
        else:
            record_hints += "\n"

    return f"""Classify this request:

REQUEST: {description}
SOURCE: {call_source}

SIMILAR RECORDS:
{record_hints}

DETERMINE:
1. TYPE:
   - Incident: Something is broken, not working, experiencing errors, or service degradation
   - Service Request: "How to" questions, requesting changes, seeking guidance, configuration requests, bringing systems online

2. ENVIRONMENT:
   - Test: Any mention of "test", "testing", "test environment", "UAT", "non-production"
   - Live: Production, live system, or no mention of test/testing

Key phrases for Service Request: "how to", "can I", "is it possible", "configure", "setup", "add", "remove", "guidance", "reduce", "arrange", "bring online", "activate", "verify functionality"
Key phrases for Incident: "can't", "unable", "error", "not working", "broken", "down", "failed"

Return ONLY:
{{
  "type": "Incident" or "Service Request",
  "environment": "Test" or "Live",
  "confidence": "high/medium/low"
}}"""


# Step 2: Product Selection (Fast - ~1s)
def build_product_prompt(summary: str, description: str, type_selected: str,
                         retrieved_records: list, product_options: list) -> str:
    """
    Build prompt for determining Product
    """
    product_hints = []
    for record in retrieved_records[:5]:
        for tag in record.get('tags', []):
            if tag in product_options:
                product_hints.append(tag)

    if len(product_options) <= 10:
        product_list = f"""CONSTRAINT: For {type_selected}, you MUST select from ONLY these {len(product_options)} products:
{chr(10).join([f"- {product}" for product in product_options])}"""
    else:
        product_list = f"""CONSTRAINT: For {type_selected}, select ONLY from: {', '.join(product_options)}"""

    return f"""Type selected: {type_selected}

DESCRIPTION: {description}

DETECTED PRODUCTS FROM SIMILAR INCIDENTS: {', '.join(set(product_hints)) if product_hints else 'None detected'}

PRODUCT SELECTION GUIDELINES:
- Match the product category to the PRIMARY issue described
- Use exact wording from the available products list
- Consider the type of technology or service affected
- Review similar incidents for guidance on common classifications

Common product selection patterns:
- Email/messaging issues → Communications
- Login/password/account issues → Account Administration  
- Desktop/laptop hardware → Hardware
- Network connectivity/access → Network
- Application software → Software
- Printers/peripherals → Hardware
- Server infrastructure → Hardware (Server)
- Database issues → Software (Database)
- Office applications → Software (Microsoft Office)

{product_list}

CRITICAL: You MUST select a product from the list above. Do not select any product not listed.

Return ONLY:
{{
  "product": "selected product exactly as shown in the list"
}}"""


# Step 3: Issue, Priority & Questions ONLY (no enhanced summary)
def build_issue_priority_prompt_only(summary: str, description: str, type_selected: str,
                                product_selected: str, retrieved_records: list,
                                issue_options: list, priority_matrix, environment_selected: str = "Live",
                                issue_precedence_override: str = None) -> str:
    """
    Issue selection, Priority calculation, and Context-Aware Clarifying Questions
    """
    # Include resolution hints from similar incidents
    similar_issues = []
    for record in retrieved_records[:5]:
        if record.get('issue'):
            similar_issues.append(record.get('issue'))

    # Build comprehensive context from similar incidents
    similar_context = ""
    if retrieved_records:
        similar_context = "\n\nSIMILAR INCIDENTS FOR CONTEXT:\n"
        for i, record in enumerate(retrieved_records[:3], 1):
            similar_context += f"Record {i}: {record.get('clean_summary', record.get('summary', ''))}\n"
            if record.get('itil_resolution', record.get('resolution_notes')):
                resolution = record.get('itil_resolution', record.get('resolution_notes', ''))[:200]
                similar_context += f"  Resolution: {resolution}...\n"

    # Priority rules (keep existing logic)
    priority_rules_text = "\n".join([
        f"- {rule.urgency} urgency + {rule.impact} impact = {rule.priority}"
        for rule in priority_matrix.rules
    ])

    # Enhanced ITIL lessons for priority (v3.3 improvements)
    priority_lessons = """
PRIORITY ASSESSMENT LESSONS:
## Lesson 1: High-urgency incidents
- A single user being unable to perform their main work tasks is High Urgency, Low Impact -> Priority 3. Medium

## Lesson 2: Medium urgency issues
- A single user with an intermittent issue where work can continue with some impairment is Medium Urgency, Low Impact -> Priority 4. Low

## Lesson 3: Service Requests
- "How-to" questions or requests for future configuration are Low Urgency, Low Impact -> Priority 4. Low

## Lesson 4: Team-wide non-critical impacts
- A non-critical feature (like a printer or report) not working for a team is Low Urgency, Medium Impact -> Priority 4. Low
"""

    # Issue Precedence Rule injection
    precedence_instruction = ""
    if issue_precedence_override:
        precedence_instruction = f"""
CRITICAL PRECEDENCE RULE: This request matches a compatibility question pattern. 
You MUST select "{issue_precedence_override}" unless the evidence clearly shows 
the technology is definitively incompatible. Do not select "Installation Media" 
for compatibility verification requests.

"""

    return f"""Complete triage for {type_selected} - {product_selected}

INCIDENT DETAILS:
SUMMARY: {summary or 'Not provided'}
DESCRIPTION: {description}

AVAILABLE ISSUES: {', '.join(issue_options)}

{precedence_instruction}SIMILAR INCIDENTS HAD THESE ISSUES: {', '.join(similar_issues[:5]) if similar_issues else 'None found'}
{similar_context}

TASK 1 - SELECT ISSUE:
{precedence_instruction}Choose the most appropriate issue category from the available options based on the incident description and similar incidents.

TASK 2 - DETERMINE PRIORITY:
Assess urgency and impact using ITIL v4 principles:

URGENCY LEVELS:
- High: Resolution needed immediately - critical business functions are stopped or severely degraded
- Medium: Resolution needed soon - business functions are impaired but workarounds exist  
- Low: Resolution can be scheduled - minor inconvenience with minimal business impact

IMPACT LEVELS:
- High: Widespread disruption - affects multiple departments, business units, or critical services
- Medium: Significant disruption - affects multiple users or a single department/team
- Low: Minimal disruption - affects only a single user or non-critical function

{priority_lessons}

PRIORITY MATRIX:
{priority_rules_text}

TASK 3 - GENERATE CONTEXT-AWARE CLARIFYING QUESTIONS:
Analyse the specific incident described above and generate 3-5 highly relevant clarifying questions.

QUESTION GENERATION PRINCIPLES:
1. **Be Incident-Specific**: Focus on THIS exact problem, not generic product questions
2. **Fill Information Gaps**: Ask only about details missing from the description that are needed for resolution
3. **Resolution-Focused**: Questions should help determine the best resolution approach
4. **Avoid Redundancy**: Don't ask for information already provided in the description
5. **Technical Relevance**: For technical issues, ask about specific technical details that matter
6. **Business Context**: For service requests, understand the business need and current state

INTELLIGENT QUESTION LOGIC:
- If this is about **compatibility/versions**: Ask about current versions, target versions, specific requirements, deployment environment
- If this is about **configuration/setup**: Ask about current state, desired outcome, business requirements, constraints
- If this is about **errors/failures**: Ask about specific error messages, when it started, what changed, scope of impact  
- If this is about **performance**: Ask about specific symptoms, timing, frequency, what operations are affected
- If this is about **access/permissions**: Ask about user roles, what they're trying to access, error messages
- If this is about **how-to/guidance**: Ask about current process, desired outcome, business context

Generate questions that a skilled analyst would ask to provide the best possible resolution for this specific situation.

Return ONLY this JSON structure:
{{
  "issue": "selected issue from available options",
  "urgency": "High/Medium/Low", 
  "impact": "High/Medium/Low",
  "priority": "1. Critical/2. High/3. Medium/4. Low",
  "clarifying_questions": [
    "Specific question 1 relevant to this exact incident?",
    "Specific question 2 that would help determine resolution approach?", 
    "Specific question 3 about missing technical/business details?",
    "Specific question 4 that addresses gaps in the description?",
    "Specific question 5 that would improve resolution quality?"
  ]
}}

IMPORTANT: Ensure ALL clarifying questions are directly relevant to the specific incident described: "{description}"
Do not generate generic questions that could apply to any incident with this product."""


# NEW Step 3.5: Enhanced Summary Generation (fast, no vector data)
def build_enhanced_summary_prompt(summary: str, description: str) -> str:
    """
    Generate enhanced summary using ONLY user-provided data
    NO vector search results or similar incidents used
    """
    return f"""Create an enhanced summary based solely on the provided information.

ORIGINAL SUMMARY: "{summary or 'Not provided'}"
DESCRIPTION: "{description}"

ENHANCED SUMMARY REQUIREMENTS:
1. **Focus on the user's actual request/problem** - ignore conversational details
2. **Extract the core issue only** - what does the user want or what's broken?
3. **Ignore process details** - remove mentions of what was explained, planned, or will happen
4. **Keep it factual and neutral** - state what the user wants, not what they were told
5. **Be concise** - aim for one clear sentence about the actual request

FILTERING RULES:
- INCLUDE: The user's goal, what they want to achieve, what's not working
- EXCLUDE: "Customer was told...", "I explained...", "Customer asked about...", "I will log a ticket..."
- EXCLUDE: Explanations of current system behaviour or limitations
- EXCLUDE: What was discussed, advised, or planned for the future

EXAMPLES:
- Input: "Customer called about login issues. I explained the password policy and will reset their account."
- Output: "User experiencing login issues"

- Input: "User wants to see all contact groups on records. I told them it's not possible and will investigate."
- Output: "User wants to display all contact groups associated with a record"

For this request, identify the core user need from the description and create a clear, neutral summary.

Return ONLY this JSON:
{{
  "enhanced_summary": "concise summary of the user's actual request or issue"
}}"""


# Step 4: Root Cause Analysis and Initial Response (Optimized)
def build_analysis_prompt(summary: str, description: str, type_selected: str,
                          product_selected: str, issue_selected: str,
                          priority: str, environment: str, retrieved_records: list,
                          key_insight_text: Optional[str] = None,
                          key_insight_source_id: Optional[str] = None) -> str:
    """
    Root cause analysis, impact assessment, and initial response
    Updated for Demo Vector v2 index - now includes INCIDENTS, SERVICE REQUESTS, PROBLEMS, and KNOWLEDGE records
    """
    resolution_hints = []

    # Process retrieved records - now includes INCIDENTS, SERVICE REQUESTS, PROBLEMS, and KNOWLEDGE
    for record in retrieved_records[:5]:
        source_record = record.get('source_record', 'INCIDENT')  # Default for backward compatibility
        ticket_type = record.get('ticket_type', '')
        record_id = record.get('incident_id', record.get('id', 'Unknown'))  # Same field for all records
        
        # Create appropriate label based on source record type
        if source_record == 'PROBLEM':
            record_label = f"Problem {record_id} ({ticket_type})" if ticket_type else f"Problem {record_id}"
        elif source_record == 'KNOWLEDGE':
            record_label = f"Knowledge Article {record_id} ({ticket_type})" if ticket_type else f"Knowledge Article {record_id}"
        else:
            record_label = f"{ticket_type} {record_id}" if ticket_type else f"Record {record_id}"

        # All records use the same resolution field
        resolution_field = record.get('itil_resolution') or record.get('resolution_notes')
        
        if resolution_field:
            # For knowledge articles, indicate they contain documented procedures
            if source_record == 'KNOWLEDGE':
                resolution_hints.append(f"- {record_label} (Documented Procedure): {resolution_field[:300]}")
            else:
                resolution_hints.append(f"- {record_label}: {resolution_field[:300]}")

    prompt_sections = []
    prompt_sections.append(
        f"Analyse {type_selected}: {product_selected} - {issue_selected}")
    prompt_sections.append(f"Priority: {priority}")
    prompt_sections.append(f"Environment: {environment}\n")
    prompt_sections.append(
        f"SUMMARY PROVIDED: {summary if summary else 'Not provided'}")
    prompt_sections.append(f"DESCRIPTION PROVIDED: {description}\n")

    # Dynamic key insight handling based on actual ticket_type
    if key_insight_text and key_insight_source_id:
        # Get the ticket type from the top result
        key_insight_ticket_type = "Incident"  # Default
        if retrieved_records:
            key_insight_ticket_type = retrieved_records[0].get('ticket_type', 'Incident')
        
        prompt_sections.append(
            f"---\nKEY TECHNICAL DETAIL FROM {key_insight_ticket_type.upper()} {key_insight_source_id}\n---\n{key_insight_text}\n------------------------------------------------------------\n")

    if resolution_hints:
        prompt_sections.append(
            f"SIMILAR INCIDENT RESOLUTIONS:\n{''.join(resolution_hints)}\n")
    else:
        prompt_sections.append("No similar resolutions found in the system.\n")

    prompt_sections.append(
        f"IMPORTANT CONTEXT: This is classified as: {type_selected}\n")

    # Get ticket type for dynamic instruction text
    key_insight_ticket_type = "incident"  # Default
    if retrieved_records and key_insight_source_id:
        for record in retrieved_records:
            if record.get('incident_id') == key_insight_source_id:
                source_type = record.get('source_record', 'INCIDENT')
                if source_type == 'PROBLEM':
                    key_insight_ticket_type = f"problem ({record.get('ticket_type', 'Problem').lower()})"
                elif source_type == 'KNOWLEDGE':
                    key_insight_ticket_type = f"knowledge article ({record.get('ticket_type', 'Knowledge').lower()})"
                else:
                    key_insight_ticket_type = record.get('ticket_type', 'Incident').lower()
                break

    prompt_sections.append(f"""
# --- START: ACCURACY ENHANCEMENT RULES ---
CRITICAL CONSTRAINT: You MUST only reference document IDs (e.g., INC000181, INC000181, PRB087681, KNB009761) that are explicitly provided in the 'SIMILAR INCIDENT RESOLUTIONS' section above. Do NOT invent, create, or refer to any other document ID.

CRITICAL RELEVANCE CHECK: You are an experienced analyst. If a 'KEY TECHNICAL DETAIL' from a {key_insight_ticket_type} is provided, determine if it represents the same **class of problem** as the user's description. Look for:
- **Same type of system/technology** (e.g., SQL Server jobs, import processes)
- **Similar error patterns** (e.g., job failures, file path issues, configuration errors)  
- **Same functional area** (e.g., data import, authentication, reporting)
- **Common infrastructure** (e.g., same servers, same processes)

If it's the same class of problem, the {key_insight_ticket_type} is relevant even if specific details differ. Adapt the resolution approach to the current context whilst leveraging the core solution pattern.
# --- END: ACCURACY ENHANCEMENT RULES ---

CRITICAL FACTUAL GROUNDING RULES:
1. You MUST base all technical details on the retrieved search results
2. Do NOT invent compatibility information, version support, or technical specifications
3. If search results show a solution works, do not contradict it
4. IMPORTANT: If search results show that a specific technical configuration IS supported or DOES work, you MUST include this POSITIVE information in your response rather than defaulting to "safer" alternatives
5. When uncertain about technical details, state "requires verification" rather than guessing
6. Reference specific document IDs when making technical claims

RETRIEVED RECORDS SHOW: {key_insight_text if key_insight_text else 'No key insight available'}
FROM {key_insight_ticket_type.upper()}: {key_insight_source_id if key_insight_source_id else 'N/A'}

Your resolution MUST align with the factual information found in the search results.

PROVIDE DETAILED ANALYSIS:

# --- START: UPDATED ROOT CAUSE INSTRUCTION ---
1. ROOT CAUSE:
   - If a 'KEY TECHNICAL DETAIL' from a {key_insight_ticket_type} has been deemed relevant by your CRITICAL RELEVANCE CHECK, you may use it as the primary explanation for the root cause.
   - Otherwise, base your analysis strictly on the user's description and other validated records.
   - For Service Requests, describe the root cause in terms of the user's goal (e.g., "The user requires confirmation and documentation...").
   - For Incidents, describe the root cause in terms of a system failure (e.g., "The system is failing to...").
# --- END: UPDATED ROOT CAUSE INSTRUCTION ---

2. SCOPE OF IMPACT: Determine who/what is affected:
   CRITICAL: You MUST base the scope description strictly on the information provided. Do NOT invent details like "slow response times" or "all users" if they are not explicitly mentioned.
   - If the description contains a specific technical condition, state that scope accurately (e.g., "Affects contacts who share an email address with a previously deactivated employee")
   - If the number of affected users is unclear, state "Unknown number of users with [specific condition]" rather than guessing
   - Single User: One person affected
   - Multiple Users: 2-10 users or single team
   - Department-Wide: Entire department or multiple teams
   - Organisation-Wide: Entire organisation or critical systems

3. INITIAL RESPONSE: Write 2-3 sentences of immediate actions the analyst should take.
   - For Service Requests: Consider starting with "guide the user through" or "provide the following procedure"
   - For Incidents: Consider starting with "verify the issue" or "apply the following workaround"
   - Reference the most relevant source type naturally in your response

4. SUMMARY QUALITY ASSESSMENT:
   Rate the summary as:
   - "good": Clear, concise, and informative
   - "poor": Vague, unclear, or lacking key information
   - "not_provided": No summary was provided

5. DESCRIPTION QUALITY ASSESSMENT (BE CRITICAL BUT FAIR):
   - Rate as "good" if it contains enough information to start an investigation, even if some details are missing.
   - Rate as "poor" ONLY if the description is completely vague (e.g., "it's broken"), unactionable, or contains no useful context. The goal is to identify truly low-quality reports, not to penalise average users.

6. ALTERNATIVE DESCRIPTION: If rated "poor", create an enhanced version that:
   - Adds assumed context based on the type/product/issue
   - Includes questions that should have been answered
   - Provides a clearer problem statement
   - Suggests what information is still needed

Return JSON:
{{
  "root_cause_preliminary": "detailed technical cause (reference {key_insight_ticket_type} if applicable)",
  "scope_of_impact": "specific description of affected systems/users",
  "scope_of_impact_category": "Single User/Multiple Users/Department-Wide/Organisation-Wide",
  "initial_response": "detailed immediate actions for analyst (2-3 sentences)",
  "summary_rating": "good/poor/not_provided",
  "description_rating": "good/poor",
  "alternative_description": "enhanced description if poor (comprehensive), empty string if good"
}}""")

    return "".join(prompt_sections)


# Step 5a: Resolution Content Generation (Focused)
def build_resolution_content_prompt(incident_details: dict, retrieved_records: list) -> str:
    logger = logging.getLogger(__name__)
    best_practices = []
    knowledge_articles = []  # Track knowledge articles separately for context

    # Process retrieved records - treat all equally based on search ranking
    for record in retrieved_records[:5]:
        source_record = record.get('source_record', 'INCIDENT')
        ticket_type = record.get('ticket_type', '')
        record_id = record.get('incident_id', record.get('id', 'Unknown'))
        
        # Create appropriate label based on source record type
        if source_record == 'PROBLEM':
            record_label = f"Problem {record_id} ({ticket_type})" if ticket_type else f"Problem {record_id}"
        elif source_record == 'KNOWLEDGE':
            record_label = f"Knowledge Article {record_id} ({ticket_type})" if ticket_type else f"Knowledge Article {record_id}"
        else:
            record_label = f"{ticket_type} {record_id}" if ticket_type else f"Record {record_id}"

        resolution_field = record.get('itil_resolution') or record.get('resolution_notes')
        
        if resolution_field:
            if source_record == 'KNOWLEDGE':
                knowledge_articles.append(f"{record_label}: {resolution_field[:300]}")
            else:
                best_practices.append(f"{record_label}: {resolution_field[:200]}")

    # Provide context about knowledge articles without artificial prioritisation
    knowledge_context = ""
    if knowledge_articles:
        knowledge_context = f"""
NOTE: Official knowledge articles found in search results:
{chr(10).join(knowledge_articles)}

These contain documented procedures. Use them when they match the user's specific need.

"""

    # Merit-based resolution fidelity check - respects search ranking
    resolution_fidelity_instruction = ""
    high_confidence_match = None

    # Check for high-confidence matches based on actual similarity scores
    for record in retrieved_records[:3]:
        score = record.get('@search.score', 0)
        if score > constants.HIGH_CONFIDENCE_THRESHOLD:
            high_confidence_match = record
            resolution_text = record.get('itil_resolution', '')
            if resolution_text:
                source_record = record.get('source_record', 'INCIDENT')
                record_id = record.get('incident_id', 'Unknown')
                
                if source_record == 'KNOWLEDGE':
                    record_label = f"Knowledge Article {record_id} ({record.get('ticket_type', 'Knowledge')})"
                    preservation_note = "This is an official, documented procedure."
                elif source_record == 'PROBLEM':
                    record_label = f"Problem {record_id} ({record.get('ticket_type', 'Problem')})"
                    preservation_note = "This is a documented problem analysis."
                else:
                    record_label = f"{record.get('ticket_type', 'Record')} {record_id}"
                    preservation_note = "This is a verified incident resolution."
                
                logger.info(f"High-confidence match detected: {record_label} (Score: {score:.3f})")
                resolution_fidelity_instruction = f"""
CRITICAL: HIGH-CONFIDENCE MATCH DETECTED (Score: {score:.3f})
The most relevant record found is {record_label}. You MUST use the specific technical steps 
from this resolution rather than generalising:

EXACT RESOLUTION TO PRESERVE:
{resolution_text}

IMPORTANT: {preservation_note} Preserve the ORIGINAL STRUCTURE and step count from this resolution. 
If the original has 4 steps, generate 4 steps. Do not expand concise resolutions into detailed 
procedures. Adapt job names and step numbers but maintain the same brevity and directness.

"""
                break

    prompt_text = f"""Generate resolution content for:
Type: {incident_details['type']}
Product: {incident_details['product']}
Issue: {incident_details['issue']}
Priority: {incident_details['priority']}
Root Cause: {incident_details['root_cause_preliminary']}
Scope: {incident_details['scope_of_impact']}

{knowledge_context}{resolution_fidelity_instruction}REQUEST TYPE CONTEXT: This is a {incident_details['type']}.{"- As a Service Request, prioritise clear procedural guidance. Similar records often contain official procedures." if incident_details['type'] == "Service Request" else "- As an Incident, focus on immediate resolution and workarounds. Similar records often contain critical fixes."}

"""

    if knowledge_articles:
        prompt_text += f"""\nOfficial Knowledge Articles in search results:
{chr(10).join(knowledge_articles)}\n"""
    
    if best_practices:
        prompt_text += f"""\nSimilar resolutions from incidents/problems:
{chr(10).join(best_practices)}\n"""
    else:
        if not knowledge_articles:
            prompt_text += "\nNo similar resolutions found in the system.\n"

    prompt_text += f"""
CRITICAL CONSTRAINT: You MUST only reference document IDs (e.g., INC12345, INC67890, PRB001234, KNB000126) that are explicitly provided in the sections above.

# --- START: MERIT-BASED RESOLUTION GENERATION RULES ---
RESOLUTION GENERATION RULES:
1.  **RELEVANCE-BASED APPROACH**: Use the highest-scoring search results as your primary source,
    regardless of record type. The semantic search has determined what's most relevant to the user's query.

2.  **HIGH-CONFIDENCE PRESERVATION**: If a high-confidence match (>0.70) exists with 
    detailed resolution steps, use those EXACT steps. Preserve specific:
    - Navigation paths ("System Administration > Account Manager")  
    - Interface names ("Service Configuration tab")
    - Button names ("Change Properties", "Add")
    - Specific examples ("'Users' or 'Incidents'")
    
3.  **KNOWLEDGE ARTICLE RECOGNITION**: When knowledge articles appear in high-ranking results,
    note their authoritative nature but don't artificially prioritise them over more relevant content.
    
4.  Be an Expert Problem Solver: Your primary goal is to provide a complete, actionable solution that an analyst can follow to resolve the ticket. Do not just diagnose the issue; provide the fix.

5.  Provide Concrete Steps: The `implementation_steps` must be a clear, step-by-step technical procedure. Avoid generic advice like "check configuration" and instead specify *what* to check.

6.  Include "Human" and Procedural Steps: A complete resolution includes all necessary process steps. Your `implementation_steps` MUST include any required 'human' actions, such as **communication** with the user, **escalation** to other teams, and steps to **confirm the fix** with the end-user.

7.  Propose Sustainable Solutions: If you identify a temporary workaround, you should also suggest a more permanent or sustainable solution if one is apparent.

8.  Synthesise, Don't Just Repeat: Blend information from the user's description and the retrieved documents into a coherent plan. Do not just restate the root cause in the resolution steps.

9.  CRITICAL FORMATTING: Do not include numbers or bullets in the implementation_steps array. Each step should start directly with the action verb (e.g., "Identify the incorrect file path" NOT "1) Identify" or "1. Identify"). The HTML formatter will add numbering automatically.
# --- END: MERIT-BASED RESOLUTION GENERATION RULES ---

Generate these specific elements using UK British English spelling throughout:
1. implementation_steps: List matching the source resolution structure (convert past tense verbs to present imperative - "identified" becomes "identify", "corrected" becomes "correct", "executed" becomes "execute") IMPORTANT: If the source content already contains proper numbered steps (1., 2., 3., etc.), preserve them as single items rather than breaking them apart. For example, if a knowledge article has "1. Navigate to X 2. Click Y 3. Select Z", keep those as complete step descriptions rather than splitting into separate array elements.
2. issue_summary: Clear technical description of the problem.
3. known_workarounds: From similar incidents or "None available"
4. diagnostic_steps: Key troubleshooting steps to verify the issue
5. resolution_summary: Brief summary for handover/documentation.

Return JSON:
{{
"implementation_steps": ["Detailed step 1", "Detailed step 2", "Detailed step 3", "Detailed step 4", "Detailed step 5"],
"issue_summary": "technical summary",
"known_workarounds": "workarounds from similar incidents or None available",
"diagnostic_steps": "key troubleshooting steps",
"resolution_summary": "summary for handover"
}}"""

    return prompt_text


# Step 5b: Format Resolution HTML (No AI - Done in code)
def format_resolution_html(incident_details: dict, resolution_content: dict) -> dict:
    """
    Format the resolution content into HTML - Done in Python, not AI.
    """
    implementation_html = "<ol>"
    for step in resolution_content.get('implementation_steps', []):
        # Remove double numbering - strip leading "1. ", "2. " etc.
        cleaned_step = re.sub(r'^\d+\.\s*', '', step.strip())
        implementation_html += f"<li>{cleaned_step}</li>"
    implementation_html += "</ol>"

    questions_html = ""
    if incident_details.get('clarifying_questions'):
        questions_html = "<ul>"
        for question in incident_details['clarifying_questions']:
            questions_html += f"<li>{question}</li>"
        questions_html += "</ul>"

    related_records_text = ', '.join(
        resolution_content.get('referenced_documents', []))

    # UPDATED: Removed Initial Analyst Actions and Priority Justification,
    # Added Implementation Plan and Clarifying Questions
    suggested_html = f"""<p><b>Issue Summary:</b> {resolution_content.get('issue_summary', 'Technical issue requiring investigation')}</p>
<p><b>Root Cause (Preliminary):</b> {incident_details.get('root_cause_preliminary', 'To be determined')}</p>
<p><b>Scope of Impact:</b> {incident_details.get('scope_of_impact', 'To be assessed')}</p>
<p><b>Clarifying Questions:</b></p>
{questions_html}
<p><b>Diagnostic Steps:</b> {resolution_content.get('diagnostic_steps', 'See implementation plan for diagnostic steps')}</p>
<p><b>Implementation Plan:</b></p>
{implementation_html}
<p><b>Known Workarounds:</b> {resolution_content.get('known_workarounds', 'None available')}</p>
<p><b>Resolution Summary:</b> {resolution_content.get('resolution_summary', 'Follow implementation plan for resolution')}</p>
<p><b>Related Records:</b> {related_records_text}</p>"""

    return {
        "implementation_plan": implementation_html,
        "suggested_resolution": suggested_html,
        "referenced_documents": resolution_content.get('referenced_documents', []),
        "list_of_clarifying_questions": questions_html
    }


# BACKWARD COMPATIBILITY: Keep original function name for fallback
def build_issue_priority_prompt(summary: str, description: str, type_selected: str,
                                product_selected: str, retrieved_records: list,
                                issue_options: list, priority_matrix, environment_selected: str = "Live",
                                issue_precedence_override: str = None) -> str:
    """
    DEPRECATED: Use build_issue_priority_prompt_only instead
    This function now just calls the new one for backward compatibility
    """
    return build_issue_priority_prompt_only(
        summary, description, type_selected, product_selected, 
        retrieved_records, issue_options, priority_matrix, 
        environment_selected, issue_precedence_override
    )


# MAIN FUNCTION - For backward compatibility
def build_user_prompt(summary: str, description: str, call_source: str,
                      retrieved_records: list, categories,
                      priority_matrix) -> str:
    """
    Full monolithic prompt - maintaining for backward compatibility.
    This is the fallback if the optimized multi-step approach fails.
    """
    records_text = ""
    for i, record in enumerate(retrieved_records[:5], 1):
        records_text += f"\nRecord {i}:\n"
        records_text += f"- ID: {record.get('id', 'N/A')}\n"
        records_text += f"- Summary: {record.get('summary', 'N/A')}\n"
        if record.get('resolution_notes'):
            records_text += f"- Resolution: {record['resolution_notes'][:300]}...\n"
        if record.get('tags'):
            records_text += f"- Tags: {', '.join(record['tags'])}\n"

    priority_rules_text = "\n".join(
        [f"- {rule.urgency} urgency + {rule.impact} impact = {rule.priority}" for rule in priority_matrix.rules])
    urgency_text = "\n".join(
        [f"- {level}: {desc}" for level, desc in priority_matrix.urgency_levels.items()])
    impact_text = "\n".join(
        [f"- {level}: {desc}" for level, desc in priority_matrix.impact_levels.items()])

    return f"""--- INCIDENT TO TRIAGE ---
Summary: {summary or 'Not provided'}
Description: {description}
Source: {call_source}

--- RETRIEVED RECORDS ---
{records_text}

--- COMPREHENSIVE TRIAGE INSTRUCTIONS ---

1. Classify Type: Must be "Incident" or "Service Request"
   - Incident: Something broken, not working, errors
   - Service Request: How-to, configuration, guidance

2. Select Product from: {', '.join(categories.product)}
   Consider keywords and similar incidents

3. Select Issue from: {', '.join(categories.issue)}
   Match to the specific problem

4. FIXED VALUES:
   environment: "Live"
   escalation_level: "Normal"

5. Determine Priority:
   Urgency: {urgency_text}
   Impact: {impact_text}
   Rules: {priority_rules_text}

6. Analyze the incident:
   - Root cause: Specific technical cause
   - Scope: Detailed impact description
   - Initial response: 2-3 sentences of immediate actions

7. Quality assessment:
   - Rate summary as good/poor/not_provided
   - Rate description as good/poor
   - Create alternative if description is poor

8. Generate clarifying questions:
   Create an HTML list of 5-8 specific questions to ask the user

9. Create detailed resolution:
   - Implementation plan: 5-8 step HTML ordered list
   - Suggested resolution: Comprehensive HTML with all ITIL sections

--- RETURN COMPLETE JSON ---
{{
  "type": "Incident or Service Request",
  "product": "from product list",
  "issue": "from issue list",
  "environment": "Live",
  "escalation_level": "Normal",
  "urgency": "High/Medium/Low",
  "impact": "High/Medium/Low",
  "priority": "1. Critical/2. High/3. Medium/4. Low",
  "summary_rating": "good/poor/not_provided",
  "description_rating": "good/poor",
  "alternative_description": "enhanced version if poor, empty if good",
  "list_of_clarifying_questions": "<ul><li>Question 1</li><li>Question 2</li>...</ul>",
  "initial_response": "detailed immediate actions",
  "implementation_plan": "<ol><li>Step 1</li><li>Step 2</li>...</ol>",
  "suggested_resolution": "<p><b>Initial Analyst Actions:</b>...</p><p><b>Issue Summary:</b>...</p>...",
  "referenced_documents": ["ID1", "ID2", "ID3"],
  "root_cause_preliminary": "detailed technical cause",
  "scope_of_impact": "detailed impact description",
  "scope_of_impact_category": "Single User/Multiple Users/Department-Wide/Organisation-Wide",
  "source": "{call_source}"
}}"""