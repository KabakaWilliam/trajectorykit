#!/usr/bin/env python3
"""
Post-processing script that applies a chain of rewrite prompts to draft articles.
Chain: REWRITE_PROMPT_OG → REFINEMENT_PROMPT_V2 → REFINEMENT_PROMPT_V3 → REFINEMENT_PROMPT_V3_B

Processes either:
1. A directory of text files (each draft in a separate file) — no research question needed
2. A JSONL file (one JSON record per line with 'article' and 'prompt' fields)

Usage:
    # Process a directory of text files (standalone articles)
    python post_process_chain.py -i ./drafts/ -o ./refined/ -c 3
    
    # Process JSONL file with per-record research questions
    python post_process_chain.py -i articles.jsonl -o refined_articles.jsonl
    
    # Test mode
    python post_process_chain.py -i articles.jsonl -o test_output.jsonl --test
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY', '')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')

from openai import OpenAI
from tqdm import tqdm

# Provider configuration
PROVIDER_CONFIG = {
    "openai": {
        "model": "gpt-5.4-2026-03-05",
        "temperature": 1.0,
        "reasoning_effort": "high",
        "max_completion_tokens": 128000,
        "base_url": None,
    },
    "anthropic": {
        "model": "claude-opus-4.6",
        "temperature": 0.6,
        "max_tokens": 128000,
        "base_url": "https://api.anthropic.com/v1",
    },
}

REWRITE_PROMPT_OG = """You are an expert research report editor. Your task is to substantially enhance the following research report to maximize its analytical depth, evidentiary rigor, and actionable value by replacing vague claims with concrete specifics, strengthening causal reasoning, and ensuring every section delivers substantive findings rather than procedural filler.

EXPANSION INSTRUCTIONS:

1. QUANTIFY EVERY EVALUATIVE CLAIM: Replace vague qualitative descriptions ("significant growth," "substantial market," "rapidly increasing") with specific numbers, percentages, or benchmarks drawn from your knowledge of the topics already discussed in the report. For example, if the report says "the market has grown significantly," replace with "the market grew 23% year-over-year to $4.7B in 2023, according to industry estimates." If the report mentions a technology's "improved performance," specify the metric (e.g., "latency reduced from 120ms to 15ms"). When you are not confident in a specific figure for a claim already in the report, reframe the claim with directional precision and analytical context rather than leaving it vague or inventing a number.

2. DEEPEN ENTITY AND CASE STUDY COVERAGE: For every landscape survey, competitive analysis, or categorical breakdown already in the report, check whether major well-known entities, sub-sectors, product lines, or landmark case studies that are directly relevant have been omitted. Add brief but substantive coverage of any missing major players or examples that you are confident belong in the discussion. For instance, if the report surveys cloud providers but omits a top-3 player, add them. Do NOT introduce entirely new domains or topics — only fill gaps within categories the report already addresses. When uncertain whether an entity is significant enough to include, err on the side of inclusion with a brief mention rather than omission.

3. CUT SCAFFOLDING AND ELIMINATE REDUNDANCY: Reduce methodological exposition, framework descriptions, evidence-grading rubrics, and meta-commentary (e.g., "In this section we will analyze...") to no more than 15-20% of total report length. If the report states the same finding or conclusion in multiple sections, consolidate it into the single most appropriate location and cross-reference elsewhere. Replace procedural descriptions of what the report *could* do with actual executed analysis. For example, transform "We propose a four-quadrant framework for evaluating X" into a completed four-quadrant analysis with entities placed and justified.

4. EXECUTE FRAMEWORKS WITH WORKED EXAMPLES: Wherever the report proposes a scoring model, classification system, evaluation matrix, or analytical method without demonstrating it, add at least one fully worked end-to-end example using entities or data already mentioned in the report. Show concrete inputs, any intermediate steps, and the specific output or score. For instance, if the report describes a risk-scoring rubric, pick one entity already discussed and walk through its score calculation step by step. This transforms theoretical frameworks into demonstrated, credible tools.

5. GROUND RISKS IN REAL INCIDENTS: For every risk category, failure mode, or governance concern discussed in the report, add at least one specific, named real-world incident, enforcement action, or documented failure with quantified consequences (e.g., financial losses, affected users, regulatory penalties). Replace abstract warnings like "data breaches pose significant risks" with "the 2017 Equifax breach exposed 147 million records and resulted in a $700M settlement." Only reference incidents you are confident actually occurred. If you cannot recall a specific incident for a given risk, strengthen the analytical framing of why that risk matters with concrete mechanistic detail instead.

6. SPECIFY REGULATORY AND STANDARDS CONTENT: When the report references a regulation, policy, or technical standard by name, extract and present its specific quantitative thresholds, compliance timelines, key provision numbers, and operational requirements. Transform "GDPR requires data protection measures" into "GDPR Article 33 requires breach notification to supervisory authorities within 72 hours; Article 83 authorizes fines up to 20M EUR or 4% of global annual turnover." Draw on your knowledge of well-known regulations already mentioned in the report. When you are unsure of specific provision details, note the regulation's general operative requirements rather than guessing at numbers.

7. UPDATE STALE REFERENCE POINTS: Scan the report for data points, benchmarks, rankings, or scenario ranges that appear outdated relative to what you know. If the report presents already-surpassed levels as future targets, or uses older figures when more recent ones from the same domain are widely known, update them. For example, if the report cites a 2020 market size as current, and you know the 2023 figure, replace it. Flag any update you make by contextualizing it (e.g., "as of 2023, this figure has reached X, up from the Y cited in earlier analyses"). Do not fabricate precise figures you are uncertain about — instead note that the reference point may be outdated and frame the analytical implication.

8. BUILD CONSOLIDATED COMPARISON TABLES: Where the report compares multiple items (tools, methods, companies, policies, technologies) across scattered prose sections, create a single unified comparison table that consolidates all key dimensions side by side. Organize the table by the user's decision criteria or goal, not by the items themselves. Include a clear tiered recommendation or ranking row. If the report already has multiple overlapping ranked lists, merge them into one coherent hierarchy with explicit justification for the ordering.

9. STRENGTHEN CAUSAL REASONING: Where the report makes macro-level claims (e.g., "AI adoption is transforming healthcare"), connect them to specific micro-level mechanisms — the behavioral changes, technical processes, or economic dynamics through which the effect actually operates. For example: "AI-assisted radiology reduces diagnostic turnaround from 48 hours to under 1 hour by automating preliminary scan classification, allowing radiologists to focus on ambiguous cases." When the report notes that a historical relationship or core assumption has shifted or broken down, explicitly assess what this means for the report's overall conclusions rather than treating it as a footnote.

10. IMPROVE SOURCE QUALITY FRAMING: Where the report cites generic or secondary sources for entity-specific claims, note the type of primary source that would be most authoritative (e.g., "per the company's 10-K filing," "according to the FDA's 510(k) clearance database"). Do NOT fabricate citations or add fake reference numbers. Do reframe existing claims to indicate the caliber of evidence behind them. For example, transform "Company X has strong revenue growth [3]" into "Company X reported $12.4B in FY2023 revenue, up 18% year-over-year per its annual filing [3]" — enriching the claim while preserving the original citation.

CONTENT SOURCING RULES:
- You MUST preserve all existing factual content from the original report
- You MAY and SHOULD expand on topics already covered in the report with additional relevant context, examples, and explanations drawn from your knowledge, PROVIDED they are directly relevant to the report's existing subject matter
- Do NOT introduce entirely new topics or tangential discussions not connected to the report's scope
- When adding context from your knowledge, present it as established knowledge, NOT as new research findings
- Keep the same language as the original report (if the report is in Chinese, write expansions in Chinese; if English, write in English)
- DATA PROVENANCE: Do NOT invent specific numerical data (revenue figures, market sizes, growth rates, financial metrics) that are not already in the original report. You MAY restructure, tabularize, and synthesize data that IS in the report. You MAY add widely known public facts (e.g., company founding dates, headquarters locations, well-known product names). When the report has data gaps, improve the analytical framing around the gap rather than filling it with ungrounded estimates.

CITATION RULES:
- Preserve ALL existing citations, references, footnotes, and source links EXACTLY as they appear
- Do NOT remove, renumber, or modify any citation markers
- New sentences you add should NOT include fabricated citations — only original content retains its citations

LENGTH TARGET:
- The enhanced report should be approximately 50% longer than the original — no more, no less
- Do NOT produce a report that is more than 60% longer than the original
- Every section should feel complete and well-developed, not summarized
- Prioritize replacing weak content with strong content over adding bulk

ORIGINAL REPORT:
{article}"""

REFINEMENT_PROMPT_V2 = """You are an expert research report editor specializing in completeness and task alignment. Your task is to enhance this research report by ensuring all requirements are fully addressed, filling knowledge gaps, and strengthening connections—while preserving every word of existing content.

CORE PRINCIPLE: This is an expansion and alignment pass, not a compression pass. Every word in the original report has value. Your job is to add coverage, depth, and clarity—not to condense.

THREE-PHASE ENHANCEMENT:

PHASE 1: INSTRUCTION COMPLIANCE & COMPLETENESS
Your first priority is ensuring the report fully addresses every requirement of the task:

1. AUDIT TASK SCOPE:
   - Identify what the task is asking for (landscape assessment? comparative analysis? risk evaluation? implementation guidance?)
   - Verify each major requirement is substantively addressed—not just mentioned
   - If a requirement is addressed implicitly, add explicit framing that makes it clear you're systematically working through it
   - Example: If the task asks to evaluate trade-offs, ensure all key trade-offs are named and explained, not just implied

2. CLOSE COVERAGE GAPS WITHIN SCOPE:
   - If the report surveys entities, approaches, or options, verify major ones aren't omitted
   - For each category discussed (e.g., vendors, methodologies, regulatory frameworks), ask: "Is anything obvious missing?"
   - Add coverage of missing major entities/approaches drawn from domain knowledge—only as brief substantive mentions, not as new topics
   - Example: If report covers cloud providers but omits a top-tier player, add them with 2-3 sentences on relevant capabilities/positioning

3. STRENGTHEN STRUCTURAL CLARITY:
   - Ensure each major section explicitly states: "Here's what this section addresses" and "Why it matters to the overall question"
   - Use consistent section structure so readers can track systematic progress through requirements
   - Add subheadings where sections are long, to make the logic visible

PHASE 2: COMPREHENSIVENESS & DEPTH-FILLING
After ensuring task alignment, deepen analytical coverage:

4. IDENTIFY AND ADD MISSING CONTEXT:
   - For every major claim or recommendation, ask: "What related concepts, alternatives, boundary conditions, or methodological variants should a domain expert see here?"
   - Add coverage of adjacent considerations without introducing new topics
   - Focus on: how scenarios differ (edge cases, variations), methodological alternatives, related regulatory/technical constraints, comparable case studies
   - Example: If recommending an approach, add coverage of when it works best, when it struggles, and relevant prerequisites

5. GROUND ABSTRACT CLAIMS IN CONCRETE EXAMPLES:
   - For every analytical finding or recommendation, add at least one concrete example or worked case
   - Replace "organizations report benefits" with "companies like [X] achieved [specific outcome]"—using context already in the report or widely known facts
   - If frameworks or processes are described, demonstrate them with a real-world scenario

6. DEEPEN COMPARATIVE ANALYSIS:
   - Where the report evaluates or ranks options/approaches, ensure all key dimensions are discussed
   - Consolidate scattered comparisons into a coherent side-by-side analysis
   - Make trade-offs explicit rather than implicit: "Approach A excels at [dimension] but requires [constraint]; Approach B trades [benefit] for [advantage]"

7. STRENGTHEN MECHANISTIC REASONING:
   - Where the report makes cause-and-effect claims, strengthen the mechanism: Why does X lead to Y? What are intermediate steps?
   - Replace vague connections with specific causal chains
   - Example: Instead of "consolidation improves efficiency," write "consolidation improves efficiency by reducing integration points (from N² to N), lowering ongoing maintenance by ~30%"

PHASE 3: NARRATIVE REFINEMENT (LIGHT TOUCH)
After expanding for completeness, refine presentation:

8. STRENGTHEN CONNECTIONS BETWEEN IDEAS:
   - Add transition sentences that explicitly bridge related concepts across sections
   - Cross-reference sections: "As discussed in Section X, [finding] directly impacts [what comes next in Section Y]"
   - Ensure logical flow: context → problem/opportunity → analysis → implications → next step

9. ENHANCE CLARITY THROUGH SIGNPOSTING:
   - Precede key data points and examples with signal phrases: "Evidence from [source type]..." / "Case studies show..." / "Industry data reveals..." / "Analysis indicates..."
   - This clarifies evidence type without inventing citations
   - Improves scannability and confidence in claims

10. IMPROVE SECTION FRAMING:
    - Each section should open with a thesis (1-2 sentences on why this section matters)
    - Each section should close with implications (what this means for the next topic or the overall conclusion)
    - Break long paragraphs—keep to <5 sentences per paragraph

PRESERVATION RULES (CRITICAL):
- Keep all existing data, citations, examples, and detailed findings EXACTLY as written
- Do NOT remove, abbreviate, or combine sections even if they seem repetitive
- Do NOT cut content to reduce length
- Do NOT create summaries or condensed versions of existing material
- Additions (transitions, gap-filling, causal strengthening, context) may add 5-10% to length—this is appropriate for comprehensiveness
- If content seems tangential, ADD a connection statement rather than removing it

CONTENT SOURCING:
- You MAY draw on domain knowledge to fill gaps (missing entities, related methodologies, boundary conditions)
- You MUST preserve all facts, figures, and citations from the original report exactly
- Do NOT invent data or create fabricated examples—only add widely known facts and scenarios that naturally extend existing topics
- Do NOT introduce entirely new domains or subjects not connected to the report's scope

OUTPUT FORMAT:
- Return only the enhanced report
- Do not include editing notes, compliance statements, or commentary
- Maintain the original voice, technical level, and depth
- The output should read as a complete, professionally developed research report with no trace of editing instructions

ORIGINAL RESEARCH QUESTION:
{question}

ORIGINAL REPORT:
{article}"""

REFINEMENT_PROMPT_V3 = """You are an expert research report editor specializing in instruction-following precision and comprehensive coverage. Your task is to enhance this research report by ensuring it fully addresses ALL requirements from the ORIGINAL RESEARCH QUESTION while preserving every word of existing content.

CORE PRINCIPLE: Separate internal planning from output writing. Plan rigorously (verify all requirements are met), then write naturally (use signposting only where connections aren't obvious). Every word in the original report has value.

FOUR-PHASE ENHANCEMENT (Internal Planning → Natural Expansion → Calibrated Signposting → Validation):

PHASE 1: REQUIREMENT AUDIT & INTERNAL MAPPING (Planning work—not output)
Before any edits, understand what the task requires:

1. DECOMPOSE TASK REQUIREMENTS:
   - Read the original research question carefully
   - Extract ALL distinct requirements (e.g., "List major vendors", "Compare on price", "Evaluate risks", "Explain methodology")
   - Categorize: scope requirements (what must be covered), analytical requirements (what analysis is needed), format requirements (how to present), depth requirements (how detailed)
   - Create a requirement checklist for internal validation—do NOT include this in the output

2. MAP CURRENT CONTENT TO REQUIREMENTS (Internal mapping):
   - Read each section of the report
   - Identify which requirement(s) it addresses
   - Note which requirements are covered substantively vs only mentioned vs missing entirely
   - This is internal work—you are building a mental map of coverage gaps

3. IDENTIFY COVERAGE GAPS:
   - Which requirements are missing entirely?
   - Which are addressed only implicitly or superficially?
   - Which categories/entities/dimensions are conspicuously absent?
   - Which comparative dimensions are missing?
   - Mark these for Phase 2 coverage

INTERNAL CHECKPOINT: You now have a requirement checklist and a gap map. Do NOT output this. Move to Phase 2.

PHASE 2: COMPREHENSIVE COVERAGE & GAP CLOSURE (Content expansion)
Now close gaps by adding content, never removing it:

4. ADD MISSING COVERAGE:
   - For each gap identified in Phase 1, add substantive coverage
   - If the task asks for specific categories/entities/comparisons, ensure all major ones are addressed
   - If the task asks for evaluation across dimensions, ensure all key dimensions appear in the report
   - If scattered comparisons exist, consolidate into a unified framework—but keep ALL original content
   - Example: If vendors are discussed in separate sections, add a unified comparison table/framework that brings all vendors together without removing original discussions

5. STRENGTHEN MECHANISTIC DEPTH & SYNTHESIZE STRATEGIC INSIGHTS:
   - For major findings or recommendations, add "why" and "how": What process leads to this result? What mechanism explains this?
   - Add boundary conditions: When does this apply? When doesn't it?
   - Add comparative context: How does this differ from alternatives?
   - Add concrete examples from the report's existing content
   - Example: Instead of "Cloud adoption improves scalability," write "Cloud adoption enables scaling from 10K to 1M users by removing the need for capacity planning, reducing deployment time from weeks to hours"

   **SYNTHESIS & "SO WHAT" REQUIREMENT—EXTRACT DECISION-RELEVANCE:**
   - After explaining mechanisms, synthesize the implication: What does this combination of factors mean for the overall question?
   - Generate NEW findings by connecting existing facts: Does this finding contradict, strengthen, or reframe other findings in the report?
   - **CRITICAL: State the decision-relevant consequence:** What strategic choice or evaluation changes given this insight?
     - Example: NOT just "consolidation reduces points from N² to N" BUT "This architecturally mandates centralized orchestration, eliminating federated approaches most teams default to—a strategic constraint organizations typically miss"
     - Extract: Who should care? What decision does this inform? What would they choose differently?
   - Explicitly state the "so what": "This means [specific consequence for strategy/decision], which impacts [what matters to the question]"
   - Example synthesis: "The 3× cost variance across vendors stems from licensing models (fact A) + feature scope (fact B) + deployment time (fact C). Combined, this means organizations choosing based on upfront price will underestimate total cost of ownership by 40-60%, making licensing transparency the PRIMARY evaluation criterion—not feature breadth as industry practice suggests"
   - Identify missing analysis: What adjacent dimension, trade-off, or boundary condition would help readers understand the full picture?
   - Add that adjacent insight—don't just acknowledge the gap
   - If mechanistic explanation reveals a strategic implication, extract and state it explicitly

6. GROUND ABSTRACT CLAIMS:
   - For every abstract conclusion or framework, add one worked example
   - Use concrete entities/numbers from the article itself, or widely known facts
   - Show: Input → Process → Output (with specifics)
   - Never invent data; only extend what's already in the report

PHASE 3: NATURAL NARRATIVE CLARITY WITH CALIBRATED SIGNPOSTING (Writing)
Now refine presentation using signposting only where connections aren't obvious:

7. NATURAL PARAGRAPH STRUCTURE:
   - Each paragraph should flow logically from what precedes it
   - Use topic sentences to indicate direction
   - Connect sentences with natural bridges (avoid forcing explicit linkage everywhere)
   - Max 5 sentences per paragraph keeps density manageable
   - Let obvious progressions flow implicitly; only add explicit transitions where the connection is non-obvious

8. CALIBRATED SIGNPOSTING (Use ONLY where needed):
   - DO use explicit transitions when jumping between disparate topics (e.g., "Having analyzed cost, we now examine deployment time because...")
   - DO use signal phrases ONLY when evidence type is non-obvious (e.g., "Industry data reveals..." for broader trends, "Our testing showed..." for specific validation)
   - DO NOT add "This section addresses requirement X" to every section—only where the connection would otherwise be unclear
   - DO NOT precede obvious progressions with explicit bridges
   - Example of NOT needed: Describing vendors in a vendor landscape → readers understand the connection without signposting

9. SECTION FRAMING (Light touch):
   - Opening: Brief thesis on why this section matters and what question it answers (1-2 sentences)
   - Body: Substantive findings with examples and reasoning (flows naturally)
   - Closing: Brief implication for the next topic or overall argument (1 sentence)
   - Only add explicit "this addresses requirement X" framing if the requirement connection is truly non-obvious
   - Example good framing: "Deployment time directly impacts adoption velocity, which leads to the cost analysis in the next section"

10. ENHANCE SCANNABILITY WHERE DENSITY REQUIRES IT:
    - If a section is long or complex, add subheadings to guide readers through topics
    - Use consistent, natural language (not forced signal phrases)
    - Ensure logical flow: context → analysis → concrete implications → next topic

PHASE 4: FINAL REQUIREMENT VALIDATION CHECKPOINT (Validation only—no new edits)
After all writing, verify coverage:

11. VALIDATE REQUIREMENT COVERAGE:
    - Use your internal requirement checklist from Phase 1
    - For each requirement, locate the specific content that addresses it
    - Verify it's addressed substantively, not just mentioned
    - If any requirement is truly missing or only surface-level, RETURN and add coverage in Phase 2—do not proceed to output

12. CHECK COMPLETENESS:
    - If task asks for "compare": All major entities/aspects are included?
    - If task asks for "landscape": All major players/approaches are visible?
    - If task asks for "analysis": Are mechanisms and "why" questions answered, not just findings?
    - If task asks for "recommendations": Are criteria and trade-offs transparent?
    - Confidence check: Can a domain expert point to content for every task requirement?

PRESERVATION RULES (CRITICAL):
- Keep ALL existing data, citations, examples, findings EXACTLY as written
- Do NOT remove, abbreviate, or combine sections
- Do NOT cut content to reduce length
- Additions (covering gaps, mechanistic depth, concrete examples) may add 5-15% to length—appropriate for completeness
- Word count: Output should be ≥ input × 0.95 (if shorter, you removed too much)

CONTENT SOURCING:
- Draw on domain knowledge to fill coverage gaps (missing entities, methodologies, boundary conditions)
- PRESERVE all facts, figures, citations from original report exactly
- Do NOT invent data or fabricated examples—add only widely known facts that naturally extend existing topics
- Do NOT introduce new domains unrelated to the report's scope

OUTPUT FORMAT:
- Return ONLY the enhanced report
- Do NOT include requirement checklists, requirement mapping, editing notes, meta-commentary, or task analysis
- Maintain original voice, technical level, and depth
- The output should read as a naturally written, professionally developed research report with no trace of instructions
- Readers may not notice editing, but the content now substantively addresses all task requirements

ORIGINAL REPORT:
{article}"""

REFINEMENT_PROMPT_V3_B = """You are an expert research report editor specializing in instruction-following precision, comprehensive coverage, and strategic clarity. Your task is to enhance this research report by ensuring it fully addresses ALL requirements from the ORIGINAL RESEARCH QUESTION while preserving every word of existing content.

<ORIGINAL RESEARCH QUESTION>
{question}
</ORIGINAL RESEARCH QUESTION>

CORE PRINCIPLE: Separate internal planning from output writing. Plan rigorously (verify all requirements are met), then write naturally (prioritize scannability and finding-level clarity). Every word in the original report has value.

FOUR-PHASE ENHANCEMENT (Internal Planning → Finding-Driven Expansion → Strategic Scannability → Validation):

PHASE 1: REQUIREMENT AUDIT & INTERNAL MAPPING (Planning work—not output)
Before any edits, understand what the task requires:

1. DECOMPOSE TASK REQUIREMENTS:
   - Read the original research question carefully
   - Extract ALL distinct requirements (e.g., "List major vendors", "Compare on price", "Evaluate risks", "Explain methodology")
   - Categorize: scope requirements (what must be covered), analytical requirements (what analysis is needed), format requirements (how to present), depth requirements (how detailed)
   - Create a requirement checklist for internal validation—do NOT include this in the output

2. MAP CURRENT CONTENT TO REQUIREMENTS (Internal mapping):
   - Read each section of the report
   - Identify which requirement(s) it addresses
   - Note which requirements are covered substantively vs only mentioned vs missing entirely
   - This is internal work—you are building a mental map of coverage gaps

3. IDENTIFY COVERAGE GAPS:
   - Which requirements are missing entirely?
   - Which are addressed only implicitly or superficially?
   - Which categories/entities/dimensions are conspicuously absent?
   - Which comparative dimensions are missing?
   - Mark these for Phase 2 coverage

INTERNAL CHECKPOINT: You now have a requirement checklist and a gap map. Do NOT output this. Move to Phase 2.

PHASE 2: COMPREHENSIVE COVERAGE, GAP CLOSURE & FINDING-LEVEL SYNTHESIS (Content expansion)
Now close gaps by adding content, never removing it:

4. ADD MISSING COVERAGE:
   - For each gap identified in Phase 1, add substantive coverage
   - If the task asks for specific categories/entities/comparisons, ensure all major ones are addressed
   - If the task asks for evaluation across dimensions, ensure all key dimensions appear in the report
   - If scattered comparisons exist, consolidate into a unified framework—but keep ALL original content
   - Example: If vendors are discussed in separate sections, add a unified comparison table/framework that brings all vendors together without removing original discussions

5. STRENGTHEN MECHANISTIC DEPTH & SYNTHESIZE STRATEGIC INSIGHTS WITH WORKED DECISION EXAMPLES:
   - For major findings or recommendations, add "why" and "how": What process leads to this result? What mechanism explains this?
   - Add boundary conditions: When does this apply? When doesn't it?
   - Add comparative context: How does this differ from alternatives?
   - Add concrete examples from the report's existing content
   - Example: Instead of "Cloud adoption improves scalability," write "Cloud adoption enables scaling from 10K to 1M users by removing the need for capacity planning, reducing deployment time from weeks to hours"

   **SYNTHESIS & "SO WHAT" REQUIREMENT—EXTRACT DECISION-RELEVANCE WITH WORKED EXAMPLE:**
   - After explaining mechanisms, synthesize the implication: What does this combination of factors mean for the overall question?
   - Generate NEW findings by connecting existing facts: Does this finding contradict, strengthen, or reframe other findings in the report?
   - **CRITICAL: State the decision-relevant consequence with a worked decision example:**
     - Example: NOT just "consolidation reduces points from N² to N" BUT "This architecturally mandates centralized orchestration, eliminating federated approaches most teams default to—a strategic constraint organizations typically miss. For example, a team evaluating whether to deploy system X should prioritize centralized infrastructure readiness in their planning, not feature completeness"
     - Extract: Who should care? What decision does this inform? What would they choose differently?
     - Add worked decision example: "Consider the choice between approaches A and B: Given the latency constraints we identified, choosing A means accepting 60-second startup but gaining flexibility; choosing B eliminates startup delay but locks infrastructure. For time-sensitive applications, this decision tips toward B; for development-focused teams, A's flexibility wins—but cost analysis [section X] shows the true differentiator"
   - Explicitly state the "so what": "This means [specific consequence for strategy/decision], which impacts [what matters to the question]"
   - Example synthesis with worked decision: "The 3× cost variance across vendors stems from licensing models (fact A) + feature scope (fact B) + deployment time (fact C). Combined, this means organizations choosing based on upfront price will underestimate total cost of ownership by 40-60%, making licensing transparency the PRIMARY evaluation criterion—not feature breadth as industry practice suggests. Practically: if your budget is $50k, pricing transparency could expand your viable vendor list from 2 to 5 options, making this a strategic decision lever"
   - Identify missing analysis: What adjacent dimension, trade-off, or boundary condition would help readers understand the full picture?
   - Add that adjacent insight—don't just acknowledge the gap
   - If mechanistic explanation reveals a strategic implication, extract and state it explicitly

6. GROUND ABSTRACT CLAIMS:
   - For every abstract conclusion or framework, add one worked example
   - Use concrete entities/numbers from the article itself, or widely known facts
   - Show: Input → Process → Output (with specifics)
   - Never invent data; only extend what's already in the report

PHASE 3: STRATEGIC READABILITY WITH FINDING-BASED SUBHEADINGS & STRONG TOPIC SENTENCES (Writing)
Now refine presentation using scannability to surface findings naturally:

7. MANDATORY STRONG TOPIC SENTENCES (Thread requirement relevance implicitly):
   - Each paragraph opening sentence must signal: "What finding, comparison, or requirement-relevant insight is this paragraph delivering?"
   - Topic sentences should thread requirement relevance WITHOUT explicit "this addresses requirement X" language
   - Examples of strong topic sentences:
     * (From a cost analysis): "Licensing model differences account for the majority of vendor cost divergence"—immediately tells reader this paragraph reveals a KEY FINDING
     * (From methodology): "Deployment latency shifts from an implementation detail to a strategic constraint in time-sensitive domains"—signals both finding AND its relevance
     * (From comparison): "While feature breadth favors Vendor A, the cost-of-ownership analysis reverses this ranking"—reader immediately understands this contradicts earlier position
   - Test: Does your topic sentence surface a FINDING or COMPARISON? Or is it just an organizational transition like "Next, we examine cost"?
   - Revision rule: If topic sentence is just organizational, strengthen it to surface the finding: "Cost analysis reveals" (change), not "Let's now look at cost" (weak)

8. CALIBRATED SIGNPOSTING (Use ONLY where needed):
   - DO use explicit transitions when jumping between disparate topics (e.g., "Having analyzed cost, we now examine deployment time because...")
   - DO use signal phrases ONLY when evidence type is non-obvious (e.g., "Industry data reveals..." for broader trends, "Our testing showed..." for specific validation)
   - DO NOT add "This section addresses requirement X" to every section—only where the connection would otherwise be unclear
   - DO NOT precede obvious progressions with explicit bridges
   - Example of NOT needed: Describing vendors in a vendor landscape → readers understand the connection without signposting

9. SECTION FRAMING WITH CLEAR STRATEGIC BRIDGE (Medium touch):
   - Opening: Brief thesis on the KEY FINDING this section reveals and what requirement it addresses (1-2 sentences, finding-focused)
   - Body: Substantive findings with examples and reasoning (flows naturally)
   - Closing: Explicit bridge to next section showing WHY this finding matters for what follows (1-2 sentences, decision-oriented)
   - Example opening: "Vendor cost structures vary by 3×, with licensing models as the primary driver rather than feature scope"—reader immediately knows the section's core finding
   - Example closing: "Understanding this licensing-driven cost divergence is essential for the next section's coverage of total cost of ownership, where initial price proves misleading"—reader sees why this section matters

10. FINDING-BASED SUBHEADINGS FOR STRATEGIC CLARITY (CRITICAL FOR THIS VERSION):
    - Subheadings MUST reference the actual FINDING or comparison in that section, not generic category names
    - ❌ WEAK subheading: "Cost Analysis" or "Vendor Comparison"
    - ✅ STRONG subheading: "Licensing Models Drive 40% of Cost Variance" or "Deployment Latency Creates Hidden Vendor Lock-In"
    - Benefit: Reader scanning the report immediately grasps the KEY FINDINGS without reading sections
    - Rule: Every subheading must answer "What did you FIND?" not "What CATEGORY is this?"
    - If a section is long or complex, use finding-based subheadings (nested under major sections) to guide readers through specific findings within the analysis
    - Example structure:
      ```
      ## Vendor Cost Architecture (Major section)
      ### Licensing Models Account for 70% of Variance (Finding-based subheading)
      ### Feature Scope Has Minimal Cost Impact (Finding-based subheading)
      ```
    - Each finding-based subheading should be immediately followed by a 1-2 sentence context statement explaining what the finding means

PHASE 4: FINAL REQUIREMENT VALIDATION CHECKPOINT (Validation only—no new edits)
After all writing, verify coverage:

11. VALIDATE REQUIREMENT COVERAGE:
    - Use your internal requirement checklist from Phase 1
    - For each requirement, locate the specific content that addresses it
    - Verify it's addressed substantively, not just mentioned
    - If any requirement is truly missing or only surface-level, RETURN and add coverage in Phase 2—do not proceed to output

12. CHECK COMPLETENESS:
    - If task asks for "compare": All major entities/aspects are included?
    - If task asks for "landscape": All major players/approaches are visible?
    - If task asks for "analysis": Are mechanisms and "why" questions answered, not just findings?
    - If task asks for "recommendations": Are criteria and trade-offs transparent?
    - Confidence check: Can a domain expert point to content for every task requirement?

PRESERVATION RULES (CRITICAL):
- Keep ALL existing data, citations, examples, findings EXACTLY as written
- Do NOT remove, abbreviate, or combine sections
- Do NOT cut content to reduce length
- Additions (covering gaps, mechanistic depth, concrete examples, worked decision examples) may add 5-15% to length—appropriate for completeness
- Word count: Output should be ≥ input × 0.95 (if shorter, you removed too much)

CONTENT SOURCING:
- Draw on domain knowledge to fill coverage gaps (missing entities, methodologies, boundary conditions)
- PRESERVE all facts, figures, citations from original report exactly
- Do NOT invent data or fabricated examples—add only widely known facts that naturally extend existing topics
- Do NOT introduce new domains unrelated to the report's scope

OUTPUT FORMAT:
- Return ONLY the enhanced report
- Do NOT include requirement checklists, requirement mapping, editing notes, meta-commentary, or task analysis
- Maintain original voice, technical level, and depth
- The output should read as a naturally written, professionally developed research report with no trace of instructions
- Readers may not notice editing, but the content now substantively addresses all task requirements with finding-level clarity

ORIGINAL REPORT:
{article}"""

# Chain of prompts to apply sequentially
REWRITE_CHAIN = [
    {"name": "REWRITE_OG", "prompt": REWRITE_PROMPT_OG},
    {"name": "REFINEMENT_V2", "prompt": REFINEMENT_PROMPT_V2},
    {"name": "REFINEMENT_V3", "prompt": REFINEMENT_PROMPT_V3},
    {"name": "REFINEMENT_V3_B", "prompt": REFINEMENT_PROMPT_V3_B},
]


def rewrite_article_single_pass(client, article_content, provider, provider_config, question, prompt_template):
    """Apply a single rewrite pass using the given prompt template."""
    try:
        prompt = prompt_template.format(
            question=question or "",
            article=article_content
        )
        
        config = provider_config[provider]
        model = config["model"]
        
        request_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_completion_tokens", config.get("max_tokens", 8192)),
        }
        
        # Add reasoning_effort for OpenAI if supported
        if provider == "openai" and "reasoning_effort" in config:
            request_params["reasoning_effort"] = config["reasoning_effort"]
        
        response = client.chat.completions.create(**request_params)
        rewritten = response.choices[0].message.content
        
        return {
            "success": True,
            "article": rewritten,
            "error": None,
        }
    except Exception as e:
        error_msg = f"Rewrite failed: {str(e)}"
        return {
            "success": False,
            "article": article_content,  # Return original on failure
            "error": error_msg,
        }


def apply_rewrite_chain(client, article_content, provider, provider_config, question):
    """Apply the full chain of rewrites sequentially."""
    current_article = article_content
    chain_results = []
    
    for i, rewrite_step in enumerate(REWRITE_CHAIN):
        step_name = rewrite_step["name"]
        prompt_template = rewrite_step["prompt"]
        
        result = rewrite_article_single_pass(
            client,
            current_article,
            provider,
            provider_config,
            question,
            prompt_template,
        )
        
        chain_results.append({
            "step": step_name,
            "success": result["success"],
            "error": result["error"],
        })
        
        current_article = result["article"]
        
        if not result["success"]:
            print(f"  ⚠ Step {i+1} ({step_name}) failed: {result['error']}")
        else:
            print(f"  ✓ Step {i+1} ({step_name}) completed")
    
    return {
        "article": current_article,
        "chain_results": chain_results,
        "success": all(r["success"] for r in chain_results),
    }


def load_input_data(input_path):
    """Load articles from either a JSONL file or directory of text files."""
    records = []
    
    input_path = Path(input_path)
    
    if input_path.suffix == ".jsonl":
        # Load from JSONL file
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        records.append({
                            "id": data.get("id", "unknown"),
                            "prompt": data.get("prompt", ""),
                            "article": data.get("article", ""),
                            "source": "jsonl",
                        })
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line")
    elif input_path.is_dir():
        # Load from directory of text files
        for file_path in sorted(input_path.glob("*.md")) + sorted(input_path.glob("*.txt")):
            with open(file_path, "r") as f:
                content = f.read()
                records.append({
                    "id": file_path.stem,
                    "prompt": "",  # No prompt for directory mode
                    "article": content,
                    "source": "directory",
                })
    else:
        raise ValueError(f"Input must be a JSONL file or directory: {input_path}")
    
    return records


def save_output_data(output_path, records, is_jsonl=True):
    """Save processed records to output file or directory."""
    output_path = Path(output_path)
    
    if is_jsonl:
        # Write to JSONL file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps({
                    "id": record["id"],
                    "prompt": record["prompt"],
                    "article": record["article"],
                    "chain_results": record.get("chain_results", []),
                    "success": record.get("success", True),
                }) + "\n")
    else:
        # Write to directory
        output_path.mkdir(parents=True, exist_ok=True)
        for record in records:
            file_path = output_path / f"{record['id']}.md"
            with open(file_path, "w") as f:
                f.write(record["article"])


def main():
    parser = argparse.ArgumentParser(
        description="Post-process drafts through a chain of rewrite prompts"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input: JSONL file or directory of text files (*.md, *.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path: JSONL file or directory (auto-detected based on input)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent rewrite chains (default: 2)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider: openai (default) or anthropic",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first record",
    )
    args = parser.parse_args()

    # Validate provider and API key
    if args.provider == "openai":
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)
        api_key = openai_api_key
    elif args.provider == "anthropic":
        if not anthropic_api_key:
            print("Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        api_key = anthropic_api_key

    # Initialize client
    client_kwargs = {"api_key": api_key}
    base_url = PROVIDER_CONFIG[args.provider].get("base_url")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Determine input/output format
    input_path = Path(args.input)
    output_path = Path(args.output)
    is_jsonl_input = input_path.suffix == ".jsonl"
    is_jsonl_output = output_path.suffix == ".jsonl"

    # Load input data
    print(f"Loading input from: {args.input}")
    records = load_input_data(args.input)
    print(f"Loaded {len(records)} records")

    # Test mode
    if args.test:
        records = records[:1]
        args.concurrency = 1
        print("TEST MODE: Processing 1 record only")

    # Process records with thread pool
    print(f"\nProvider: {args.provider}")
    print(f"Model: {PROVIDER_CONFIG[args.provider]['model']}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print(f"Rewrite chain: {' → '.join([s['name'] for s in REWRITE_CHAIN])}\n")

    output_lock = threading.Lock()

    def process_record(record):
        """Apply full rewrite chain to a record."""
        article_id = record["id"]
        print(f"\nProcessing: {article_id}")
        
        result = apply_rewrite_chain(
            client,
            record["article"],
            args.provider,
            PROVIDER_CONFIG,
            record["prompt"],
        )
        
        # Update record with result
        record["article"] = result["article"]
        record["chain_results"] = result["chain_results"]
        record["success"] = result["success"]
        
        return record

    # Execute with thread pool
    processed_records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_record, record): record["id"] for record in records}

        with tqdm(total=len(records), desc="Processing articles") as pbar:
            for future in as_completed(futures):
                try:
                    processed_record = future.result()
                    processed_records.append(processed_record)
                except Exception as e:
                    article_id = futures[future]
                    print(f"Error processing {article_id}: {e}")
                finally:
                    pbar.update(1)

    # Save output
    print(f"\nSaving results to: {args.output}")
    save_output_data(args.output, processed_records, is_jsonl=is_jsonl_output)

    # Print summary
    success_count = sum(1 for r in processed_records if r.get("success", False))
    print(f"\n✓ Completed: {success_count}/{len(processed_records)} successful")
    if success_count < len(processed_records):
        print(f"⚠ Failed: {len(processed_records) - success_count} (fallback to original)")


if __name__ == "__main__":
    main()
