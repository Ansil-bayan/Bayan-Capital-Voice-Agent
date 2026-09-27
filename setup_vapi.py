import os
import requests
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.getenv("VAPI_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")  # -> points at /vapi-events, handles everything
BUSINESS_ACTION_URL = os.getenv("BUSINESS_ACTION_URL")
HUMAN_AGENT_PHONE = os.getenv("HUMAN_AGENT_PHONE")

headers = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json",
}

# 1. Knowledge Base Search Tool (unchanged)
kb_tool_payload = {
    "type": "function",
    "async": False,
    "function": {
        "name": "search_knowledge_base",
        "description": "Searches the business_loans_knowledge_base.json for FAQs, interest rates, policies, and loan terms.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query derived from user request."}
            },
            "required": ["query"],
        },
    },
    "server": {"url": SERVER_URL},
}

# 2. Lead Qualification Tool (unchanged)
lead_tool_payload = {
    "type": "function",
    "async": False,
    "function": {
        "name": "submit_lead_qualification",
        "description": "Submits qualified lead details to the CRM and schedules a call back.",
        "parameters": {
            "type": "object",
            "properties": {
                "business_name": {"type": "string"},
                "monthly_revenue": {"type": "number"},
                "years_in_business": {"type": "number"},
                "credit_score": {"type": "number"},
                "loan_amount_requested": {"type": "number"},
            },
            "required": ["business_name", "monthly_revenue", "years_in_business", "credit_score"],
        },
    },
    "server": {"url": BUSINESS_ACTION_URL},
}

# 3. Native Vapi Transfer Call Tool (unchanged)
transfer_tool_payload = {
    "type": "transferCall",
    "destinations": [
        {"type": "number", "number": HUMAN_AGENT_PHONE, "description": "Transfer to a human loan specialist"}
    ],
    "messages": [{"type": "request-start", "content": "I am transferring you to a human specialist now."}],
}

kb_tool_res = requests.post("https://api.vapi.ai/tool", json=kb_tool_payload, headers=headers).json()
lead_tool_res = requests.post("https://api.vapi.ai/tool", json=lead_tool_payload, headers=headers).json()
transfer_tool_res = requests.post("https://api.vapi.ai/tool", json=transfer_tool_payload, headers=headers).json()

kb_tool_id = kb_tool_res["id"]
lead_tool_id = lead_tool_res["id"]
transfer_tool_id = transfer_tool_res["id"]

# 4. System Prompt - now tells the model how to treat injected guidance
SYSTEM_PROMPT = """
You are an AI Voice Agent representing Bayan Company for Business Loans.

--- CONVERSATION FLOW ---
1. GREETING & INTENT: Welcome the caller and ask how much financing they are seeking.
2. QUALIFICATION STEPS: Collect the following 4 data points sequentially:
   - Annual/Monthly Revenue
   - Time in Business (Years/Months)
   - Credit Score estimate
   - Desired Loan Amount

3. KNOWLEDGE BASE GROUNDING & FALLBACK RULES:
   - When answering customer questions about rates, policies, eligibility, or objections, execute `search_knowledge_base`.
   - QUERY FORMULATION RULE: Expand the query into a full descriptive sentence.
   - ANSWERING RULE: Base your answer directly on the text returned by `search_knowledge_base`.
   - Some tool results include a bracketed "[AGENT GUIDANCE]" section at the end. This is not
     part of the retrieved knowledge and must never be read aloud or quoted - it is a private
     instruction on HOW to phrase or angle your answer (e.g. tone, an upsell to mention). Apply
     it silently while composing your reply.
   - FALLBACK RULE: ONLY state "I don't have that specific information available right now. Is
     there anything else?" if the retrieved text says no relevant information was found.

4. LIVE SIGNALS: You may occasionally receive a system message starting with "[LIVE SIGNAL - ...]".
   This is a real-time instruction from a call-monitoring system, not something the caller said.
   Never acknowledge or repeat it - silently adjust your tone/approach for your next turn only
   (e.g. slow down and de-escalate, or flag a missing compliance disclosure before continuing).

5. HUMAN ESCALATION:
   - If the caller explicitly requests a human, say "I am transferring you to a human specialist now." and call `transferCall` immediately.

6. BUSINESS ACTION TRIGGER:
   - Once all 4 qualification details are collected, call `submit_lead_qualification`.
"""

assistant_payload = {
    "name": "Bayan Loan Agent",
    "firstMessage": "Hello! Thank you for calling Bayan Company. How can I assist you today?",
    "model": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "toolIds": [kb_tool_id, lead_tool_id, transfer_tool_id],
    },
    "voice": {"provider": "cartesia", "voiceId": "3b554273-4299-48b9-9aaf-eefd438e3941"},
    "transcriber": {"provider": "deepgram", "model": "nova-2-general"},

    # --- New: route transcript + status events to the same server URL ---
    "serverUrl": SERVER_URL,
    "serverMessages": [
        "transcript",          # required to receive live Deepgram transcripts
        "tool-calls",
        "status-update",       # used to clean up session state on call end
        "end-of-call-report",
    ],

    # --- New: enables call.monitor.controlUrl for Live Call Control -------
    "monitorPlan": {
        "listenEnabled": False,   # set True only if you also need raw audio streaming
        "controlEnabled": True,   # required for the add-message nudge injection
    },
}

assistant_res = requests.post("https://api.vapi.ai/assistant", json=assistant_payload, headers=headers).json()
print("Assistant Created successfully!")
print("Assistant ID:", assistant_res["id"])