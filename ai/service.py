"""
AeroCast AI Assistant Service.
Direct integration with AIML API (Powered by Google Gemini 2.5 Flash).
Provides hyper-local mitigation generation, natural language urban copilot chat,
interactive 'What-If' policy simulation, and executive situation reports.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
import httpx

from config import settings

logger = logging.getLogger("aerocast.ai")


class AIAssistantService:
    """
    Service for dispatching direct LLM inference requests to AIML API
    configured with Google Gemini 2.5 Flash.
    """

    def __init__(self):
        self.api_key = settings.AIML_API_KEY
        self.base_url = settings.AIML_API_BASE_URL.rstrip("/")
        self.model = settings.AIML_MODEL
        self.timeout = 30.0

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "AIML_API_KEY is not configured. Please set AIML_API_KEY in your .env file."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Execute chat completion request directly against AIML API."""
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(
                        "AIML API Error (Status %d): %s",
                        response.status_code,
                        response.text,
                    )
                    raise RuntimeError(
                        f"AIML API returned status {response.status_code}: {response.text}"
                    )
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except httpx.RequestError as exc:
                logger.error("HTTP network failure connecting to AIML API: %s", str(exc))
                raise RuntimeError(f"Network error connecting to AIML API: {str(exc)}") from exc

    async def generate_zone_mitigation(
        self,
        zone_id: str,
        zone_name: str,
        spatial_metrics: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Generate hyper-local multi-stakeholder operational action plan for a specific zone.
        """
        system_prompt = (
            "You are AeroCast AI — Chief Urban Risk Intelligence Copilot for Lahore District. "
            "You specialize in predictive environmental crisis management, smog spike mitigation, "
            "urban heat island response, and hydrological flood control. "
            "Produce structured, complete, and tactical operational directives using rich Markdown "
            "with tables, bold key metrics, and agency-specific action cards. Never cut off or use placeholders."
        )

        lang_instruction = "Provide the entire briefing in professional English."
        if language == "ur":
            lang_instruction = "Provide the briefing in formal Urdu (اردو) with professional disaster-management terminology."
        elif language == "roman_ur":
            lang_instruction = "Provide the briefing in clear Roman Urdu (Urdu written in Latin alphabet) for field dispatch officers."

        user_prompt = f"""
Generate an Operational Mitigation Briefing for this Lahore computational zone:

ZONE IDENTIFIER: {zone_id} ({zone_name})
TELEMETRY PROFILE:
- Current PM2.5: {spatial_metrics.get('pm25_current', 'N/A')} µg/m³
- 24h Predicted PM2.5: {spatial_metrics.get('pm25_forecast_24h', 'N/A')} µg/m³
- Urban Heat Island (UHI) Index: {spatial_metrics.get('uhi_index', 'N/A')} (0.0 to 1.0)
- Flash Flood Vulnerability Score: {spatial_metrics.get('flood_score', 'N/A')} (0.0 to 1.0)
- Population Density: {spatial_metrics.get('population_density', 'N/A')} persons/km²
- DEM Elevation: {spatial_metrics.get('elevation_m', 'N/A')} m | Slope: {spatial_metrics.get('slope_pct', 'N/A')}%
- Active Satellite Fire Hotspots nearby: {spatial_metrics.get('fire_hotspots_count', 0)} detected

REQUIRED STRUCTURE (Use Markdown with Tables & Clear Sections):

### 🚨 Operational Threat Classification
> State the composite threat tier (e.g. `[CRITICAL EMERGENCY]` / `[HIGH WATCH]` / `[MODERATE]`) and a 2-sentence executive summary of the primary imminent risks.

### 📊 Zone Risk & Telemetry Baseline
| Environmental Metric | Current Observed | 24h Predictive Horizon | Threat Level | Primary Driver |
| :--- | :--- | :--- | :--- | :--- |

### 📋 Multi-Agency Operational Directives
* **🚓 Traffic Police & Punjab EPA**:
  - Point 1: Specific heavy vehicle curfews, corridor diversions, or odd-even enforcement.
  - Point 2: Anti-smog water misting cannons deployment schedule on major arteries.
* **🚰 WASA & Municipal Services**:
  - Point 1: Preventive drain desilting and low-lying basin pre-clearing.
  - Point 2: Heavy dewatering mobile pump positioning in identified depression spots.
* **🚑 Rescue 1122 & Emergency Healthcare**:
  - Point 1: Pre-positioning mobile respiratory triage units and oxygen reserve readiness.
  - Point 2: Shaded misting/cooling stations if UHI index is elevated.

### 🛡️ Public Health Advisory
- **Vulnerable Citizens (Asthma / Elderly / Children)**: Specific protective steps.
- **Schools & Outdoor Workers**: Recommended hours of operation and mask mandates.

### 🎯 Projected Mitigation Impact
| Inter-Agency Intervention | Target Reduction | Expected Timeframe | Projected Outcome |
| :--- | :--- | :--- | :--- |

{lang_instruction}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        ai_response = await self._call_llm(messages, temperature=0.3, max_tokens=3500)
        return {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "model_used": self.model,
            "language": language,
            "mitigation_plan": ai_response,
        }

    async def ask_copilot(
        self,
        query: str,
        context_summary: Optional[Dict[str, Any]] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Process natural language queries from administrators or citizens
        grounded in the Lahore 241-zone dataset.
        """
        system_prompt = (
            "You are AeroCast AI Urban Risk Copilot — Chief Environmental Intelligence Officer for Lahore District. "
            "You have direct real-time access to the live computational telemetry and predictive models covering all 241 Lahore zones.\n\n"
            "CRITICAL RULES & METRIC SCALES:\n"
            "1. NEVER say 'As an AI I don't have real-time data' or use placeholders.\n"
            "2. Air Quality (PM2.5): Measured in µg/m³ (WHO Moderate <= 35, Unhealthy 55-150, Very Unhealthy 150-250, Hazardous > 250).\n"
            "3. Flash Flood Runoff Risk Score: Always on a continuous 0.00 to 1.00 scale (0.00-0.25 Low/GREEN, 0.25-0.50 Moderate/YELLOW, 0.50-0.75 High/ORANGE, 0.75-1.00 Severe/RED). NEVER invent or scale to 0-10!\n"
            "4. Urban Heat Island (UHI) Index: Continuous 0.00 to 1.00 scale (0.00-0.30 Low, 0.30-0.55 Moderate, 0.55-0.75 High, >0.75 Severe).\n"
            "5. ALWAYS quote the exact zone IDs and exact values from the provided ACTIVE LIVE DISTRICT CONTEXT.\n"
            "6. Format answers using rich Markdown with markdown tables, clear bullet points, bold key values, and agency-specific operational directives (WASA, Traffic Police, EPA, Rescue 1122)."
        )

        context_str = ""
        if context_summary:
            context_str = f"\nACTIVE LIVE DISTRICT CONTEXT & 24-HOUR HAZARD PREDICTIONS:\n{json.dumps(context_summary, indent=2)}\n"

        lang_instruction = "Respond in clear, professional English."
        if language == "ur":
            lang_instruction = "Respond in formal, complete Urdu (اردو)."
        elif language == "roman_ur":
            lang_instruction = "Respond in clear, complete Roman Urdu."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{context_str}\nUSER INQUIRY: {query}\n\n{lang_instruction}\nProvide a complete, detailed, and structured response:",
            },
        ]

        response = await self._call_llm(messages, temperature=0.3, max_tokens=3500)
        return {
            "query": query,
            "model_used": self.model,
            "response": response,
        }

    async def simulate_policy(
        self,
        interventions: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate 'What-If' policy intervention scenarios using Gemini 2.5 Flash.
        """
        system_prompt = (
            "You are AeroCast Environmental Policy Simulation Engine for Lahore District. "
            "Quantify the multi-hazard environmental, public health, and operational impacts of proposed urban policy interventions. "
            "Deliver an executive-level simulation report with comparison tables, domain-specific impact analysis, "
            "and operational feasibility evaluations."
        )

        user_prompt = f"""
Simulate and evaluate the outcome of the following proposed policy interventions for Lahore District:

CURRENT DISTRICT BASELINE:
- Mean PM2.5: {baseline_metrics.get('mean_pm25', 185.0)} µg/m³
- Peak 24h Forecast PM2.5: {baseline_metrics.get('peak_pm25', 380.0)} µg/m³
- Mean Heat Island Anomaly: +{baseline_metrics.get('mean_uhi', 2.8)}°C
- Active Zones under High Risk: {baseline_metrics.get('high_risk_zones', 48)} / 241

PROPOSED INTERVENTION PARAMETERS:
- Odd-Even Vehicle Rationing: {interventions.get('traffic_reduction_pct', 0)}% traffic volume reduction
- Heavy Diesel Night Curfew: {'Enforced (22:00-06:00)' if interventions.get('heavy_diesel_ban') else 'Disabled'}
- Anti-Smog Water Misting Cannons: {interventions.get('water_cannons_deployed', 0)} units deployed on major corridors
- Industrial Boiler Scrubber Compliance: {interventions.get('industrial_clampdown_pct', 0)}% enforcement
- WASA Preventive Drain Pre-Clearing: {'Active pre-monsoon clearing' if interventions.get('drain_preclearing') else 'Disabled'}

TASK:
Produce an Executive Policy Simulation Evaluation formatted in clean, rich Markdown:

### 📈 Executive Impact Scorecard
| Policy Indicator | Baseline Status | Simulated Outcome | Expected Delta | Reliability Score |
| :--- | :--- | :--- | :--- | :--- |
(Include District Mean PM2.5, Peak PM2.5, Critical Threat Zones, Estimated Hospital ER Avoidance %, and WASA Waterlogging Drainage Speed)

### 🔍 Sector-by-Sector Impact Analysis
1. **🌫️ Atmospheric Smog & Particulate Dynamics**:
   - Explain how vehicular reduction and scrubber compliance suppress PM2.5 boundary-layer trapping.
2. **🌡️ Urban Heat Island & Thermal Inversion Relief**:
   - Assess impact of misting cannons and reduced engine thermal emissions on local surface temperatures.
3. **🌊 Hydrological Runoff & Drainage De-Bottlenecking**:
   - Assess how WASA drain pre-clearing alleviates flash flooding risks in low-lying zones.
4. **🏥 Public Health & Emergency Healthcare Capacity**:
   - Estimated reduction in acute respiratory and cardiovascular emergency admissions over a 72-hour window.

### ⚙️ Operational Feasibility & Agency Friction Matrix
| Implementing Agency | Key Responsibilities | Implementation Friction | Feasibility Rating |
| :--- | :--- | :--- | :--- |
(Cover Traffic Police, Punjab EPA, WASA, Rescue 1122 with High/Medium/Low Feasibility and primary operational bottlenecks)

### 💡 Strategic Policy Recommendation
> Summary verdict (e.g. `[HIGHLY RECOMMENDED]` / `[CONDITIONALLY EFFECTIVE]`) with the most cost-effective sequencing of interventions for the Deputy Commissioner.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        ai_response = await self._call_llm(messages, temperature=0.2, max_tokens=2000)
        return {
            "interventions": interventions,
            "model_used": self.model,
            "simulation_report": ai_response,
        }

    async def generate_situation_report(
        self,
        district_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an Executive District Daily Situation Report (DSR)
        formatted for the Deputy Commissioner Lahore and Director General PDMA.
        """
        system_prompt = (
            "You are the Chief Intelligence Officer of AeroCast — the Urban Risk Platform for Lahore District. "
            "Generate a formal, authoritative, and complete Executive Daily Situation Report (DSR) for Lahore District leadership. "
            "Combine real-time sensor telemetry, 24-hour predictive machine learning forecasts, "
            "NASA satellite fire hotspots, and multi-agency operational directives. "
            "CRITICAL RULES:\n"
            "1. Output complete, valid Markdown tables with clean headers and concise separator rows (|:---|:---|).\n"
            "2. NEVER pad table delimiter rows with repetitive dashes (use exactly 3 dashes: |:---|:---|:---|).\n"
            "3. Ensure the report is fully written and covers all sections without stopping midway."
        )

        user_prompt = f"""
Generate an official Executive Daily Situation Report (DSR) based strictly on the following live district telemetry:

LIVE DISTRICT TELEMETRY DATA:
{json.dumps(district_summary, indent=2)}

REQUIRED REPORT STRUCTURE:

# 🏛️ DISTRICT ENVIRONMENTAL SITUATION REPORT (DSR)
**Security Classification:** OFFICIAL / SENSITIVE  
**Issued To:** Deputy Commissioner Lahore & Director General PDMA Punjab  
**Monitoring Horizon:** Next 24 Hours  
**Overall Threat Level:** `[HIGH RISK - SMOG & THERMAL ADVISORY]`

---

### 📊 1. District Telemetry & Hazard Dashboard
| Indicator | Observed / Forecasted Value | Standard Threshold | Operational Status |
| :--- | :--- | :--- | :--- |
(Fill with:
- Computational Coverage: 241 Metric Zones (100% District)
- District Mean PM2.5 (from mean_pm25_ugm3)
- Peak 24h Advance PM2.5 Forecast (from peak_pm25_forecast_ugm3 and peak_zone)
- Active Smog Crisis Zones (from active_smog_emergency_zones)
- Peak Flash Flood Vulnerability (from peak_flood_score and peak_flood_zone)
- Active Satellite Fire Hotspots (from firms_fire_hotspots_detected)
Use the EXACT numbers from the LIVE DISTRICT TELEMETRY DATA above.)

---

### 📌 2. Strategic Hazard Assessment
- Analyze boundary layer thermal inversion dynamics, particulate stagnation, and wind dispersion conditions.
- Analyze satellite fire hotspots trajectory and transboundary stubble smoke transport.
- Analyze urban heat island thermal inertia and localized drainage risks.

---

### 📍 3. Priority Vulnerable Zones & Action Matrix
| Zone Identifier | Hazard Type | Metric Value | Risk Tier | Tactical Field Directive |
| :--- | :--- | :--- | :--- | :--- |
(Populate with the top priority zones from the telemetry data with specific tactical field directives)

---

### ⚡ 4. Departmental Operational Orders
- **🚰 WASA (Water and Sanitation Agency):** Dewatering pump pre-positioning, trunk drain clearing.
- **🚓 City Traffic Police & Punjab EPA:** Diesel emission curfews, misting cannon deployment, kiln enforcement.
- **🚑 Rescue 1122 & District Health Authority:** Emergency triage readiness, vulnerable citizen advisories.

---

### 🔮 5. 24-Hour Synoptic Outlook & Recommendations
> Executive synthesis and high-priority recommendations for the Deputy Commissioner.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        ai_response = await self._call_llm(messages, temperature=0.3, max_tokens=4096)
        
        # Safety clean: Compress any runaway dash sequences
        ai_response = re.sub(r':?-{4,}:?', ':---', ai_response)
        ai_response = re.sub(r'-{4,}', '---', ai_response)

        return {
            "model_used": self.model,
            "situation_report": ai_response,
        }


# Global singleton instance
ai_service = AIAssistantService()
