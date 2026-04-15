"""
HydroCast — Remedy & Precaution Engine
Maps model predictions to WHO-standard treatment protocols,
WASH precautions, community actions, and government protocols.
Customises advice based on district sanitation profile.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DISEASES, RISK_THRESHOLDS, EMERGENCY_CONTACTS

logger = logging.getLogger("hydrocast.remedy_engine")

# ══════════════════════════════════════════════════════════════
# WHO-STANDARD REMEDY DATABASE
# ══════════════════════════════════════════════════════════════

REMEDY_DATABASE: dict = {
    "Cholera": {
        "emergency": [
            {"text":"Activate District Rapid Response Team (DRRT) immediately","priority":"CRITICAL","timeframe":"0–2 hrs"},
            {"text":"Deploy ORS sachets to all PHCs within 6 hours","priority":"CRITICAL","timeframe":"0–6 hrs"},
            {"text":"Issue boil-water advisory via SMS broadcast and loudspeaker vans","priority":"CRITICAL","timeframe":"0–6 hrs"},
            {"text":"Set up Cholera Treatment Centres (CTCs) with IV fluid stocks","priority":"HIGH","timeframe":"6–12 hrs"},
            {"text":"Notify state IDSP control room and block medical officers","priority":"HIGH","timeframe":"0–4 hrs"},
        ],
        "medical": [
            {"drug":"ORS (WHO low-osmolarity)","dose":"200–400 mL per loose stool","route":"Oral","duration":"Until diarrhoea stops","notes":"Primary treatment — start immediately","who_reference":"WHO 2023 Cholera Guidelines"},
            {"drug":"Ringer's Lactate IV","dose":"100 mL/kg in first 3–4 hours","route":"Intravenous","duration":"Severe dehydration only","notes":"Signs: sunken eyes, unable to drink, skin pinch slow","who_reference":"WHO 2023"},
            {"drug":"Doxycycline","dose":"300 mg single dose (adults)","route":"Oral","duration":"Single dose","notes":"First-line antibiotic; reduces duration by 50%","who_reference":"WHO 2023"},
            {"drug":"Azithromycin","dose":"1 g single dose (adults); 20 mg/kg children","route":"Oral","duration":"Single dose","notes":"Use if Doxycycline contraindicated (pregnancy, children <8yr)","who_reference":"WHO 2023"},
            {"drug":"Zinc","dose":"20 mg/day for children under 5","route":"Oral","duration":"10 days","notes":"Reduces duration and severity; do not skip","who_reference":"WHO/UNICEF 2004"},
        ],
        "wash": [
            {"action":"Chlorinate all drinking water sources","target":"0.5 mg/L free residual chlorine at consumer end","priority":"CRITICAL"},
            {"action":"Flush and disinfect suspected contaminated pipelines","target":"Bleach solution 1%","priority":"CRITICAL"},
            {"action":"Collect water samples from 10 sites for lab testing","target":"Report within 24 hrs","priority":"HIGH"},
            {"action":"Seal identified contaminated wells; provide water tankers","target":"All affected wards","priority":"HIGH"},
            {"action":"Distribute water purification tablets (NaOCl) to households","target":"All households in affected area","priority":"HIGH"},
            {"action":"Repair broken sewer-water pipe crossovers","target":"PHE department within 48 hrs","priority":"MEDIUM"},
        ],
        "precaution": [
            {"advisory":"Boil drinking water for minimum 1 minute before consumption","audience":"All residents","priority":"CRITICAL"},
            {"advisory":"Wash hands with soap before eating and after toilet use","audience":"All residents","priority":"HIGH"},
            {"advisory":"Avoid raw vegetables, salads, uncooked seafood","audience":"All residents","priority":"HIGH"},
            {"advisory":"Use only sealed packaged or boiled water for ice and beverages","audience":"All residents","priority":"HIGH"},
            {"advisory":"Report rice-water diarrhoea immediately to nearest PHC","audience":"All residents","priority":"HIGH"},
            {"advisory":"Avoid open defecation — use latrines/toilets at all times","audience":"Rural residents","priority":"HIGH"},
        ],
        "community": [
            {"action":"ASHA workers: door-to-door ORS distribution and surveillance","actor":"ASHA workers","priority":"CRITICAL"},
            {"action":"Anganwadi centres: monitor under-5 children daily","actor":"Anganwadi workers","priority":"HIGH"},
            {"action":"Gram Panchayat: organise community chlorination of water sources","actor":"GP members","priority":"HIGH"},
            {"action":"Set up Oral Rehydration Therapy (ORT) corners at Anganwadis","actor":"ANM/ASHA","priority":"HIGH"},
            {"action":"Community volunteers: disinfect public spaces with bleach weekly","actor":"SHG volunteers","priority":"MEDIUM"},
        ],
        "government": [
            {"action":"Report to IDSP as Cholera Outbreak — mandatory within 24 hrs","authority":"CMO/State IDSP","legal_basis":"Epidemic Diseases Act 1897"},
            {"action":"DM to chair District Health Emergency Meeting within 12 hrs","authority":"District Magistrate","legal_basis":"NDMA Guidelines"},
            {"action":"Release funds from District Contingency Fund","authority":"DM/CEO ZP","legal_basis":"SDRF norms"},
            {"action":"Coordinate with NHM for emergency drug stocks","authority":"DHO","legal_basis":"NHM guidelines"},
            {"action":"Invoke Section 144 CrPC near contaminated water sources if needed","authority":"District Magistrate","legal_basis":"CrPC 1973"},
        ],
    },
    "Typhoid": {
        "emergency": [
            {"text":"Alert CMO and activate Block Rapid Response Teams","priority":"CRITICAL","timeframe":"0–4 hrs"},
            {"text":"Deploy rapid diagnostic test kits (Widal/Typhidot) to district labs","priority":"CRITICAL","timeframe":"0–6 hrs"},
            {"text":"Issue food-safety advisory; inspect all street food vendors","priority":"HIGH","timeframe":"6–24 hrs"},
            {"text":"Identify index cases; trace contacts in 10-household radius","priority":"HIGH","timeframe":"24–48 hrs"},
        ],
        "medical": [
            {"drug":"Azithromycin","dose":"10–20 mg/kg/day","route":"Oral","duration":"5 days","notes":"WHO first-line for uncomplicated typhoid in India","who_reference":"WHO 2018 Typhoid Guidelines"},
            {"drug":"Ceftriaxone","dose":"50–75 mg/kg/day IV","route":"Intravenous","duration":"10–14 days","notes":"Use for severe or complicated typhoid","who_reference":"WHO 2018"},
            {"drug":"Fluoroquinolones (Ciprofloxacin)","dose":"—","route":"—","duration":"—","notes":"AVOID — high resistance rates in India (NARST 2023 data). Do not prescribe.","who_reference":"NARST India 2023"},
            {"drug":"Paracetamol","dose":"15 mg/kg every 6 hrs","route":"Oral/IV","duration":"As needed","notes":"Fever management only — avoid NSAIDs (GI perforation risk)","who_reference":"Standard care"},
            {"drug":"Typbar-TCV Vaccine","dose":"0.5 mL single dose","route":"IM","duration":"Single dose","notes":"Reactive vaccination for household contacts and at-risk population","who_reference":"WHO 2018"},
        ],
        "wash": [
            {"action":"Test piped water supply for S. Typhi contamination","target":"Source + endpoint","priority":"CRITICAL"},
            {"action":"Hyperchlorinate water distribution network","target":"1 mg/L free residual chlorine at consumer end","priority":"HIGH"},
            {"action":"Identify and seal leaking sewer-water pipe crossovers","target":"PHE department within 24 hrs","priority":"CRITICAL"},
            {"action":"Inspect and certify food handlers at hotels, dhabas, canteens","target":"Food Safety Officer","priority":"HIGH"},
            {"action":"Promote dedicated handwashing stations at food establishments","target":"All eateries in affected area","priority":"MEDIUM"},
        ],
        "precaution": [
            {"advisory":"Drink only boiled or bottled water; avoid ice from unknown sources","audience":"All residents","priority":"HIGH"},
            {"advisory":"Eat fully cooked food; avoid raw fruits and vegetables unless peeled","audience":"All residents","priority":"HIGH"},
            {"advisory":"Do not prepare food for others if you have been diagnosed with typhoid","audience":"Food handlers","priority":"CRITICAL"},
            {"advisory":"Wash hands with soap for minimum 20 seconds","audience":"All residents","priority":"HIGH"},
            {"advisory":"Get Typbar-TCV vaccine if not vaccinated in past 3 years","audience":"Household contacts","priority":"HIGH"},
        ],
        "community": [
            {"action":"Mass awareness campaign on typhoid symptoms and prevention","actor":"ASHA/ANM","priority":"HIGH"},
            {"action":"Organise reactive Typbar-TCV vaccination camp for under-15 population","actor":"PHC staff","priority":"HIGH"},
            {"action":"Monitor school children daily for persistent fever (>38.5°C for 3 days)","actor":"School health team","priority":"MEDIUM"},
            {"action":"Sanitation survey to map open defecation hotspots","actor":"Village health committee","priority":"MEDIUM"},
        ],
        "government": [
            {"action":"Report Typhoid Cluster to state IDSP within 48 hrs of detection","authority":"CMO","legal_basis":"IDSP Protocol"},
            {"action":"Procure Typbar-TCV vaccine from cold chain facility","authority":"DHO","legal_basis":"NHM guidelines"},
            {"action":"Engage municipal water authority to urgently repair sewer-pipe crossovers","authority":"PHE/Municipality","legal_basis":"Maharashtra Municipal Corporation Act"},
            {"action":"Activate Food Safety Officer to inspect and shut non-compliant establishments","authority":"FSSAI/Food Safety Officer","legal_basis":"Food Safety Act 2006"},
        ],
    },
    "ADD": {
        "emergency": [
            {"text":"Deploy ORS sachets to all healthcare facilities and ASHA workers","priority":"CRITICAL","timeframe":"0–6 hrs"},
            {"text":"Set up Oral Rehydration Corners (ORCs) at all PHCs and sub-centres","priority":"HIGH","timeframe":"0–12 hrs"},
            {"text":"Identify cases with bloody diarrhoea (possible Shigella/E.coli)","priority":"HIGH","timeframe":"0–24 hrs"},
            {"text":"Alert Food Safety team for market and vendor inspections","priority":"HIGH","timeframe":"0–24 hrs"},
        ],
        "medical": [
            {"drug":"ORS (WHO low-osmolarity)","dose":"200–400 mL per loose stool","route":"Oral","duration":"Until diarrhoea stops","notes":"First-line for ALL cases regardless of age","who_reference":"WHO 2023"},
            {"drug":"Ringer's Lactate IV","dose":"Based on dehydration severity","route":"Intravenous","duration":"Until oral rehydration tolerated","notes":"Only for severe dehydration","who_reference":"WHO 2023"},
            {"drug":"Zinc","dose":"20 mg/day for children under 5","route":"Oral","duration":"10 days","notes":"Reduces duration by 25% and future recurrence","who_reference":"WHO/UNICEF"},
            {"drug":"Antibiotics","dose":"—","route":"—","duration":"—","notes":"Do NOT give antibiotics for watery diarrhoea unless Cholera/Shigella confirmed by culture","who_reference":"WHO 2023"},
            {"drug":"Metronidazole","dose":"400 mg 3× daily (adults)","route":"Oral","duration":"5–7 days","notes":"Only if amoebic dysentery suspected (persistent bloody stool)","who_reference":"WHO 2023"},
        ],
        "wash": [
            {"action":"Test and chlorinate all drinking water sources","target":"0.5 mg/L free residual chlorine","priority":"HIGH"},
            {"action":"Ensure safe disposal of patient faeces using lime disinfection","target":"All PHCs and households","priority":"HIGH"},
            {"action":"Remove solid waste near water bodies post-rainfall/flood","target":"Municipalities within 48 hrs","priority":"HIGH"},
            {"action":"Distribute water purification tablets to high-density areas","target":"Urban slums and crowded localities","priority":"MEDIUM"},
        ],
        "precaution": [
            {"advisory":"Boil water and store in clean covered containers","audience":"All residents","priority":"HIGH"},
            {"advisory":"Wash hands with soap before meals and after toilet visits","audience":"All residents","priority":"HIGH"},
            {"advisory":"Avoid buying food from vendors with no handwashing facility","audience":"All residents","priority":"MEDIUM"},
            {"advisory":"Continue breastfeeding during diarrhoea — do not stop for infants","audience":"Mothers of infants","priority":"HIGH"},
            {"advisory":"Seek medical care if diarrhoea lasts more than 3 days or blood appears","audience":"All residents","priority":"CRITICAL"},
        ],
        "community": [
            {"action":"ASHA workers: demonstrate correct ORS preparation at household level","actor":"ASHA workers","priority":"HIGH"},
            {"action":"Monitor children under 5 daily in flood-affected areas","actor":"ASHA/ANM","priority":"HIGH"},
            {"action":"Set up community ORT corners in schools and Anganwadis","actor":"ANM/ASHA","priority":"MEDIUM"},
            {"action":"Conduct handwashing demonstrations at water collection points","actor":"SHG/community","priority":"MEDIUM"},
        ],
        "government": [
            {"action":"Classify as ADD Outbreak if >3 cases from same area in 7 days — report to IDSP","authority":"CMO/DSO","legal_basis":"IDSP Protocol"},
            {"action":"Coordinate with PHE for emergency water supply and chlorination","authority":"DHO/PHE","legal_basis":"Maharashtra Water Supply Act"},
            {"action":"Ensure adequate ORS stock: district → block → PHC → ASHA","authority":"DHO","legal_basis":"NHM guidelines"},
            {"action":"Deploy district surveillance team for active case search","authority":"District Surveillance Officer","legal_basis":"IDSP Protocol"},
        ],
    },
}

# Response timeline phases
RESPONSE_TIMELINE: dict[str, list[dict]] = {
    "critical": [
        {"phase":"0–6 HRS",   "title":"🚨 Immediate",   "actions":"Activate DRRT · Deploy ORS · Boil-water advisory · Notify IDSP"},
        {"phase":"6–48 HRS",  "title":"⚡ Short-term",  "actions":"Water testing · Chlorination · Case isolation · Contact tracing"},
        {"phase":"3–14 DAYS", "title":"📋 Medium-term", "actions":"Vaccination drive · WASH infrastructure repair · Surveillance ↑"},
        {"phase":"2–4 WKS",   "title":"✅ Long-term",   "actions":"Infrastructure fix · Community training · Debrief & review"},
    ],
    "high": [
        {"phase":"0–24 HRS",  "title":"⚡ Alert",       "actions":"Notify CMO · Pre-position ORS · Activate surveillance"},
        {"phase":"1–7 DAYS",  "title":"📋 Preventive",  "actions":"Water testing · WASH inspection · Community advisory"},
        {"phase":"1–4 WKS",   "title":"🔬 Monitor",     "actions":"Enhanced surveillance · Resource pre-positioning"},
        {"phase":"Ongoing",   "title":"✅ Sustain",      "actions":"Maintain surveillance · Community engagement"},
    ],
    "medium": [
        {"phase":"Week 1",    "title":"👁 Watch",        "actions":"Passive surveillance · Stock check · Water quality monitoring"},
        {"phase":"Week 2–3",  "title":"📊 Assess",       "actions":"Trend analysis · Community awareness"},
        {"phase":"Week 4+",   "title":"🔄 Review",       "actions":"Monthly review · Update risk score"},
        {"phase":"Ongoing",   "title":"📝 Document",     "actions":"Baseline documentation · Preparedness planning"},
    ],
}


# ══════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class TimelineStep:
    phase:   str
    title:   str
    actions: str

@dataclass
class RemedyPlan:
    district:            str
    disease:             str
    risk_score:          float
    risk_level:          str
    ai_recommendation:   str
    emergency_actions:   list = field(default_factory=list)
    medical_protocols:   list = field(default_factory=list)
    wash_actions:        list = field(default_factory=list)
    precautions:         list = field(default_factory=list)
    community_actions:   list = field(default_factory=list)
    government_protocol: list = field(default_factory=list)
    timeline:            list = field(default_factory=list)
    emergency_contacts:  dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# REMEDY ENGINE
# ══════════════════════════════════════════════════════════════════

class RemedyEngine:
    """
    Generates district-specific WHO-standard remedy and precaution plans.

    Customises advice based on:
      - Predicted disease (Cholera / Typhoid / ADD)
      - Risk score (critical / high / medium / low)
      - District sanitation coverage (from NFHS-5)
      - Rainfall anomaly (from IMD)
    """

    def __init__(self) -> None:
        self.db = REMEDY_DATABASE
        logger.info("RemedyEngine initialised with WHO-standard protocols.")

    def _determine_risk_level(self, score: float) -> str:
        for level, thr in [("critical", 0.8), ("high", 0.6), ("medium", 0.4)]:
            if score >= thr:
                return level
        return "low"

    def generate_plan(
        self,
        district:            str,
        disease:             str,
        risk_score:          float,
        district_sanitation: float = 0.5,
        rainfall_anomaly:    float = 0.0,
    ) -> RemedyPlan:
        """
        Generate a full RemedyPlan for a district outbreak prediction.

        Parameters
        ----------
        district            : district name
        disease             : one of DISEASES
        risk_score          : model output probability (0–1)
        district_sanitation : NFHS-5 sanitation coverage (0–1)
        rainfall_anomaly    : IMD rainfall % anomaly

        Returns
        -------
        RemedyPlan  dataclass
        """
        risk_level = self._determine_risk_level(risk_score)
        protocols  = self.db.get(disease, self.db["ADD"])

        # ── Prioritise WASH actions based on sanitation level
        wash = list(protocols["wash"])
        if district_sanitation < 0.4:
            wash.insert(0, {
                "action": f"URGENT: {district} has critical sanitation deficit "
                          f"({district_sanitation:.0%} coverage). "
                          "Prioritise toilet construction and OD-free declaration.",
                "target": "Gram Panchayat / Swachh Bharat Mission",
                "priority": "CRITICAL",
            })

        # ── Emphasise monsoon messaging if rainfall is anomalous
        precautions = list(protocols["precaution"])
        if rainfall_anomaly > 100:
            precautions.insert(0, {
                "advisory": f"MONSOON ALERT: Rainfall {rainfall_anomaly:.0f}% above normal. "
                            "Avoid contact with floodwater. Do not use flooded wells.",
                "audience": "All residents",
                "priority": "CRITICAL",
            })

        recommendation = self.generate_ai_recommendation_text(
            district, disease, risk_score, risk_level,
            district_sanitation, rainfall_anomaly,
        )

        timeline = [
            TimelineStep(**step)
            for step in RESPONSE_TIMELINE.get(risk_level, RESPONSE_TIMELINE["medium"])
        ]

        return RemedyPlan(
            district            = district,
            disease             = disease,
            risk_score          = risk_score,
            risk_level          = risk_level,
            ai_recommendation   = recommendation,
            emergency_actions   = protocols["emergency"],
            medical_protocols   = protocols["medical"],
            wash_actions        = wash,
            precautions         = precautions,
            community_actions   = protocols["community"],
            government_protocol = protocols["government"],
            timeline            = timeline,
            emergency_contacts  = self.get_emergency_contacts(district),
        )

    def generate_ai_recommendation_text(
        self,
        district:   str,
        disease:    str,
        risk_score: float,
        risk_level: str,
        sanitation: float = 0.5,
        rainfall:   float = 0.0,
    ) -> str:
        """Generate the AI recommendation paragraph shown in the dashboard."""
        horizon = "14 days" if risk_level in ["critical","high"] else "28 days"
        pct     = f"{risk_score:.0%}"

        drivers = []
        if rainfall > 50:
            drivers.append(f"monsoon rainfall anomaly (+{rainfall:.0f}%)")
        if sanitation < 0.5:
            drivers.append(f"low sanitation coverage ({sanitation:.0%})")
        drivers_str = " and ".join(drivers) if drivers else "epidemiological signals"

        action = {
            "critical": "Immediate multi-sector emergency response required.",
            "high":     "Heightened surveillance and preventive action strongly advised.",
            "medium":   "Monitor closely and prepare contingency supplies.",
            "low":      "Routine surveillance is sufficient at this time.",
        }.get(risk_level, "Monitor and report.")

        return (
            f"{district} district shows {risk_level.upper()} outbreak probability "
            f"({pct}) for {disease} within {horizon}. "
            f"Primary drivers identified by AI: {drivers_str}. {action}"
        )

    def get_response_timeline(self, risk_level: str) -> list[TimelineStep]:
        """Return 4-phase response timeline for a given risk level."""
        phases = RESPONSE_TIMELINE.get(risk_level, RESPONSE_TIMELINE["medium"])
        return [TimelineStep(**p) for p in phases]

    def get_emergency_contacts(self, district: str) -> dict:
        """Return emergency contact numbers for Maharashtra."""
        contacts = dict(EMERGENCY_CONTACTS)
        # District-specific contacts (representative examples)
        district_contacts = {
            "Raigad":   {"District Hospital Alibag": "02141-222244"},
            "Thane":    {"District Hospital Thane": "022-25377333"},
            "Nashik":   {"District Hospital Nashik": "0253-2577777"},
            "Pune":     {"Sassoon Hospital Pune": "020-26128000"},
            "Mumbai City":   {"BMC Health Office": "022-22620159"},
        }
        contacts.update(district_contacts.get(district, {}))
        return contacts


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = RemedyEngine()

    plan = engine.generate_plan(
        district            = "Raigad",
        disease             = "Cholera",
        risk_score          = 0.91,
        district_sanitation = 0.34,
        rainfall_anomaly    = 187.0,
    )

    print(f"\n── Remedy Plan for {plan.district} ──")
    print(f"Disease    : {plan.disease}")
    print(f"Risk level : {plan.risk_level.upper()}")
    print(f"Risk score : {plan.risk_score:.2%}")
    print(f"\nAI Recommendation:\n{plan.ai_recommendation}")
    print(f"\nEmergency actions ({len(plan.emergency_actions)}):")
    for a in plan.emergency_actions[:3]:
        print(f"  [{a['priority']}] {a['text']}")
    print(f"\nMedical protocols ({len(plan.medical_protocols)}):")
    for m in plan.medical_protocols[:2]:
        print(f"  {m['drug']}: {m['dose']} ({m['route']}) — {m['notes']}")
    print(f"\nTimeline phases:")
    for t in plan.timeline:
        print(f"  {t.phase:12s} {t.title} — {t.actions}")
    print(f"\nEmergency contacts: {list(plan.emergency_contacts.keys())[:4]}")
    print("\n✅ remedy_engine.py smoke test passed.")
