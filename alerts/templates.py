"""
AeroCast Multi-Lingual Alert Message Templates.
Generates localized early warning messages in English, Urdu (اردو), and Roman Urdu.
"""

from typing import Dict, Any
from .models import HazardType, SeverityLevel


def format_smog_alert(
    zone_id: str,
    zone_name: str,
    forecast_pm25: float,
    current_pm25: float,
    ci_range: list,
    severity: SeverityLevel,
) -> Dict[str, str]:
    """Generate multi-lingual smog early warning messages."""
    p_val = round(forecast_pm25, 1)
    c_val = round(current_pm25, 1)

    if severity == SeverityLevel.EMERGENCY:
        return {
            "en": (
                f"[AEROCAST HAZARD ALERT] SEVERE SMOG SPIKE predicted for {zone_id} ({zone_name}). "
                f"24h PM2.5 forecast: {p_val} µg/m³ (80% CI: {ci_range[0]}-{ci_range[1]}). "
                f"Air Quality is HAZARDOUS. N95 masks mandatory. Outdoor activity suspended."
            ),
            "ur": (
                f"【ایرو کاسٹ ایمرجنسی الرٹ】 زون {zone_id} ({zone_name}) میں شدید اسموگ کی پیشگوئی! "
                f"آئندہ 24 گھنٹوں میں پی ایم 2.5 کا تخمینہ: {p_val} مائیکرو گرام ہے۔ "
                f"شہری غیر ضروری طور پر باہر نہ نکلیں اور N95 ماسک لازمی استعمال کریں۔ ہیلپ لائن: 1122"
            ),
            "roman_ur": (
                f"[AEROCAST ALERT] Zone {zone_id} ({zone_name}) mein aglay 24 ghantay mein SHADEED SMOG ({p_val} ug/m3) ki peshgoi hai. "
                f"Hawa shadeed muzir-e-sehat hai. Bahir jatay waqt N95 mask lazmi pehnein aur khuli jagah warzish se guraiz karein."
            ),
        }
    else:  # WATCH / WARNING
        return {
            "en": (
                f"[AEROCAST ADVISORY] High Pollution spike forecasted for {zone_id} ({zone_name}). "
                f"24h PM2.5: {p_val} µg/m³ (Current: {c_val} µg/m³). "
                f"Sensitive groups, elderly, and children should limit prolonged outdoor exertion."
            ),
            "ur": (
                f"【ایرو کاسٹ الرٹ】 زون {zone_id} ({zone_name}) میں فضائی آلودگی میں اضافے کا خدشہ۔ "
                f"متوقع پی ایم 2.5: {p_val} مائیکرو گرام۔ "
                f"سانس کے مریض، بزرگ اور بچے کھلی فضا میں سرگرمیوں سے پرہیز کریں۔"
            ),
            "roman_ur": (
                f"[AEROCAST ADVISORY] Zone {zone_id} ({zone_name}) mein kal PM2.5 barh kar {p_val} ug/m3 honay ka imkan hai. "
                f"Dama aur saans ke mareez ehtiyat karein aur outdoor activities mehdood rakhein."
            ),
        }


def format_flood_alert(
    zone_id: str,
    zone_name: str,
    flood_score: float,
    precip_mm: float,
    inundation_desc: str,
    severity: SeverityLevel,
) -> Dict[str, str]:
    """Generate multi-lingual flash flood and urban waterlogging alert messages."""
    f_score = round(flood_score, 2)
    p_mm = round(precip_mm, 1)

    if severity == SeverityLevel.EMERGENCY:
        return {
            "en": (
                f"[AEROCAST CRITICAL ALERT] FLASH FLOOD & WATERLOGGING WARNING for {zone_id} ({zone_name}). "
                f"Runoff Risk Index: {f_score}/1.00 (Rainfall: {p_mm} mm). "
                f"Expected Inundation: {inundation_desc}. WASA Emergency Dewatering & Suction Pumps Activated."
            ),
            "ur": (
                f"【ایرو کاسٹ سیلاب الرٹ】 زون {zone_id} ({zone_name}) میں شدید اربن فلڈنگ اور پانی جمع ہونے کا خطرہ! "
                f"متوقع بارش: {p_mm} ملی میٹر، خطرے کا اسکور: {f_score}۔ "
                f"نشیبی راستوں اور انڈر پاسز سے گریز کریں۔ واسا کنٹرول روم: 1334 پر رابطہ کریں۔"
            ),
            "roman_ur": (
                f"[AEROCAST FLOOD ALERT] Zone {zone_id} ({zone_name}) mein shadeed barish ({p_mm} mm) ke baes arban flooding ka khatra! "
                f"Inundation: {inundation_desc}. Low-lying underpasses aur unpaved sarak par safar se guraiz karein. WASA Helpline: 1334."
            ),
        }
    else:  # WATCH / ADVISORY
        return {
            "en": (
                f"[AEROCAST WATCH] Localized waterlogging watch for {zone_id} ({zone_name}). "
                f"Runoff Risk: {f_score}/1.00. Expect minor street ponding and traffic slowdowns."
            ),
            "ur": (
                f"【ایرو کاسٹ انتباہ】 زون {zone_id} ({zone_name}) میں بارش کے باعث سڑکوں پر پانی جمع ہونے کا امکان۔ احتیاط سے ڈرائیو کریں۔"
            ),
            "roman_ur": (
                f"[AEROCAST WATCH] Zone {zone_id} ({zone_name}) mein barish ke baes sarak par pani jama honay ka imkan. Traffic delays mutawaqqo hain."
            ),
        }


def format_heat_alert(
    zone_id: str,
    zone_name: str,
    uhi_score: float,
    temp_c: float,
    severity: SeverityLevel,
) -> Dict[str, str]:
    """Generate multi-lingual Urban Heat Island / extreme heat alert messages."""
    u_score = round(uhi_score, 2)
    t_val = round(temp_c, 1)

    return {
        "en": (
            f"[AEROCAST HEAT WARNING] Severe Urban Heat Island anomaly in {zone_id} ({zone_name}). "
            f"UHI Vulnerability: {u_score}/1.00 (Temp: {t_val}°C). "
            f"Stay hydrated, avoid direct peak sun exposure (11am-4pm), seek shaded cooling centers."
        ),
        "ur": (
            f"【ایرو کاسٹ ہیٹ ویو الرٹ】 زون {zone_id} ({zone_name}) میں شدید اربن ہیٹ آئی لینڈ اور گرمی کی شدت! "
            f"درجہ حرارت: {t_val} ڈگری سینٹی گریڈ۔ "
            f"پانی کا کثرت سے استعمال کریں اور دن 11 سے 4 بجے تک دھوپ میں بلا ضرورت نکلنے سے پرہیز کریں۔"
        ),
        "roman_ur": (
            f"[AEROCAST HEAT ALERT] Zone {zone_id} ({zone_name}) mein shiddat-e-hararat aur Urban Heat Island anomaly ({u_score}/1.00, {t_val} C). "
            f"Pani zyada piyein aur dhoop se bachein."
        ),
    }
