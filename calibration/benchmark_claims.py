"""
calibration/benchmark_claims.py
──────────────────────────────────
24-claim labeled benchmark for calibrating the risk-score weights in
core/verdict.py: 9 true, 9 false, 6 unverifiable, across science,
history, health, politics, technology and local/obscure categories.

Each claim maps to a target risk tier:
    true          -> Low     (0-33)
    unverifiable  -> Medium  (34-66)
    false         -> High    (67-100)
"""

from __future__ import annotations

from typing import Dict, List

EXPECTED_TIER = {"true": "Low", "unverifiable": "Medium", "false": "High"}

LABELED_CLAIMS: List[Dict[str, str]] = [
    # ── true ──
    {"text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "ground_truth": "true", "category": "science"},
    {"text": "The Apollo 11 mission landed humans on the Moon in July 1969.", "ground_truth": "true", "category": "history"},
    {"text": "According to NASA, the global average surface temperature in 2024 was 1.29 C above the pre-industrial baseline.", "ground_truth": "true", "category": "science"},
    {"text": "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London.", "ground_truth": "true", "category": "science"},
    {"text": "The Eiffel Tower was inaugurated in Paris in 1889 for the Exposition Universelle.", "ground_truth": "true", "category": "history"},
    {"text": "Mount Everest is the highest mountain peak above sea level on Earth.", "ground_truth": "true", "category": "geography"},
    {"text": "The United Nations was established in 1945 following the conclusion of World War II.", "ground_truth": "true", "category": "politics"},
    {"text": "Photosynthesis is the biological process by which green plants convert sunlight into chemical energy.", "ground_truth": "true", "category": "science"},
    {"text": "Brazil is the largest country by both surface area and population in South America.", "ground_truth": "true", "category": "geography"},
    # ── false ──
    {"text": "The Great Wall of China was built in the 20th century by the Republic of China government in 1952.", "ground_truth": "false", "category": "history"},
    {"text": "Vaccines have been proven to cause autism according to official CDC medical research.", "ground_truth": "false", "category": "health"},
    {"text": "Humans only use 10 percent of their total brain capacity during daily functioning.", "ground_truth": "false", "category": "science"},
    {"text": "The Earth is flat and surrounded by an impenetrable wall of Antarctic ice guarded by world militaries.", "ground_truth": "false", "category": "science"},
    {"text": "Drinking highly alkaline water cures type 1 diabetes within thirty days without medical insulin.", "ground_truth": "false", "category": "health"},
    {"text": "NASA staged and faked all six Apollo moon landings on an underground movie soundstage in Nevada.", "ground_truth": "false", "category": "history"},
    {"text": "Eating carrots gives humans superhuman night vision capabilities due to military radar technology.", "ground_truth": "false", "category": "health"},
    {"text": "The Titanic ocean liner was deliberately swapped with the damaged Olympic sister ship as an insurance fraud scheme.", "ground_truth": "false", "category": "history"},
    {"text": "5G wireless cellular towers emit high-frequency radiation specifically engineered to spread biological viral infections.", "ground_truth": "false", "category": "technology"},
    # ── unverifiable ──
    {"text": "The village panchayat of Kothaguda in Rangareddy district approved a Rs 4.2 crore drainage project on 12 August 2025.", "ground_truth": "unverifiable", "category": "local/obscure"},
    {"text": "A local baker in Badulla, Sri Lanka won the provincial sourdough artisan trophy on 14 March 2024.", "ground_truth": "unverifiable", "category": "local/obscure"},
    {"text": "Resident Ramesh Patel of Sector 14 Gandhinagar reported finding a 1912 silver coin under his garden driveway last Sunday.", "ground_truth": "unverifiable", "category": "local/obscure"},
    {"text": "The municipal council of San Juan Nepomuceno enacted an ordinance requiring all street lamps to be painted violet by October.", "ground_truth": "unverifiable", "category": "local/obscure"},
    {"text": "A boutique florist shop called Petals & Thorns in Ballarat closed yesterday after 37 days of continuous operation.", "ground_truth": "unverifiable", "category": "local/obscure"},
    {"text": "High school teacher Deborah Higgins of Oamaru completed a 1000-piece wooden puzzle of Lake Tekapo in under four hours.", "ground_truth": "unverifiable", "category": "local/obscure"},
]
