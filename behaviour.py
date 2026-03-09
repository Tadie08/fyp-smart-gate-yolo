import sqlite3
import json
from datetime import datetime, timedelta
from collections import Counter
import math

DB_PATH = "behaviour.db"

# ─────────────────────────────────────────
# 1. DATABASE SETUP
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Every single entry gets logged here
    c.execute('''CREATE TABLE IF NOT EXISTS entry_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        hour INTEGER NOT NULL,
        day_of_week INTEGER NOT NULL,
        access_granted INTEGER NOT NULL,
        risk_score REAL DEFAULT 0.0,
        flag TEXT DEFAULT 'NORMAL'
    )''')
    
    # One row per plate — gets updated over time
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_profile (
        plate_id TEXT PRIMARY KEY,
        visit_count INTEGER DEFAULT 0,
        typical_hours TEXT DEFAULT '[]',
        typical_days TEXT DEFAULT '[]',
        avg_entry_hour REAL DEFAULT 0.0,
        first_seen TEXT,
        last_seen TEXT,
        classification TEXT DEFAULT 'UNKNOWN'
    )''')
    
    conn.commit()
    conn.close()
    print("Database initialized!")

# ─────────────────────────────────────────
# 2. LOG AN ENTRY
# ─────────────────────────────────────────
def log_entry(plate_id, access_granted, risk_score, flag):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    
    c.execute('''INSERT INTO entry_log 
        (plate_id, timestamp, hour, day_of_week, access_granted, risk_score, flag)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (plate_id, now.isoformat(), now.hour, 
         now.weekday(), access_granted, risk_score, flag))
    
    conn.commit()
    conn.close()

# ─────────────────────────────────────────
# 3. UPDATE VEHICLE PROFILE
# ─────────────────────────────────────────
def update_profile(plate_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    
    # Get all historical entries for this plate
    c.execute('SELECT hour, day_of_week FROM entry_log WHERE plate_id = ?', 
              (plate_id,))
    history = c.fetchall()
    
    if not history:
        conn.close()
        return
    
    hours = [h[0] for h in history]
    days = [h[1] for h in history]
    visit_count = len(history)
    avg_hour = sum(hours) / len(hours)
    
    # Most common hours and days
    typical_hours = [h for h, count in Counter(hours).most_common(5)]
    typical_days = [d for d, count in Counter(days).most_common(5)]
    
    # Classify the vehicle
    if visit_count >= 10:
        classification = "RESIDENT"
    elif visit_count >= 3:
        classification = "FREQUENT_VISITOR"
    elif visit_count >= 1:
        classification = "VISITOR"
    else:
        classification = "UNKNOWN"
    
    # Upsert profile
    c.execute('''INSERT INTO vehicle_profile 
        (plate_id, visit_count, typical_hours, typical_days, 
         avg_entry_hour, first_seen, last_seen, classification)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(plate_id) DO UPDATE SET
        visit_count = ?,
        typical_hours = ?,
        typical_days = ?,
        avg_entry_hour = ?,
        last_seen = ?,
        classification = ?''',
        (plate_id, visit_count, json.dumps(typical_hours), 
         json.dumps(typical_days), avg_hour,
         now.isoformat(), now.isoformat(), classification,
         visit_count, json.dumps(typical_hours),
         json.dumps(typical_days), avg_hour,
         now.isoformat(), classification))
    
    conn.commit()
    conn.close()

# ─────────────────────────────────────────
# 4. RISK SCORING ENGINE (THE A+ PART)
# ─────────────────────────────────────────
def calculate_risk_score(plate_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()
    
    # Get profile
    c.execute('SELECT * FROM vehicle_profile WHERE plate_id = ?', (plate_id,))
    profile = c.fetchone()
    
    # Brand new plate — high risk
    if not profile:
        conn.close()
        return 0.85, "UNKNOWN_PLATE"
    
    visit_count = profile[1]
    typical_hours = json.loads(profile[2])
    typical_days = json.loads(profile[3])
    avg_hour = profile[4]
    classification = profile[7]
    
    risk = 0.0
    flag = "NORMAL"
    
    # Factor 1: Is this an unusual hour for this plate?
    hour_deviation = abs(current_hour - avg_hour)
    if hour_deviation > 6:
        risk += 0.40
        flag = "UNUSUAL_TIME"
    elif hour_deviation > 3:
        risk += 0.20
    
    # Factor 2: Is this an unusual day?
    if current_day not in typical_days and visit_count > 3:
        risk += 0.25
        flag = "UNUSUAL_DAY"
    
    # Factor 3: Frequency — is this plate appearing too often today?
    c.execute('''SELECT COUNT(*) FROM entry_log 
                 WHERE plate_id = ? 
                 AND timestamp > ?''',
              (plate_id, (now - timedelta(hours=24)).isoformat()))
    today_count = c.fetchone()[0]
    
    if today_count > 5:
        risk += 0.30
        flag = "HIGH_FREQUENCY"
    
    # Factor 4: Resident bonus — known residents get lower base risk
    if classification == "RESIDENT":
        risk = max(0.0, risk - 0.20)
    elif classification == "UNKNOWN":
        risk += 0.15
    
    # Cap risk at 1.0
    risk = min(1.0, risk)
    
    # Determine final flag
    if risk >= 0.7:
        flag = "HIGH_RISK"
    elif risk >= 0.4:
        flag = "MEDIUM_RISK"
    else:
        flag = "LOW_RISK"
    
    conn.close()
    return round(risk, 2), flag

# ─────────────────────────────────────────
# 5. ADAPTIVE GATE DECISION
# ─────────────────────────────────────────
def gate_decision(plate_id, whitelist):
    risk_score, flag = calculate_risk_score(plate_id)
    
    if plate_id in whitelist:
        if risk_score >= 0.7:
            decision = "SLOW_OPEN"  # Known plate but suspicious behaviour
        else:
            decision = "AUTO_OPEN"  # Normal resident
    else:
        if risk_score >= 0.7:
            decision = "DENY"       # Unknown + high risk
        elif risk_score >= 0.4:
            decision = "LOG_ONLY"   # Unknown but medium risk
        else:
            decision = "VISITOR_OPEN"  # Unknown but low risk visitor
    
    # Log this entry
    access_granted = 1 if decision in ["AUTO_OPEN", "SLOW_OPEN", 
                                        "VISITOR_OPEN"] else 0
    log_entry(plate_id, access_granted, risk_score, flag)
    update_profile(plate_id)
    
    return decision, risk_score, flag

# ─────────────────────────────────────────
# 6. TEST THE SYSTEM
# ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    
    whitelist = ["ABC1234", "XYZ5678", "DEF9999"]
    
    print("\n--- Simulating vehicle entries ---\n")
    
    # Simulate ABC1234 entering regularly at normal hours
    test_plates = [
        ("ABC1234", 8),   # Resident, normal morning
        ("ABC1234", 8),
        ("ABC1234", 9),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 8),
        ("ABC1234", 2),   # Resident but at 2am — suspicious!
        ("XYZ5678", 17),  # Another resident, evening
        ("XYZ5678", 18),
        ("XYZ5678", 17),
        ("UNKNOWN99", 3), # Unknown plate at 3am — high risk
    ]
    
    for plate, sim_hour in test_plates:
        decision, risk, flag = gate_decision(plate, whitelist)
        print(f"Plate: {plate:12} | Hour: {sim_hour:02d}:00 | "
              f"Risk: {risk:.2f} | Flag: {flag:15} | Decision: {decision}")
    
    print("\n--- Vehicle Profiles ---\n")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT plate_id, visit_count, avg_entry_hour, classification FROM vehicle_profile')
    for row in c.fetchall():
        print(f"Plate: {row[0]:12} | Visits: {row[1]:3} | "
              f"Avg Hour: {row[2]:5.1f} | Class: {row[3]}")
    conn.close()
