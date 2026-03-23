import sqlite3
import time
from behaviour import gate_decision

WHITELIST = ['ABC123', 'ML773', 'TX8971', 'YZ3527']

scenarios = [
    # Whitelist residents — SLOW_OPEN only at risk>=0.7, otherwise AUTO_OPEN even with UNUSUAL_TIME
    ('Resident unusual hour, whitelisted',   'ML773',      'UNUSUAL_TIME', 'AUTO_OPEN'),
    ('Resident normal hour, whitelisted',    'ABC123',     'LOW_RISK',     'AUTO_OPEN'),
    # Non-whitelist residents — go through visitor logic
    ('Resident unusual hour, not whitelisted','KL5678',    'UNUSUAL_TIME', 'VISITOR_OPEN'),
    ('Resident unusual hour, not whitelisted','TD001',     'UNUSUAL_TIME', 'VISITOR_OPEN'),
    ('Resident unusual hour, not whitelisted','TD1001',    'UNUSUAL_TIME', 'VISITOR_OPEN'),
    # Frequent visitor unusual hour — risk>=0.4 → LOG_ONLY
    ('Freq visitor unusual hour',            'NXY590',    'UNUSUAL_TIME', 'LOG_ONLY'),
    # Blacklisted — always DENY
    ('Blacklisted plate BAD001',             'BAD001',    'BLACKLISTED',  'DENY'),
    ('Blacklisted plate ZZZ999',             'ZZZ999',    'BLACKLISTED',  'DENY'),
    # Unknown new plate — risk 0.85 → DENY for non-whitelist
    ('Brand new unknown plate',              'NEWPLATE999','UNKNOWN_PLATE','DENY'),
    # Whitelisted resident normal
    ('Whitelisted resident low risk',        'YZ3527',    'LOW_RISK',     'AUTO_OPEN'),
]

# Set avg_entry_hour to 10.0 so deviation from current hour (~3am) = 7hrs → UNUSUAL_TIME
import sqlite3
c = sqlite3.connect('behaviour.db')
for plate in ['ML773','ABC123','KL5678','TD001','TD1001','NXY590','YZ3527']:
    c.execute('UPDATE vehicle_profile SET avg_entry_hour=10.0 WHERE plate_id=?', (plate,))
c.execute("UPDATE vehicle_profile SET classification='FREQUENT_VISITOR' WHERE plate_id='NXY590'")
# ABC123 and YZ3527 — set to current hour so no deviation → LOW_RISK
import time
current = time.localtime().tm_hour
c.execute('UPDATE vehicle_profile SET avg_entry_hour=? WHERE plate_id=?', (current, 'ABC123'))
c.execute('UPDATE vehicle_profile SET avg_entry_hour=? WHERE plate_id=?', (current, 'YZ3527'))
c.commit()
c.close()

print(f"{'#':<3} {'Scenario':<42} {'Exp Flag':<18} {'Act Flag':<18} {'Exp Dec':<15} {'Act Dec':<15} PASS?")
print('-' * 120)

passed = 0
for i, (desc, plate, exp_flag, exp_decision) in enumerate(scenarios, 1):
    decision, risk, flag = gate_decision(plate, WHITELIST, dry_run=True)
    flag_pass = exp_flag in flag
    dec_pass = decision == exp_decision
    ok = flag_pass and dec_pass
    if ok:
        passed += 1
    print(f"{i:<3} {desc:<42} {exp_flag:<18} {flag:<18} {exp_decision:<15} {decision:<15} {'PASS' if ok else 'FAIL'}")

print(f"\nResults: {passed}/10 passed — Precision: {passed/10*100:.0f}%")

# Restore original avg_entry_hours
c = sqlite3.connect('behaviour.db')
for plate, avg in [('ML773',7.59),('KL5678',6.33),('TD001',8.28),
                   ('TD1001',7.93),('NXY590',6.21)]:
    c.execute('UPDATE vehicle_profile SET avg_entry_hour=? WHERE plate_id=?', (avg, plate))
c.execute("UPDATE vehicle_profile SET classification='RESIDENT' WHERE plate_id='NXY590'")
c.commit()
c.close()
print('(avg_entry_hours restored)')
