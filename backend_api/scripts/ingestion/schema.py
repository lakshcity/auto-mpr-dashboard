"""
Canonical schema definition for Auto MPR cases.

This file is the single source of truth for:
- CSV column order
- API field → CSV column mapping
"""

# 🔹 Final CSV column order (DO NOT change casually)
CSV_COLUMNS = [
    "caseid",
    "reportedon",
    "category",
    "statuscode",
    "casetype",
    "Cops-Rops",
    "sqlrequired",
    "expclosedate",
    "closedate",
    "aging",
    "currentowner",
    "ConfigurationEffort[Hr]",
    "PM Effort [Hr]",
    "TestingEffort [Hr]",
    "TotalEffort [Hr]",
    "L1 Doc Approval",
    "L2 Doc Approval",
    "L3 Doc Approval",
    "channel",
    "details",
    "subject"
]

# 🔹 Mapping: API field → CSV column
API_TO_CSV_FIELD_MAP = {
    "caseid": "caseid",
    "reportedon": "reportedon",
    "category": "category",
    "statuscode": "statuscode",
    "cas_ex5_68": "casetype",
    "cas_ex2_184": "Cops-Rops",
    "cas_ex2_187": "sqlrequired",
    "cas_ex6_27": "expclosedate",
    "closedate": "closedate",
    "aging": "aging",
    "cas_ex1_169": "currentowner",
    "cas_ex2_197": "ConfigurationEffort[Hr]",
    "cas_ex6_49": "PM Effort [Hr]",
    "cas_ex6_25": "TestingEffort [Hr]",
    "cas_ex6_24": "TotalEffort [Hr]",
    "cas_ex2_181": "L1 Doc Approval",
    "cas_ex2_182": "L2 Doc Approval",
    "cas_ex2_183": "L3 Doc Approval",
    "cas_ex1_116": "channel",
    "details": "details",
    "subject": "subject"
}
