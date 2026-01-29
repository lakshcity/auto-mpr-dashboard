"""
Canonical schema definition for Auto MPR cases.

This file is the single source of truth for:
- CSV column order
- API field → CSV column mapping
"""

# 🔹 Final CSV column order (DO NOT change casually)
CSV_COLUMNS = [
    "caseid",          # Primary key (Case ID)
    "reportedon",      # Case creation timestamp
    "category",        # Case category (lookupmaster.name)
    "statuscode",      # Case status (statuscodemaster.statuscode)
    "aging",           # Aging in days (business aging metric)
    "casetype",        # Case type (Requirement / Issue / Change)

    "cas_ex2_184",     # COPS / ROPS
    "cas_ex2_187",     # SQL Required (Yes/No)
    "cas_ex6_27",      # Expected Close Date
    "closedate",       # Actual Close Date

    "cas_ex2_197",     # Configuration Effort (Hours)
    "cas_ex6_49",      # PM Effort (Hours)
    "cas_ex6_25",      # Testing Effort (Hours)
    "cas_ex6_24",      # Total Effort (Hours)

    "cas_ex2_181",     # L1 Document Approval
    "cas_ex2_182",     # L2 Document Approval
    "cas_ex2_183",     # L3 Document Approval

    "cas_ex1_116",     # Channel (Email / Portal / Phone)
    "details",         # Case description
    "subject",         # Case subject/title
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
# =========================
# Business Field Mapping
# =========================
# Use this for analytics, summaries, UI, and BI logic
# Do NOT use for ingestion or upsert

BUSINESS_FIELD_MAP = {
    "cas_ex2_184": "cops_rops",
    "cas_ex2_187": "sql_required",
    "cas_ex6_27": "expected_close_date",
    "cas_ex2_197": "configuration_effort_hours",
    "cas_ex6_49": "pm_effort_hours",
    "cas_ex6_25": "testing_effort_hours",
    "cas_ex6_24": "total_effort_hours",
    "cas_ex2_181": "l1_doc_approval",
    "cas_ex2_182": "l2_doc_approval",
    "cas_ex2_183": "l3_doc_approval",
    "cas_ex1_116": "channel",
}

