# schema.py
# =====================================
# Canonical CSV Schema for Auto-MPR
# =====================================
# IMPORTANT:
# - Column names MUST exactly match SQL aliases
# - Order here defines CSV column order
# - Business logic happens elsewhere
# =====================================

CSV_COLUMNS = [
    "caseid",
    "MPRID",
    "reportedon",
    "category",
    "ownername",
    "COPS_ROPS",
    "sqlrequired",
    "expcloseddate",
    "closeddate",
    "MPR_Subject",
    "kit_category",
    "kit_Subcategory",
    "Action",
    "Count",
    "kit_Additionaleffort",
    "kit_Total_Effort",
    "ageing",
    "configurationeffort",
    "testingeffort",
    "totaleffort",
    "L1_Doc_Approval",
    "L2_Doc_Approval",
    "L3_Doc_Approval",
    "channel",
    "casetype",
    "details",
    "subject",
    "Statuscode"
]


# Primary key used across snapshot + master upserts
PRIMARY_KEY = "caseid"


# Optional: grouping hints for downstream logic / BI
NUMERIC_FIELDS = [
    "Count",
    "kit_Additionaleffort",
    "kit_Total_Effort",
    "configurationeffort",
    "testingeffort",
    "totaleffort"
]

DATE_FIELDS = [
    "reportedon",
    "expcloseddate",
    "closeddate"
]
