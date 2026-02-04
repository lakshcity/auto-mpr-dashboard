import os
from datetime import date, timedelta
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Resolve DB dynamically (today - 1)
YESTERDAY = date.today() - timedelta(days=1)
# Use the prefix from .env or default to 'my_'
prefix = os.getenv("DB_NAME_PREFIX", "my_")
DB_NAME = f"{prefix}{YESTERDAY.strftime('%Y_%m_%d')}"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432),
    "database": DB_NAME,
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),  # <--- Now hidden!
}

# ... (rest of your query code stays the same) ...

# ============================
# Canonical MPR + Case Query
# NOTE:
# - This query is used AS-IS
# - Aliases are business-friendly by design
# - schema.py MUST match these aliases exactly
# ============================

CASES_QUERY = """
SELECT
    cs.caseid                         AS caseid,
    k.relatedtoid                     AS MPRID,
    cs.reportedon                     AS reportedon,
    lup.name                          AS category,
    cs.currentowner                   AS ownername,
    cex2.cas_ex2_184                  AS COPS_ROPS,
    cex2.cas_ex2_187                  AS sqlrequired,
    cex3.cas_ex6_27                   AS expcloseddate,
    cs.closedate                      AS closeddate,
    k.relatedtoname                   AS MPR_Subject,
    kex1.kit_ex1_20                   AS kit_category,
    kex1.kit_ex1_21                   AS kit_Subcategory,
    kex1.kit_ex1_11                   AS Action,
    kex1.kit_ex1_13                   AS Count,
    kex1.kit_ex1_14                   AS kit_Additionaleffort,
    kex1.kit_ex1_2                    AS kit_Total_Effort,
    cex1.cas_ex1_169                  AS ageing,
    cex2.cas_ex2_197                  AS configurationeffort,
    cex3.cas_ex6_25                   AS testingeffort,
    cex3.cas_ex6_24                   AS totaleffort,
    cex2.cas_ex2_181                  AS L1_Doc_Approval,
    cex2.cas_ex2_182                  AS L2_Doc_Approval,
    cex2.cas_ex2_183                  AS L3_Doc_Approval,
    cex1.cas_ex1_116                  AS channel,
    cex5.cas_ex5_68                   AS casetype,
    cs.details                        AS details,
    cs.subject                        AS subject,
    stc.statuscode                    AS Statuscode
FROM cases cs
INNER JOIN cas_ex1 cex1
    ON cs.caseid = cex1.cas_ex1_id
   AND cs.ownerid = cex1.ownerid
INNER JOIN cas_ex2 cex2
    ON cs.caseid = cex2.cas_ex2_id
   AND cs.ownerid = cex2.ownerid
INNER JOIN cas_ex6 cex3
    ON cs.caseid = cex3.cas_ex6_id
   AND cs.ownerid = cex3.ownerid
INNER JOIN lookupmaster lup
    ON cs.subcategoryid1 = lup.lookupid
   AND cs.ownerid = lup.ownerid
INNER JOIN statuscodemaster stc
    ON cs.statuscodeid = stc.statuscodeid
   AND cs.ownerid = stc.ownerid
INNER JOIN cas_ex5 cex5
    ON cs.caseid = cex5.cas_ex5_id
   AND cs.ownerid = cex5.ownerid
INNER JOIN kit k
    ON k.relatedtoid = cs.caseid
INNER JOIN kit_ex1 kex1
    ON kex1.kit_ex1_id = k.customobjectid
WHERE cs.ownerid = 2
  AND cs.layoutid = 102475;
"""
