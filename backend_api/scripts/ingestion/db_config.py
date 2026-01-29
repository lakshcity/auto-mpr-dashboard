# db_config.py
# =========================
# Database Configuration
# =========================

DB_CONFIG = {
    "host": "192.168.0.81",
    "port": 5432,
    "database": "my_2026_01_27",
    "user": "postgres",
    "password": "abc123",
}

# =========================
# Canonical Cases Query
# IMPORTANT:
# - SQL aliases MUST exactly match schema.py -> CSV_COLUMNS
# - Do NOT use business-friendly names here
# =========================

CASES_QUERY = """
SELECT
    distinct cs.caseid                     AS caseid,
    cs.reportedon                 AS reportedon,
    lup.name                      AS category,
    stc.statuscode                AS statuscode,
    cex1.cas_ex1_169              AS aging,
    cex5.cas_ex5_68               AS casetype,

    cex2.cas_ex2_184              AS cas_ex2_184,
    cex2.cas_ex2_187              AS cas_ex2_187,
    cex3.cas_ex6_27               AS cas_ex6_27,
    cs.closedate                  AS closedate,

    cex2.cas_ex2_197              AS cas_ex2_197,
    cex3.cas_ex6_49               AS cas_ex6_49,
    cex3.cas_ex6_25               AS cas_ex6_25,
    cex3.cas_ex6_24               AS cas_ex6_24,

    cex2.cas_ex2_181              AS cas_ex2_181,
    cex2.cas_ex2_182              AS cas_ex2_182,
    cex2.cas_ex2_183              AS cas_ex2_183,

    cex1.cas_ex1_116              AS cas_ex1_116,
    cs.details                    AS details,
    cs.subject                    AS subject

FROM cases cs
LEFT JOIN cas_ex1 cex1 ON cs.caseid = cex1.cas_ex1_id
LEFT JOIN cas_ex2 cex2 ON cs.caseid = cex2.cas_ex2_id
LEFT JOIN cas_ex6 cex3 ON cs.caseid = cex3.cas_ex6_id
LEFT JOIN cas_ex5 cex5 ON cs.caseid = cex5.cas_ex5_id
LEFT JOIN lookupmaster lup ON cs.subcategoryid1 = lup.lookupid
LEFT JOIN statuscodemaster stc ON cs.statuscodeid = stc.statuscodeid

WHERE cs.ownerid = 2 and layoutid = 102475;
"""
