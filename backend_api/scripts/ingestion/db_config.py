# db_config.py

DB_CONFIG = {
    "host": "192.168.0.81",
    "port": 5432,
    "database": "my_2026_01_27",
    "user": "postgres",
    "password": "abc123",
}

# Single source of truth for DB query
CASES_QUERY = """
SELECT
    cs.caseid as caseid,
    cs.reportedon as reportedon,
	lup.name as category,
	cs.currentowner as ownername,
    cex2.cas_ex2_184 as COPS_ROPS,
    cex2.cas_ex2_187 as sqlrequired,
    cas_ex6_27 as expcloseddate ,
    cs.closedate as closeddate,
    cex1.cas_ex1_169 as ageing,
    cex2.cas_ex2_197 as configurationeffort,
    cex3.cas_ex6_25 as testingeffort,
    cex3.cas_ex6_24 as totaleffort,
    cex2.cas_ex2_181 as L1_Doc_Approval,
    cex2.cas_ex2_182 as L2_Doc_Approval,
    cex2.cas_ex2_183 as L3_Doc_Approval,
    cex1.cas_ex1_116 as channel,
	cex5.cas_ex5_68 as casetype,
    cs.details as details,
    cs.subject as subject,
	stc.statuscode as Statuscode
	
FROM cases as cs
INNER JOIN cas_ex1 cex1 on cs.caseid=cex1.cas_ex1_id
Inner join cas_ex2 cex2 on cs.caseid=cex2.cas_ex2_id
Inner join cas_ex6 cex3 on cs.caseid=cex3.cas_ex6_id
INNER JOIN LOOKUPMASTER lup on cs.subcategoryid1=lup.lookupid
INNER JOIN Statuscodemaster stc on cs.statuscodeid=stc.statuscodeid
INNER JOIN cas_ex5 cex5 on cs.caseid=cex5.cas_ex5_id
where cs.ownerid = 2;

"""
