import requests
from typing import List, Dict


API_URL = (
    "https://myuat.crmnextlab.com/restapibiztech3/"
    "crmWebApi/FetchObject?objectType=case&itemid=0&viewid=0"
)

OUTPUT_FIELDS = [
    "caseid",
    "reportedon",
    "category",
    "statuscode",
    "aging",
    "casetype",
    "cas_ex2_184",
    "cas_ex2_187",
    "cas_ex6_27",
    "closedate",
    "cas_ex1_169",
    "cas_ex2_197",
    "cas_ex6_49",
    "cas_ex6_25",
    "cas_ex6_24",
    "cas_ex2_181",
    "cas_ex2_182",
    "cas_ex2_183",
    "cas_ex1_116",
    "details",
    "subject"
]

def fetch_cases(token: str) -> list[dict]:

    # 🔴 TEMPORARY MOCK BEHAVIOR
    if token == "MOCK_TOKEN_FOR_DEV_ONLY":
        print("⚠️ Using MOCK fetch_cases: returning empty case list")
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "outputFieldList": OUTPUT_FIELDS,
        "objectSearchCondition": [{
            "FieldName": "CaseID",
            "Value": "-1",
            "Operation": "3"
        }],
        "queryOptions": {
            "advanceFilterExpression": "",
            "OrderByFieldName": "",
            "VisibilityOption": "",
            "PageSize": ""
        }
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    return data.get("result", [])

