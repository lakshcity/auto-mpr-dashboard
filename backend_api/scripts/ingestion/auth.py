import requests

LOGIN_URL = "https://myuat.crmnextlab.com/restapibiztech3/oauth2/token"


def get_access_token(username: str, password: str) -> str:
    payload = {
        "userName": username,
        "password": password
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        LOGIN_URL,
        json=payload,
        headers=headers
    )

    print("Auth status:", response.status_code)
    print("Auth response:", response.text)

    # 🔴 TEMPORARY MOCK (until real creds are fixed)
    if response.status_code == 403:
        print("⚠️ Using MOCK TOKEN due to invalid credentials")
        return "MOCK_TOKEN_FOR_DEV_ONLY"

    response.raise_for_status()

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Access token not found in response")

    return token

