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
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError("Access token not found in response")

    return token
