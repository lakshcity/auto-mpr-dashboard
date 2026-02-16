import hashlib

USERS = {
    "admin1": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin"
    },
    "consultant1": {
        "password": hashlib.sha256("consult123".encode()).hexdigest(),
        "role": "consultant"
    }
}


def verify_user(username, password):
    if username not in USERS:
        return None

    hashed = hashlib.sha256(password.encode()).hexdigest()

    if USERS[username]["password"] == hashed:
        return USERS[username]["role"]

    return None
