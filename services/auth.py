import hashlib

USERS = {
    "akarsh": {"password": "akarsh123", "role": "consultant"},
    "anubhav": {"password": "anubhav123", "role": "consultant"},
    "deepali": {"password": "deepali123", "role": "consultant"},
    "vivek": {"password": "vivek123", "role": "consultant"},
    "sagar": {"password": "sagar123", "role": "consultant"},
    "veeresh": {"password": "veeresh123", "role": "consultant"},
    "vishal": {"password": "vishal123", "role": "consultant"},
    "consult1": {"password": "consult123", "role": "consultant"},
    "dheeraj": {"password": "dheeraj123", "role": "admin"},
    "aditya": {"password": "aditya123", "role": "admin"},
    "admin1": {"password": "admin123", "role": "admin"},
    "laksh": {"password": "laksh123", "role": "admin_analyst"},
    "himanshu": {"password": "himanshu123", "role": "admin_analyst"},
    "amit": {"password": "amit123", "role": "admin_analyst"},
    "analyst1": {"password": "analyst123", "role": "admin_analyst"},
}


def verify_user(username, password):
    user = USERS.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None
