
from dotenv import load_dotenv
import os
import httpx  


load_dotenv()

API_KEY = os.getenv("API_KEY")



def check_env():
    print("=== Checking Environment ===")

    if API_KEY:
        print(f" API_KEY loaded → starts with: {API_KEY[:8]}...")
    else:
        print(" API_KEY not found — check your .env file")



def test_http_request():
    print("\n=== Testing HTTP Request ===")

    url = "https://official-joke-api.appspot.com/random_joke"

    response = httpx.get(url)

    print(f"Status Code: {response.status_code}")

    data = response.json()

    print(f"Setup: {data['setup']}")
    print(f"Punchline: {data['punchline']}")

def check_python_version():
    import sys
    print(f"\n=== Python Version ===")
    print(f"Python {sys.version}")
    print(" Good to go!" if sys.version_info >= (3, 11) else " Please upgrade to Python 3.11+")



if __name__ == "__main__":
    check_python_version()
    check_env()
    test_http_request()
    print("\n Day 1 Complete ")