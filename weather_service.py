import os
import requests

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city):

    if not city or not city.strip():
        return None

    if not OPENWEATHER_API_KEY:
        print("OPENWEATHER_API_KEY not found")
        return None

    try:
        params = {
            "q": city.strip(),
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None