import requests

# USGS Earthquake Hazards Program — free, no API key required.
# This feed contains all earthquakes of magnitude 4.5+ in the last 24 hours.
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"


def get_disaster_alerts():
    """
    Fetches recent significant earthquakes (magnitude >= 4.5, last 24h)
    from the USGS Earthquake Hazards Program.

    Returns a list of dicts, each containing:
        type, place, magnitude, time (epoch ms), url, lat, lon

    Returns an empty list if the request fails, so the app can
    continue running even if this feed is temporarily unavailable.
    """
    try:
        response = requests.get(USGS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        alerts = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None, None])

            alerts.append({
                "type": "Earthquake",
                "place": props.get("place", "Unknown location"),
                "magnitude": props.get("mag"),
                "time": props.get("time"),
                "url": props.get("url"),
                "lon": coords[0],
                "lat": coords[1],
            })
        return alerts
    except requests.exceptions.RequestException:
        return []