from math import radians, sin, cos, sqrt, atan2


def _distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points (Haversine formula), in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def filter_nearby_alerts(alerts, city_lat, city_lon, radius_km=500):
    """
    Filters a list of alerts (from disaster_feed.get_disaster_alerts) to only
    those within `radius_km` of the given coordinates. Adds a 'distance_km'
    field and sorts by proximity (closest first).
    """
    nearby = []
    for a in alerts:
        if a.get("lat") is None or a.get("lon") is None:
            continue
        dist = _distance_km(city_lat, city_lon, a["lat"], a["lon"])
        if dist <= radius_km:
            a = dict(a)  # avoid mutating the original dict
            a["distance_km"] = round(dist, 1)
            nearby.append(a)
    return sorted(nearby, key=lambda x: x["distance_km"])


def calculate_risk(predicted_class, confidence, weather, alerts):
    """
    Combines three signals into a single 0-100 risk score:

    1. AI image prediction (60% weight)
       - If the model detected a disaster (not "normal"), the score scales
         directly with its confidence.
       - If the model is confident it's "normal", this contributes very little.

    2. Current weather conditions (up to 25 points)
       - Conditions that make the *detected* disaster type worse
         (e.g. low humidity + high wind for fire, active rain for flood/landslide)
         add to the score.

    3. Nearby live disaster alerts, e.g. recent earthquakes (up to 15 points)
       - A strong nearby event (M5.0+) suggests the area is already under stress.

    Returns a float between 0 and 100.
    """
    score = 0.0

    # ---- 1. AI prediction ----
    if predicted_class == "normal":
        score += (100 - confidence) * 0.3
    else:
        score += confidence * 0.6

    # ---- 2. Weather conditions ----
    if weather:
        try:
            temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]
            condition = weather["weather"][0]["main"].lower()
        except (KeyError, IndexError):
            temp = humidity = wind_speed = None
            condition = ""

        weather_score = 0
        if predicted_class in ("fire", "smoke"):
            if temp is not None and temp > 35:
                weather_score += 10
            if humidity is not None and humidity < 30:
                weather_score += 10
            if wind_speed is not None and wind_speed > 8:
                weather_score += 5
        elif predicted_class == "flood":
            if "rain" in condition or "storm" in condition or "drizzle" in condition:
                weather_score += 15
            if humidity is not None and humidity > 80:
                weather_score += 10
        elif predicted_class == "landslide":
            if "rain" in condition or "storm" in condition:
                weather_score += 20

        score += min(weather_score, 25)

    # ---- 3. Nearby disaster alerts ----
    if alerts:
        significant = [a for a in alerts if (a.get("magnitude") or 0) >= 5.0]
        if significant:
            score += 15
        else:
            score += 7

    return round(min(score, 100), 1)