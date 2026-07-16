"""Hard-coded weather data and the tool functions exposed to the LLM."""

from __future__ import annotations

from overmind import tool

# Fake weather "database" - deliberately hard-coded for this demo agent.
WEATHER_DATA: dict[str, dict[str, float | str]] = {
    "london": {"description": "Overcast with light drizzle", "temperature_c": 14.0},
    "new york": {"description": "Sunny with a light breeze", "temperature_c": 22.0},
    "tokyo": {"description": "Humid and partly cloudy", "temperature_c": 27.0},
    "sydney": {"description": "Clear skies", "temperature_c": 19.0},
    "cairo": {"description": "Hot and dry", "temperature_c": 36.0},
    "moscow": {"description": "Snowing, bitterly cold", "temperature_c": -8.0},
    "nairobi": {"description": "Mild and sunny", "temperature_c": 21.0},
    "reykjavik": {"description": "Windy with sleet", "temperature_c": 3.0},
}


@tool()
def get_weather(location: str) -> dict[str, str | float]:
    """Look up the current weather for a hard-coded set of locations."""
    key = location.strip().lower()
    if key not in WEATHER_DATA:
        known = ", ".join(sorted(name.title() for name in WEATHER_DATA))
        return {
            "error": f"No weather data for '{location}'. Known locations: {known}"
        }

    data = WEATHER_DATA[key]
    return {
        "location": location.title(),
        "description": data["description"],
        "temperature_c": data["temperature_c"],
    }


@tool()
def calculate_temperature_difference(temperature_a: float, temperature_b: float) -> dict[str, float | str]:
    """Calculate the absolute difference between two temperatures in Celsius."""
    difference = round(abs(temperature_a - temperature_b), 2)
    warmer = "a" if temperature_a > temperature_b else ("b" if temperature_b > temperature_a else "equal")
    return {
        "difference_c": difference,
        "warmer": warmer,
    }


# Tool schemas in the OpenAI function-calling format.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather description and temperature (Celsius) for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city, e.g. 'London' or 'Tokyo'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_temperature_difference",
            "description": "Calculate the absolute difference in Celsius between two temperatures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature_a": {
                        "type": "number",
                        "description": "The first temperature in Celsius.",
                    },
                    "temperature_b": {
                        "type": "number",
                        "description": "The second temperature in Celsius.",
                    },
                },
                "required": ["temperature_a", "temperature_b"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "get_weather": get_weather,
    "calculate_temperature_difference": calculate_temperature_difference,
}
