# !pip install -U "langchain>=0.3.12" \
#                "langchain-core>=0.3.30" \
#                "langchain-google-genai>=2.0.0"


import requests
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
# Make sure your .env file contains:  OPENAI_API_KEY="your_api_key"
load_dotenv()

# -----------------------------
# 1. Weather API function
# -----------------------------
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Define a Pydantic schema for tool inputs
# This adds validation, type hints, and better tool calling behavior.
class WeatherInput(BaseModel):
    city: str = Field(..., description="Name of the city to get weather for")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    # Geocode the city
    r = requests.get(GEOCODING_API_URL, params={"name": city, "count": 1}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        return f"Error: could not find location for {city}"
    lat = data["results"][0]["latitude"]
    lon = data["results"][0]["longitude"]

    # Fetch weather
    w = requests.get(
        FORECAST_URL,
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=15,
    )
    w.raise_for_status()
    weather = w.json().get("current_weather")
    return f"City: {city}, Lat: {lat}, Lon: {lon}, Current Weather: {weather}"

# -----------------------------
# 3. Define the Chat Model
# -----------------------------
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)

# -----------------------------
# 4. Create Agent
# -----------------------------
agent = create_agent(
    model=model,
    tools=[get_weather],  # pass the decorated tool directly

)

user_input = "What is the weather in Tokyo?"
response = agent.invoke( {"messages": [{"role": "user", "content":user_input}]}
)
print(response)
