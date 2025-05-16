# FastAPI backend that processes an image and sends insulin data to Nightscout
import base64
import io
import re
import pytz
import json
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NIGHTSCOUT_TOKEN = os.getenv("NIGHTSCOUT_TOKEN")
NIGHTSCOUT_URL = os.getenv("NIGHTSCOUT_URL")

MODEL_NAME = "meta-llama/llama-3.2-11b-vision-instruct"
LOCAL_TZ = pytz.timezone("Europe/Zagreb")

# Initialize OpenAI client for the image analysis model
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

@app.get("/")
def root():
    return {"message": "Image to Nightscout API radi"}

def image_to_base64(image_file: UploadFile):
    # Read and convert the image to base64-encoded JPEG
    image = Image.open(image_file.file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_data(base64_image: str):
    # Prepare prompt for the vision model to extract insulin data
    prompt = (
        "Na ovoj slici se nalazi unos inzulina."
        " Izvuci dan i mjesec (bez godine), vrijeme i količinu inzulina u jedinicama (U)."
        " Vrati isključivo JSON listu u formatu: [ {\"date\":\"MM-DD\", \"time\":\"HH:MM\", \"insulin\": BROJ}, ... ]"
        " Ako nema jasnih podataka, vrati poruku 'Nije pronađeno'."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        temperature=0.2
    )
    return response.choices[0].message.content

def parse_extracted_data(extracted):
    try:
        # Try to parse as JSON string
        data = json.loads(extracted)
        print("✅ Parsirano kao JSON string.")
    except (TypeError, json.JSONDecodeError):
        # If already a list or dict, or not JSON
        data = extracted
        print("ℹ️ Korišteno kao već parsirana struktura ili tekst.")
    return data

def send_to_nightscout(date_str: str, time_str: str, insulin_units):
    # If date is in MM-DD format (no year), prepend current year
    if re.fullmatch(r"\d{2}-\d{2}", date_str):
        current_year = datetime.now().year
        day, month = date_str.split("-")
        date_str = f"{current_year}-{month.zfill(2)}-{day.zfill(2)}"
    # Combine date and time and localize to specified timezone
    local_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    now_local = datetime.now(LOCAL_TZ)
    local_dt = LOCAL_TZ.localize(local_dt)
    if local_dt > now_local:
        return f"Preskočeno - budući datum: {local_dt}"
    # Convert to UTC and format for Nightscout
    utc_dt = local_dt.astimezone(pytz.utc)
    treatment_time = utc_dt.isoformat()
    payload = {
        "enteredBy": "image-web",
        "eventType": "Insulin",
        "insulin": insulin_units,
        "created_at": treatment_time
    }
    response = requests.post(
        f"{NIGHTSCOUT_URL}/api/v1/treatments.json",
        json=payload,
        params={"token": NIGHTSCOUT_TOKEN}
    )
    if response.status_code == 200:
        return f"Uploaded to NS: {treatment_time}, {insulin_units}U"
    else:
        return f"Greška: {response.status_code} - {response.text}"

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """Receive an image, extract insulin data, return the data (no Nightscout upload)."""
    try:
        base64_img = image_to_base64(image)
        result = extract_data(base64_img)
        try:
            data = parse_extracted_data(result)
        except Exception as e:
            return {"error": f"GPT nije vratio valjan JSON. {result}", "response": result}
        # Build preview list of detected entries
        entries_list = []
        preview_list = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and all(k in entry for k in ["date", "time", "insulin"]):
                    entries_list.append(entry)
                    preview_list.append(f"{entry['date']} {entry['time']} – {entry['insulin']}U")
        elif isinstance(data, dict):
            if all(k in data for k in ["date", "time", "insulin"]):
                entries_list.append(data)
                preview_list.append(f"{data['date']} {data['time']} – {data['insulin']}U")
        else:
            # If data is a string or unrecognized format (e.g., "Nije pronađeno")
            return {"message": str(data)}
        # Return detected data without sending to Nightscout
        return {
            "message": "Pronađeni podaci:",
            "data": preview_list,
            "entries": entries_list,
            "note": "Odaberite zapise za slanje i pošaljite ih na /confirm endpoint."
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/confirm")
async def confirm_entries(entries: list = Body(...)):
    """Receive a list of insulin records and send them to Nightscout."""
    messages = []
    for entry in entries:
        if isinstance(entry, dict) and all(k in entry for k in ["date", "time", "insulin"]):
            msg = send_to_nightscout(entry["date"], entry["time"], entry["insulin"])
            messages.append(msg)
    return {
        "message": "✅ Podaci poslani.",
        "log": messages
    }

