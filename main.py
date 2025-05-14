# FastAPI backend koji prima sliku i šalje inzulin podatke u Nightscout

import base64
import io
import re
import pytz
import json
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
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

# Konfiguracija
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NIGHTSCOUT_TOKEN = os.getenv("NIGHTSCOUT_TOKEN")
NIGHTSCOUT_URL = os.getenv("NIGHTSCOUT_URL")

MODEL_NAME = "meta-llama/llama-3.2-11b-vision-instruct:free"
LOCAL_TZ = pytz.timezone("Europe/Zagreb")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


def image_to_base64(image_file: UploadFile):
    image = Image.open(image_file.file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_data(base64_image):
    prompt = (
        "Na ovoj slici se nalazi unos inzulina."
        " Izvuci dan i mjesec (bez godine), vrijeme i količinu inzulina u jedinicama (U)."
        " Vrati JSON listu u formatu: [ {\"date\":\"MM-DD\", \"time\":\"HH:MM\", \"insulin\": BROJ}, ... ]"
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


def send_to_nightscout(date_str, time_str, insulin_units):
    if re.fullmatch(r"\d{2}-\d{2}", date_str):
        current_year = datetime.now().year
        day, month = date_str.split("-")
        date_str = f"{current_year}-{month.zfill(2)}-{day.zfill(2)}"

    local_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    now_local = datetime.now(LOCAL_TZ)
    local_dt = LOCAL_TZ.localize(local_dt)

    if local_dt > now_local:
        return f"Preskočeno - budući datum: {local_dt}"

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
        return f"{treatment_time}, {insulin_units}U"
    else:
        return f"Greška: {response.status_code} - {response.text}"


@app.post("/upload")
def upload_image(image: UploadFile = File(...)):
    try:
        base64_img = image_to_base64(image)
        result = extract_data(base64_img)

        try:
            data = json.loads(result)
        except:
            return {"error": "GPT nije vratio valjan JSON.", "response": result}

        messages = []
        if isinstance(data, list):
            for entry in data:
                if all(k in entry for k in ["date", "time", "insulin"]):
                    msg = send_to_nightscout(entry["date"], entry["time"], entry["insulin"])
                    messages.append(msg)
                else:
                    messages.append(f"Neispravan unos: {entry}")
        elif isinstance(data, dict):
            msg = send_to_nightscout(data["date"], data["time"], data["insulin"])
            messages.append(msg)

        return {"message": "\n".join(messages)}

    except Exception as e:
        return {"error": str(e)}

