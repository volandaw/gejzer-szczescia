import streamlit as st
import joblib
import numpy as np
import boto3
import os
import io
from dotenv import load_dotenv
from openai import OpenAI
import json
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Wczytaj klucze
load_dotenv()
# Inicjalizacja Langfuse
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# Połącz z DO Spaces i pobierz model
@st.cache_resource
def load_model_from_spaces():
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    bucket = os.getenv('BUCKET_NAME')
    obj = s3.get_object(Bucket=bucket, Key='Polmaraton/model_polmaraton.pkl')
    model = joblib.load(io.BytesIO(obj['Body'].read()))
    return model

# Klient OpenAI
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Wyłuskaj dane z tekstu przez LLM
@observe()
def extract_data_with_llm(text):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Wyłuskaj dane biegacza z tekstu i zwróć JSON:
                {
                    "plec": "M" lub "K" lub null,
                    "wiek": liczba lub null,
                    "czas_5km_minuty": liczba minut lub null
                }
                Zwróć TYLKO JSON, nic więcej."""
            },
            {"role": "user", "content": text}
        ]
    )
    result = json.loads(response.choices[0].message.content)
    langfuse_context.update_current_observation(
        input=text,
        output=result
    )
    return result

# Interfejs aplikacji
st.title("🏃 Kalkulator Półmaratonu")
st.write("Przedstaw się i powiedz o swojej płci, wieku i czasie na 5km!")

tekst = st.text_area("Twój opis:", placeholder="Np. Cześć, jestem Andrzej, mężczyzna, mam 65 lat i 5km biegnę w 28 minut")

if st.button("Oblicz czas!"):
    if tekst:
        with st.spinner("Analizuję..."):
            # Wyłuskaj dane
            dane = extract_data_with_llm(tekst)
            
            # Sprawdź czy wszystkie dane są dostępne
            brakujace = []
            if dane.get('plec') is None:
                brakujace.append("płeć")
            if dane.get('wiek') is None:
                brakujace.append("wiek")
            if dane.get('czas_5km_minuty') is None:
                brakujace.append("czas na 5km")
            
            if brakujace:
                st.warning(f"Brakuje danych: {', '.join(brakujace)}")
            else:
                # Przewiduj czas
                model = load_model_from_spaces()
                plec = 0 if dane['plec'] == 'M' else 1
                wiek_kat = (dane['wiek'] // 10) * 10
                czas_5km_sek = dane['czas_5km_minuty'] * 60
                
                X = np.array([[plec, wiek_kat, czas_5km_sek]])
                czas_sek = model.predict(X)[0]
                
                godziny = int(czas_sek // 3600)
                minuty = int((czas_sek % 3600) // 60)
                sekundy = int(czas_sek % 60)
                
                st.success(f"🏅 Przewidywany czas półmaratonu: {godziny}h {minuty}min {sekundy}s")
                st.json(dane)
                