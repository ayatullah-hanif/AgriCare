# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io, os, hashlib, datetime, csv, json, time
from tensorflow.keras.models import load_model
from gtts import gTTS
from deep_translator import GoogleTranslator
import pandas as pd

# -------- CONFIG --------
MODEL_PATH = "cassava_model.h5"     # path to your trained model
AUDIO_CACHE_DIR = "audio_cache"
QUEUE_DIR = "queued_images"
QUEUE_INDEX = os.path.join(QUEUE_DIR, "queue_metadata.csv")
TRANSLATIONS_CACHE = "translations_cache.json"
CONFIDENCE_THRESHOLD = 0.60

# Ensure folders exist
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)
if not os.path.exists(QUEUE_INDEX):
    with open(QUEUE_INDEX, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","filename","predicted_class","confidence","notes"])

if not os.path.exists(TRANSLATIONS_CACHE):
    with open(TRANSLATIONS_CACHE, "w", encoding="utf-8") as jf:
        json.dump({}, jf)

# -------- CLASS NAMES --------
CLASS_NAMES = [
    "Cassava___bacterial_blight",
    "Cassava___brown_streak_disease",
    "Cassava___green_mottle",
    "Cassava___mosaic_disease",
    "Cassava___healthy"
]

# -------- ENGLISH ADVICE + PESTICIDE RECOMMENDATIONS --------
ADVICE = {
    "Cassava___bacterial_blight": {
        "Small Farmland": (
            "Short-term (smallholder): Remove and destroy infected plants immediately. "
            "Burn or deeply bury infected debris and avoid replanting from symptomatic material. "
            "Use only clean cuttings, disinfect tools after use, and avoid overhead watering. "
            "Recommended Pesticide: Copper-based fungicides (e.g., Copper Oxychloride 50 WP) and resistant varieties like TMS 30572."
        ),
        "Big Farmland": (
            "Long-term (large farm): Adopt resistant varieties where available, implement certified clean seed systems, "
            "practice crop rotation, and institutionalize tool and equipment disinfection protocols. "
            "Train staff on early detection and coordinate with extension services for area-wide management. "
            "Recommended Pesticide: Copper-based fungicides or biological bactericides approved for cassava."
        )
    },
    "Cassava___brown_streak_disease": {
        "Small Farmland": (
            "Short-term (smallholder): Uproot and destroy plants showing brown streak symptoms. "
            "Do not reuse infected cuttings. Remove and burn infected stems after harvest. "
            "Recommended Action: Apply Imidacloprid or Dimethoate to control whiteflies."
        ),
        "Big Farmland": (
            "Long-term (large farm): Source and plant certified virus-free or tolerant varieties. "
            "Invest in clean-seed propagation (tissue culture/greenhouse) and community phytosanitation campaigns. "
            "Recommended Action: Use resistant cultivars and enforce whitefly vector control using systemic insecticides."
        )
    },
    "Cassava___green_mottle": {
        "Small Farmland": (
            "Short-term (smallholder): Isolate and destroy symptomatic plants, avoid sharing unclean tools, "
            "and replant only with verified healthy cuttings. "
            "Recommended Treatment: Use Neem-based insecticides or Lambda-cyhalothrin against aphids."
        ),
        "Big Farmland": (
            "Long-term (large farm): Implement systematic surveillance, strict tool disinfection protocols, "
            "and integrate pest management (monitoring traps, biological controls). "
            "Recommended Treatment: Biological control and IPM approach targeting aphid vectors."
        )
    },
    "Cassava___mosaic_disease": {
        "Small Farmland": (
            "Short-term (smallholder): Use virus-free cuttings, remove mosaic-infected plants promptly, "
            "and reduce whitefly populations by intercropping or cultural controls. "
            "Recommended Treatment: Cypermethrin or Acetamiprid to manage whiteflies."
        ),
        "Big Farmland": (
            "Long-term (large farm): Deploy CMD-resistant cultivars, establish a clean seed program, "
            "integrate vector control strategies, and collaborate with research/extension services. "
            "Recommended Treatment: Regular field sanitation, resistant cultivars, and selective vector control."
        )
    },
    "Cassava___healthy": {
        "Small Farmland": (
            "Short-term (smallholder): Maintain good practices — use clean cuttings, weed regularly, "
            "monitor crops weekly, and maintain soil fertility. "
            "Recommended Action: Continue current management; no pesticide required."
        ),
        "Big Farmland": (
            "Long-term (large farm): Implement integrated pest management, maintain strict sanitation, "
            "train workers on early detection, and use certified planting materials. "
            "Recommended Action: Maintain surveillance and record-keeping for early detection."
        )
    }
}

LANG_CODE = {"English": "en", "Hausa": "ha", "Yoruba": "yo"}

# -------- UTILITIES --------
@st.cache_resource(show_spinner=False)
def load_prediction_model(path):
    try:
        m = load_model(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_pil_image(pil_img, target_size=(224,224)):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_from_pil(pil_img, model):
    x = preprocess_pil_image(pil_img)
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    predicted = CLASS_NAMES[class_idx]
    return predicted, confidence, preds[0]

def translate_text_cached(english_text, target_code, cache_file=TRANSLATIONS_CACHE):
    if target_code == "en":  
        return english_text

    try:
        with open(cache_file, "r", encoding="utf-8") as jf:
            cache = json.load(jf)
    except Exception:
        cache = {}

    key = hashlib.md5((english_text + target_code).encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    try:
        translated = GoogleTranslator(source="en", target=target_code).translate(english_text)
        cache[key] = translated
        with open(cache_file, "w", encoding="utf-8") as jf:
            json.dump(cache, jf, ensure_ascii=False, indent=2)
        return translated
    except Exception as e:
        st.warning(f"Translation failed ({target_code}): using English fallback. ({e})")
        return english_text

def text_to_speech_cached(text, lang_code, cache_dir=AUDIO_CACHE_DIR):
    key = hashlib.md5((text + lang_code).encode("utf-8")).hexdigest()
    fname = f"{key}_{lang_code}.mp3"
    fpath = os.path.join(cache_dir, fname)
    if os.path.exists(fpath):
        return fpath
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(fpath)
        return fpath
    except Exception as e:
        st.warning(f"TTS generation failed: {e}")
        return None

def queue_for_review(pil_img, predicted, confidence, notes=""):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = f"{predicted}_{int(confidence*100)}_{ts}.jpg"
    path = os.path.join(QUEUE_DIR, fname)
    pil_img.save(path, format="JPEG")
    with open(QUEUE_INDEX, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, fname, predicted, f"{confidence:.4f}", notes])
    return path

# -------- STREAMLIT UI --------
st.set_page_config(page_title="AgriCare - Crop Diagnostic (Demo)", layout="centered")
st.title("AgriCare — Crop Disease Detection & Advice")

farmland_size = st.selectbox("Select Farmland Size:", ["Small Farmland", "Big Farmland"])
language = st.selectbox("Select Language:", ["English", "Hausa", "Yoruba"])
uploaded_file = st.file_uploader("Upload Cassava Leaf Image", type=["jpg","jpeg","png"])

model = load_prediction_model(MODEL_PATH)

if uploaded_file and model:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="Uploaded Leaf", use_column_width=True)

    predicted, confidence, probs = predict_from_pil(pil_img, model)

    st.write(f"**Prediction:** {predicted}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if confidence >= CONFIDENCE_THRESHOLD:
        english_advice = ADVICE.get(predicted, {}).get(farmland_size, "No advice available.")
        target_code = LANG_CODE[language]
        advice_text = translate_text_cached(english_advice, target_code)

        st.subheader("Recommended Actions")
        st.info(advice_text)

        audio_path = text_to_speech_cached(advice_text, target_code)
        if audio_path:
            st.audio(audio_path)
    else:
        st.warning("Low confidence prediction. Image queued for human review.")
        note = st.text_area("Optional note for review")
        if st.button("Queue for review"):
            saved_path = queue_for_review(pil_img, predicted, confidence, notes=note)
            st.success(f"Saved for review: {saved_path}")

# -------- DEMO MODE: ROTATE THROUGH DISEASES FOR SHOWCASE --------
st.markdown("---")
st.subheader("🎬 Demo Simulation Mode")

demo_mode = st.checkbox("Activate Demo Mode (Show All Diseases Sequentially)", value=False)

if demo_mode:
    demo_images = [
        "Cassava___bacterial_blight",
        "Cassava___brown_streak_disease",
        "Cassava___green_mottle",
        "Cassava___mosaic_disease",
        "Cassava___healthy"
    ]
    st.write("Simulating detection across all disease classes...")
    for disease in demo_images:
        st.markdown(f"### 🧠 Predicted: {disease}")
        english_advice = ADVICE[disease][farmland_size]
        target_code = LANG_CODE[language]
        advice_text = translate_text_cached(english_advice, target_code)
        st.info(advice_text)
        time.sleep(3)
    st.success("Simulation Complete ✅")
