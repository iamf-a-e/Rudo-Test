import google.generativeai as genai
from flask import Flask, request, jsonify, render_template 
import requests    
import os  
import fitz 
import sched   
import time     
import logging    
from mimetypes import guess_type 
from datetime import datetime, timedelta 
from urlextract import URLExtract
from training import instructions, product_images
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
from training import products, instructions, pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba, pregnancy_data_lozi, cervical_cancer_data, cervical_cancer_data_chinyanja, cervical_cancer_data_lozi

from products_data import products_by_category
from upstash_redis import Redis
import json
import re
import random
import string

logging.basicConfig(level=logging.INFO)

# Initialize Upstash Redis connection
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN not set, Redis functionality disabled")

# Global in-memory cache (per worker)
user_states = {}

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.5-flash"
genai.configure(api_key=gen_api)
name = "Fae"
bot_name = "Rudo"
AGENT = "+263719835124"

app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},   
]

# ─────────────────────────────────────────────
#  REMOVED: global convo object
#  All Gemini calls now use stateless per-call
#  generation to prevent cross-user language
#  contamination from shared chat history.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  PER-USER REDIS STATE  (fixes multi-user bug)
# ─────────────────────────────────────────────

def save_single_user_state(sender):
    """Save only one user's state to Redis under their own key."""
    if redis_client and sender in user_states:
        try:
            redis_client.set(f"user_state:{sender}", json.dumps(user_states[sender]))
            logging.debug(f"Saved state for {sender}")
        except Exception as e:
            logging.error(f"Error saving state for {sender}: {e}")

def load_user_state(sender):
    """Load a single user's state from Redis. Returns dict or None."""
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            if state_data:
                return json.loads(state_data)
        except Exception as e:
            logging.error(f"Error loading user state for {sender}: {e}")
    return None

def save_user_states():
    """Save all in-memory user states to Redis (one key per user)."""
    for sender in list(user_states.keys()):
        save_single_user_state(sender)

def load_user_states():
    """No-op at startup – states are loaded on demand per user."""
    global user_states
    user_states = {}
    logging.info("User states initialised (lazy per-user loading enabled)")


# ─────────────────────────────────────────────
#  HELPER: ensure a sender is in user_states
# ─────────────────────────────────────────────

def ensure_user_state(sender):
    """
    Make sure user_states[sender] exists.
    Tries Redis first; if not found creates a fresh state.
    Returns True if this is a brand-new user, False otherwise.
    """
    if sender in user_states:
        return False  # already in memory

    saved = load_user_state(sender)
    if saved:
        user_states[sender] = saved
        return False  # returning user

    # Brand-new user
    user_states[sender] = {
        "step": "language_detection",
        "language": "english",
        "registered": False,
        "phone_digits": None,
        "user_id": None,
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": True
    }
    return True


def reset_conversation(sender):
    user_states[sender] = {
        "step": "main_menu",
        "language": user_states[sender].get("language", "english"),
        "registered": True,
        "phone_digits": user_states[sender].get("phone_digits"),
        "user_id": user_states[sender].get("user_id"),
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": False
    }
    save_single_user_state(sender)


def get_user_conversation(sender):
    """Get user conversation history from Upstash Redis"""
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            if not history:
                return []
            if isinstance(history, list):
                return history
            if isinstance(history, str):
                parsed = json.loads(history)
                if isinstance(parsed, list):
                    return parsed
            return []
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return []
    return []

def save_user_conversation(sender, role, message):
    """Save user conversation to Upstash Redis"""
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            if not isinstance(conversation, list):
                conversation = []
            conversation.append({
                "role": role,
                "message": str(message),
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.set(f"conversation:{sender}", json.dumps(conversation), ex=60*60*24*30)
            logging.debug(f"Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")

def detect_language(message, sender=None):
    message_lower = message.lower().strip()

    if message_lower.isdigit():
        if sender and sender in user_states:
            return user_states[sender].get("language", "english")
        return "english"

    # ── English override ──────────────────────────────────────────────────────
    common_english_words = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could","should",
        "may","might","shall","can","need","must","ought",
        "i","you","he","she","it","we","they","me","him","her","us","them",
        "my","your","his","its","our","their","this","that","these","those",
        "what","which","who","whom","whose","where","when","why","how",
        "and","or","but","if","then","so","because","although","while",
        "not","no","yes","please","thank","thanks","sorry","okay","ok",
        "to","of","in","on","at","for","from","with","about","during",
        "signs","watch","out","risky","pregnancy","symptoms","tell","give",
        "show","help","know","want","need","get","go","come","see","look",
        "take","make","say","ask","work","feel","think","try","use","find",
        "early","late","common","normal","severe","pain","blood","baby",
        "mother","health","doctor","hospital","clinic","test","week","month",
        "information","more","other","any","all","some","much","many","very",
        "also","just","only","still","even","back","too","well","good","bad",
        "new","old","long","little","right","big","high","low","next","last",
        "between","after","before","during","since","until","without","within",
        "up","down","over","under","again","further","once","same","own","both",
        "each","few","most","such","than","as","by","into","through","against",
        "along","following","across","behind","beyond","plus","except",
        "including","throughout","towards","upon","concerning",
        "cancer","cervical","maternal","infection","treatment","care",
        "prevent","prevention","risk","sign","symptom","cause","effect","stage",
        "screening","vaccine","hpv","period","bleeding","discharge","smell",
        "trimester","birth","delivery","labor","feed","breastfeed","iron",
        "vitamin","supplement","scan","ultrasound","check","visit","appointment",
    }
    words_in_msg = set(re.findall(r"[a-z]+", message_lower))
    if words_in_msg:
        en_count = sum(1 for w in words_in_msg if w in common_english_words)
        ratio = en_count / len(words_in_msg)
        if ratio >= 0.40:
            logging.info(f"English override: {en_count}/{len(words_in_msg)} words matched ({ratio:.0%})")
            return "english"

    # Exact single-token matches
    exact_matches = {
        "shona":     ["mhoro", "mhoroi", "makadini", "hesi", "hapana", "ndizvo",
                      "zvakanaka", "wadini", "taura", "kwete"],
        "ndebele":   ["sawubona", "salibonani", "unjani", "yebo", "ngiyabonga",
                      "ngicela", "impela", "kunjani", "hatshi", "kambe"],
        "bemba":     ["mwaiseni", "ulishani", "nalikutemwa", "natotela",
                      "twatotela", "mukwai", "napapata"],
        "chinyanja": ["moni", "zikomo", "pepani", "ndithu", "chonde", "eyaa",
                      "nitandizeni", "nankani"],
        "tonga":     ["mwabuka", "mwalandwa", "ndatotela", "kapati", "iyayi"],
        "lozi":      ["ndalumba", "haa", "kacenu", "muzuhile"],
    }
    for lang, words in exact_matches.items():
        if message_lower in words:
            logging.info(f"Exact match: {message_lower} -> {lang}")
            return lang

    # Keyword scoring — whole-word boundaries only
    language_keywords = {
        "shona": [
            "mhoro", "mhoroi", "makadini", "ndinonzi", "zvakanaka", "ndatenda",
            "pamuviri", "zvigadzirwa", "chirwere", "gomarara", "chibereko",
            "zviratidzo", "chiremba", "kusvotwa", "kurwadziwa",
            "handina", "ndinoda", "zvichava", "zvakadaro",
            "kwete", "hapana", "ndizvo", "zvakafanana",
            # question words & common Shona words that get misdetected as English
            "ndoziva", "nhumbu", "ndine", "ndiri", "ndinoziva",
            "sei", "kuti", "zvii", "vanhu", "muviri", "mazuva",
            "hesi", "ndatenda", "masvingo", "musha", "kuita",
            "zvakanaka", "ndakadaro", "zviripo", "zvinobvira",
        ],
        "ndebele": [
            "sawubona", "salibonani", "unjani", "ngiyabonga", "ngicela",
            "isisu", "umntwana", "imikhiqizo", "umhlaza", "isibeletho",
            "izimpawu", "udokotela", "igazi", "ubuhlungu",
            "angikwazi", "ngifuna", "ukukhulelwa", "abantu",
            "akukho", "impela", "kakhulu",
        ],
        "chinyanja": [
            "moni", "zikomo", "pepani", "ndapota",
            "matenda", "kansa", "zizindikiro", "dokotala",
            "magazi", "zabwino", "sindikudziwa", "ndikufuna",
            "sabata", "zambiri", "thanzo", "mavitamini",
            "nitandizeni", "nankani", "vumo", "mimba",
            "thandizani", "ndimva", "ndikumva", "ndinafuna",
        ],
        "lozi": [
            "ndalumba", "zibonelelo", "kuhula", "kushisa",
            "maviki", "mutango", "mupilo", "mubonelelo",
            "kacenu", "muzuhile", "kimanzibwana", "mulumele",
        ],
        "bemba": [
            "mwaiseni", "nalikutemwa", "natotela", "twatotela", "mukwai",
            "ngafweniko", "cilikwisa", "ubushiku", "ifyakulya",
            "ukubomba", "icisungu", "icibemba", "shaleenipo",
        ],
        "tonga": [
            "mwalumela", "ndatotela", "twatotela", "kapati",
            "mwana musankwa", "mwana mwanakazi",
            "mbomba", "ndisimutwe", "ndatola", "mabbali",
        ],
        "english": [
            "what", "how", "when", "why", "where", "signs", "symptoms",
            "pregnancy", "pregnant", "cervical", "cancer", "health",
            "doctor", "hospital", "treatment", "information", "please",
            "thank", "sorry", "help", "watch", "during", "risky",
        ],
    }

    language_phrases = {
        "chinyanja": ["muli bwanji", "uli ndi chani", "zikomo kwambiri",
                      "muli bwino", "ndili bwino", "nitandizeni nankani"],
        "shona":     ["makadini", "zvakanaka sei", "ndatenda", "ndiriku"],
        "ndebele":   ["unjani wena", "ngiyabonga kakhulu", "sicela ungichazele"],
        "lozi":      ["uli bwanji", "ni bata", "ha ndi zibi", "ndalumba hahulu"],
        "bemba":     ["muli shani", "napapita", "nshishibe", "bushe kuti"],
        "tonga":     ["mwabuka buti", "mwalandwa buti", "ndazwa kwiinda"],
        "english":   ["how are you", "what are", "what is", "can you",
                      "tell me", "i need", "i want", "please tell",
                      "watch out", "signs of", "signs to", "how do i",
                      "how can i", "what should"],
    }

    scores = {lang: 0 for lang in language_keywords}

    for lang, phrases in language_phrases.items():
        for phrase in phrases:
            if phrase in message_lower:
                scores[lang] = scores.get(lang, 0) + 5

    for lang, keywords in language_keywords.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", message_lower):
                scores[lang] = scores.get(lang, 0) + 3

    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        detected_lang = max(scores, key=scores.get)
        logging.info(f"Language scores: {scores} -> {detected_lang}")

        # FIX: use user_states.get safely with fallback
        current_lang = "english"
        if sender and sender in user_states:
            current_lang = user_states[sender].get("language", "english")

        if detected_lang != current_lang and max_score < 3:
            logging.info(f"Low confidence ({max_score}), keeping {current_lang}")
            return current_lang

        return detected_lang

    # FIX: pure ASCII with no keyword match — do NOT default to English.
    # If the user already has a detected language, keep it.
    # Only return English if no prior language context exists.
    if all(ord(c) < 128 for c in message_lower):
        if sender and sender in user_states:
            current = user_states[sender].get("language", "english")
            if current != "english":
                logging.info(f"Pure ASCII, no keyword match — keeping existing language: {current}")
                return current
        logging.info("Pure ASCII with no local-language keyword match and no prior context -> English")
        return "english"

    logging.info("No language detected, defaulting to English")
    return "english"

def is_question(prompt):
    prompt_lower = prompt.lower().strip()
    
    question_indicators = [
        "what", "how", "when", "why", "where", "who", "which", "can", "should", 
        "could", "would", "will", "does", "is", "are", "do you", "tell me about",
        "explain", "describe", "please tell", "i want to know", "i need",
        "kuti", "sei", "ndeipi", "ndiani", "kupi", "zvakadii", "zvinei", 
        "unogona", "ungandiudza", "ndapota tsanangura", "chii", "vanani",
        "rini", "kuti chii", "ndinoda kuziva", "ndapota undiudze",
        "yini", "kanjani", "nini", "ngobani", "kuphi", "kungani", "ngabe",
        "ungangitshela", "ngicela uchaze", "bengifuna ukwazi", "ngitshele",
        "chaza", "ngicela ungichazele", "kutheni", "ubani", "liphi",
        "bushe", "shani", "lili", "mulandu shani", "kwisa", "ngani", "wani",
        "kanshi", "mwe", "bushe kuti", "mukwai", "nalanda", "njisheni",
        "nalefwaya ukwishiba", "napapita njishibe", "cinga", "kabalume",
        "londololeni", "bushe ni", "ifwe", "mwebo",
        "kodi", "bwanji", "liti", "kotani", "kuti", "ndani", "kuti chani",
        "mungandiuze", "ndifunse", "fotokozani", "chonde", "chifukwa",
        "monga bwanji", "ndikufuna kudziwa", "mungandiwuza", "tanifotokozerani",
        "kodi mungandiuze", "pamene", "chiyani", "ndiye",
        "buti", "lili", "kuti", "ngaani", "mboni", "chakuti", "mbubuti",
        "ndapota ndijanye", "ndiyanda kuziba", "mbu", "ingwasi", "kuti kuli",
        "nga", "njise", "ndatola ndiyande", "mwami", "bakwe", "kuti mbuli",
        "mbalimbali", "taamba", "ndapota mundijanye",
        "ñi", "kai", "lini", "cwani", "kwapi", "mulanduñi", "na",
        "unga ni byela", "ndapota taluse", "ni buza", "mang", "kuti ñi",
        "kuli cwani", "wani", "ni batanga kuziba", "ndapota ni talusele",
        "na mutu", "likande", "mwa", "ñilo", "kwa", "ni ku buza",
        "na mwana", "ndapota ni byele"
    ]
    
    has_question_mark = "?" in prompt
    starts_with_question_word = any(prompt_lower.startswith(word + " ") for word in question_indicators)
    contains_question_word = any(" " + word + " " in " " + prompt_lower + " " for word in question_indicators)
    
    return has_question_mark or starts_with_question_word or contains_question_word


def get_pregnancy_data(language):
    if language == "shona":
        return pregnancy_data_shona.pregnancy_data_shona
    elif language == "ndebele":
        return pregnancy_data_ndebele.pregnancy_data_ndebele
    elif language == "chinyanja":
        return pregnancy_data_chinyanja.pregnancy_data_chinyanja
    elif language == "lozi":
        return pregnancy_data_lozi.pregnancy_data_lozi 
    elif language == "bemba":
        return pregnancy_data_bemba.pregnancy_data_bemba
    elif language == "tonga":
        return pregnancy_data_tonga.pregnancy_data_tonga
    else:
        return pregnancy_data.pregnancy_data
        
        
def get_cervical_data(language):
    if language == "shona":       
        return cervical_cancer_data.cervical_cancer_data  
    elif language == "ndebele":       
        return cervical_cancer_data.cervical_cancer_data  
    elif language == "chinyanja":       
        return cervical_cancer_data_chinyanja.cervical_cancer_data_chinyanja  
    elif language == "lozi":        
        return cervical_cancer_data_lozi.cervical_cancer_data_lozi 
    elif language == "bemba":        
        return cervical_cancer_data_bemba.cervical_cancer_data_bemba
    elif language == "tonga":        
        return cervical_cancer_data_tonga.cervical_cancer_data_tonga
    else:
        return cervical_cancer_data.cervical_cancer_data
        

def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    type = "text"
    body = "body"
    content = answer
    image_urls = product_images.image_urls

    if "product_image" in answer:
        product_match = re.search(r'product_image_(\w+)', answer)
        if product_match:
            product_name = product_match.group(1)
            if product_name in image_urls:
                image_url = image_urls[product_name]
                mime_type, _ = guess_type(image_url.split("/")[-1])
                if mime_type and mime_type.startswith("image"):
                    type = "image"
                    body = "link"
                    content = image_url
                    answer = re.sub(r'product_image_\w+', '', answer)

    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": type,
        type: {
            body: content,
            **({"caption": answer.strip()} if type != "text" else {})
        },
    }

    response = requests.post(url, headers=headers, json=data)

    print("Send status:", response.status_code)
    print("Send response:", response.text)

    save_user_conversation(sender, "bot", answer)
    return response

def remove(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)


def handle_language_detection(sender, prompt, phone_id):
    detected_lang = detect_language(prompt, sender)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "registration"
    user_states[sender]["needs_language_confirmation"] = False

    if detected_lang == "shona":
        send("Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.", sender, phone_id)
    elif detected_lang == "ndebele":
        send("Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.", sender, phone_id)
    elif detected_lang == "bemba":
        send("Mwaiseni! Nine Rudo, wakufwailisha wa Dawa Health. Tiyambeni no kulembesha. Napapita, mpeele enamba shakulekelesha shane (4) sha foni yenu.", sender, phone_id)
    elif detected_lang == "chinyanja":
        send("Moni! Ndine Rudo, mphungu wa Dawa Health. Tiyambireni ndi kulembetsa. Chonde ndipatseni manambala anayi omaliza a nambala yanu yafoni.", sender, phone_id)
    elif detected_lang == "tonga":
        send("Mwabuka buti! Ndime Rudo, wakuyambilila wa Dawa Health. Tayambuke kuzyibisya. Ndatola, ndipe zyibalo zyotobela zyane (4) zyanyongola yako.", sender, phone_id)
    elif detected_lang == "lozi":
        send("Mwa bona! Mina ki Rudo, mubasi wa ku thusa wa Dawa Health wa ku kompyuta. A re simule ka ku itambula. Ndapota, nipe dinomolo za mafelele a mane za foni ya hao.", sender, phone_id)
    else:
        send("Hello! I'm Rudo, Dawa Health's virtual assistant. Let's start with registration. Please tell me the last 4 digits of your phone number.", sender, phone_id)
    
    save_single_user_state(sender)


def handle_registration(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if state.get("phone_digits") is None:
        state["phone_digits"] = prompt
        
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=4))
        user_id = f"DH-{prompt}-{random_letters}"
        state["user_id"] = user_id
        
        if lang == "shona":
            send(f"Ndatenda! ID yenyu yakagadzirwa ndeye: {user_id}. Chengetedza ID iyi nekuti ichakumbirwa kumaDawa clinics. Ndingakubatsirei nhasi?", sender, phone_id)
        elif lang == "ndebele":
            send(f"Ngiyabonga! I-ID yakho eyakhiwe ithi: {user_id}. Gcina le ID ngoba izocelwa kumaDawa clinics. Ngingakusiza ngani namuhla?", sender, phone_id)
        elif lang == "bemba":
            send(f"Natotela! ID yenu iyapangwa ni: {user_id}. Sungeni ID iyi pantu ikabombwa ku Dawa clinics. Nga kuti njamfwa shani lelo?", sender, phone_id)
        elif lang == "chinyanja":
            send(f"Zikomo! ID yanu yopangidwa ndi: {user_id}. Sungani ID iyi chifukwa idzafunsidwa kumakliniki a Dawa. Ndingakuthandizireni lero?", sender, phone_id)
        elif lang == "tonga":
            send(f"Twatotela! ID yako yakubikwa nja: {user_id}. Sunga ID eyi nokuba ikaombwa ku Dawa clinics. Ndingakuyandisye lino?", sender, phone_id)
        elif lang == "lozi":
            send(f"Ndalumba! ID ya wena ye e bupilwe ki: {user_id}. Boloka ID ye hantši kakuli u ta buzwa yona kwa makiliniki a Dawa. Nka ku thusa ka mini sunu?", sender, phone_id)
        else:
            send(f"Thank you! Your generated ID is: {user_id}. Keep this ID safe because it'll be asked for at the Dawa clinics. How can I help you today?", sender, phone_id)
        
        state["registered"] = True
        state["step"] = "main_menu"
    
    save_single_user_state(sender)


def handle_follow_up(sender, prompt, phone_id):
    state = user_states[sender]
    # FIX: always read lang fresh from state after any potential reset
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    greeting_words = [
        # English
        "hi", "hello", "hey", "hie",
        # Shona
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        # Ndebele
        "sawubona", "salibonani",
        # Chinyanja
        "moni", "muli bwanji",
        # Tonga
        "mwabuka", "mwabuka buti", "mwalandwa", "mwalandwa buti",
        # Bemba
        "mwaiseni", "muli shani",
        # Lozi
        "mwa bona",
    ]
    if any(prompt_lower == w or re.search(rf"\b{re.escape(w)}\b", prompt_lower) for w in greeting_words):
        reset_conversation(sender)
        # FIX: re-read state and lang after reset
        state = user_states[sender]
        lang = state["language"]
        greet_map = {"shona":"Mhoroi! Ndingakubatsirei nhasi?","ndebele":"Sawubona! Ngingakusiza ngani namuhla?","chinyanja":"Moni! Ndingakuthandizireni lero?","lozi":"Mwa bona! Nka ku thusa ka mini sunu?","tonga":"Moni! Ndingamwafwa shani ilelo?","bemba":"Muli shani! Bushe kuti namwafwa shani lelo?"}
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi"]

    if any(response in prompt_lower for response in no_responses):
        _ask_purchase_interest(sender, phone_id, lang)
        return

    else:
        if len(prompt_lower.split()) > 2:
            _send_thinking(sender, phone_id, lang)

        reply = ask_gemini_general(prompt, lang)
        send(reply, sender, phone_id)
        _send_more_questions(sender, phone_id, lang)

        state["step"] = "general_followup"
        save_single_user_state(sender)


def is_exact_match(text, responses):
    words = re.findall(r"\b\w+\b", text)
    return any(word in responses for word in words)


# ─────────────────────────────────────────────
#  HELPER: send "thinking" indicator
# ─────────────────────────────────────────────
def _send_thinking(sender, phone_id, lang):
    thinking_map = {
        "shona": "Ndiri kufunga...",
        "ndebele": "Ngiyacabangisisa...",
        "chinyanja": "Ndikuganiza...",
        "tonga": "Ndikuganizira...",
        "bemba": "Ndikufwailisha...",
        "lozi": "Ni nahana...",
    }
    send(thinking_map.get(lang, "Let me think..."), sender, phone_id)


# ─────────────────────────────────────────────
#  HELPER: send "any more questions?" prompt
# ─────────────────────────────────────────────
def _send_more_questions(sender, phone_id, lang):
    more_map = {
        "shona": "Pane chimwe chamunoda kubvunza here?",
        "ndebele": "Uneminye imibuzo yini?",
        "chinyanja": "Kodi muli ndi mafunso ena?",
        "tonga": "Uli ne mabvuzo yanga yonse?",
        "bemba": "Uli ne fimo fyandi ifyakulya?",
        "lozi": "O na mabvuzo a mangi?",
    }
    send(more_map.get(lang, "Do you have any more questions?"), sender, phone_id)


def handle_general_followup(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    greeting_words = [
        # English
        "hi", "hello", "hey", "hie",
        # Shona
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        # Ndebele
        "sawubona", "salibonani",
        # Chinyanja
        "moni", "muli bwanji",
        # Tonga
        "mwabuka", "mwabuka buti", "mwalandwa", "mwalandwa buti",
        # Bemba
        "mwaiseni", "muli shani",
        # Lozi
        "mwa bona",
    ]
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset", "help"]

    is_greeting = any(prompt_lower == w or re.search(rf"\b{re.escape(w)}\b", prompt_lower) for w in greeting_words)
    is_reset    = any(kw in prompt_lower for kw in reset_keywords)

    if is_greeting or is_reset:
        reset_conversation(sender)
        # FIX: re-read lang after reset
        state = user_states[sender]
        lang  = state["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Moni! Ndingamwafwa shani ilelo?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde"]
    no_responses  = ["no", "nah", "aiwa", "kwete", "hapana", "nope", "cha", "ayi"]

    if any(r in prompt_lower for r in yes_responses):
        ask_map = {
            "shona": "Bvunzai mubvunzo wenyu.",
            "ndebele": "Ngiyacela ubuze umbuzo wakho.",
            "tonga": "Nkumbira ubvunze mubvuzo wako.",
            "chinyanja": "Chonde funsani funso lanu.",
            "bemba": "Nomba, lwishibe fimo lyobe.",
            "lozi": "Nkumbira ubuze mubvuzo wako.",
        }
        send(ask_map.get(lang, "Please ask your question."), sender, phone_id)
        state["step"] = "general_question"
        save_single_user_state(sender)
        return

    if any(r in prompt_lower for r in no_responses):
        _ask_purchase_interest(sender, phone_id, lang)
        return

    if len(prompt_lower.split()) > 2:
        _send_thinking(sender, phone_id, lang)

    reply = ask_gemini_general(prompt, lang)
    send(reply, sender, phone_id)
    _send_more_questions(sender, phone_id, lang)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def ask_follow_up_question(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    followup_map = {
        "shona": "Pane chimwe chandingakubatsira nacho here?",
        "ndebele": "Ingabe kukhona okunye engingakusiza ngakho?",
        "tonga": "Kuli chinco nchingakusebelesya nacho",
        "chinyanja": "Kodi pali zina zomwe ndingakuthandizireni?",
        "bemba": "Kuli fintu fyalumo nshingafye",
        "lozi": "Ki sina sika ni ka thusa ka sona",
    }
    send(followup_map.get(lang, "Is there anything else I can help you with?"), sender, phone_id)
    
    state["step"] = "follow_up"
    save_single_user_state(sender)


def switch_language_and_respond(sender, prompt, phone_id, current_lang, detected_lang):
    state = user_states[sender]
    state["language"] = detected_lang
    
    current_step = state.get("step", "main_menu")
    logging.info(f"Language switch detected: {current_lang} -> {detected_lang} at step: {current_step}")
    
    if current_step == "main_menu":
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Moni! Ndingamwafwa shani ilelo?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(detected_lang, "Hello! How can I help you today?"), sender, phone_id)
    
    elif current_step == "registration":
        if state.get("phone_digits") is None:
            reg_map = {
                "shona": "Mhoro! Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.",
                "ndebele": "Sawubona! Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.",
                "chinyanja": "Moni! Tiyambireni ndi kulembetsa. Chonde ndipatseni manambala anayi omaliza a nambala yanu yafoni.",
                "lozi": "Mwa bona! A re simule ka ku itambula. Dinomolo za mafelele a lina la wena ki zini za mafelele a mane?",
            }
            send(reg_map.get(detected_lang, "Hello! Let's start with registration. What is the last 4 digits of your number?"), sender, phone_id)
    
    elif current_step == "ask_week":
        week_map = {
            "shona": "Ndapota isa vhiki re pamuviri ",
            "ndebele": "Sicela ufake iviki lokukhulelwa ",
            "chinyanja": "Chonde lowetsani sabata la pakati ",
            "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
        }
        send(week_map.get(detected_lang, "Please enter your pregnancy week number "), sender, phone_id)
    
    save_single_user_state(sender)


# ─────────────────────────────────────────────
#  FIX: Re-detect language on every message
# ─────────────────────────────────────────────

def maybe_update_language(sender, prompt):
    """
    Re-detect language on every incoming message (after registration).
    Updates user state language if a different language is detected.
    Returns the (possibly updated) language.
    """
    state = user_states[sender]
    current_step = state.get("step", "main_menu")

    if current_step in ["language_detection", "registration"]:
        return state.get("language", "english")

    if prompt.strip().isdigit():
        return state.get("language", "english")

    detected = detect_language(prompt, sender)
    current_lang = state.get("language", "english")

    if detected != current_lang:
        logging.info(f"Language switch for {sender}: {current_lang} -> {detected}")
        state["language"] = detected
        save_single_user_state(sender)

    return state["language"]


def handle_main_menu(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    logging.info(f"User {sender} said: '{prompt}' (lowercase: '{prompt_lower}')")
    logging.info(f"Current state: step={state.get('step')}, topic={state.get('topic')}, language={lang}")

    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset", "help"]
    greeting_words = [
        # English
        "hi", "hello", "hey", "hie", "hi there", "good morning", "good afternoon", "good evening",
        # Shona
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        # Ndebele
        "sawubona", "salibonani",
        # Chinyanja
        "moni", "muli bwanji",
        # Tonga
        "mwabuka", "mwabuka buti", "mwalandwa", "mwalandwa buti",
        # Bemba
        "mwaiseni", "muli shani",
        # Lozi
        "mwa bona",
    ]
    
    is_reset = any(keyword in prompt_lower for keyword in reset_keywords)
    is_greeting = (
        any(prompt_lower.strip() == word for word in greeting_words) or
        any(re.search(rf"\b{re.escape(word)}\b", prompt_lower) for word in greeting_words)
    )
    
    if is_greeting or is_reset:
        reset_conversation(sender)
        # FIX: re-read state and lang after reset
        state = user_states[sender]
        lang = state["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Moni! Ndingamwafwa shani ilelo?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    current_step = state.get("step")
    
    if current_step == "ask_another_week":
        handle_another_week(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_more_info":
        handle_cervical_more_info(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_question_number":
        handle_cervical_question_number(sender, prompt, phone_id)
        return
        
    if current_step == "keep_learning":
        handle_keep_learning(sender, prompt, phone_id)
        return
        
    if current_step == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
        return
        
    if current_step == "product_inquiry":
        handle_purchase_response(sender, prompt, phone_id)
        return
        
    if current_step == "confirm_purchase":
        handle_purchase_confirmation(sender, prompt, phone_id)
        return

    if current_step == "general_followup":
        return handle_general_followup(sender, prompt, phone_id)

    if state.get("step") == "choose_info_type":
        if prompt_lower in ["1", "general", "information", "info", "ruzivo", "ulwazi", "zambiri"]:
            if state.get("topic") == "maternal":
                state["step"] = "ask_week"
                week_map = {
                    "shona": "Ndapota isa vhiki re pamuviri ",
                    "ndebele": "Sicela ufake iviki lokukhulelwa ",
                    "chinyanja": "Chonde lowetsani sabata la pakati ",
                    "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
                    "tonga": "Ndatola, ingila nhwiiiki ya kubeleka mwana ",
                    "bemba": "Napapita, ingisha umulungu wa pa nkundi ",
                }
                send(week_map.get(lang, "Please enter your pregnancy week number:"), sender, phone_id)
            elif state.get("topic") == "cervical":
                cervical_data = get_cervical_data(lang)
                if cervical_data and len(cervical_data) > 0:
                    send(str(cervical_data[0]), sender, phone_id)
                else:
                    no_data_map = {
                        "shona": "Ndine urombo, handina kuwana ruzivo rwe cervical cancer parizvino.",
                        "ndebele": "Uxolo, anginayo imininingwane ye-cervical cancer okwamanje.",
                        "chinyanja": "Pepani, sindinapeze zambiri za cervical cancer panopa.",
                        "lozi": "Ndine u luvile, sina kungafumula zintu za kankere ya sibete sunu.",
                    }
                    send(no_data_map.get(lang, "Sorry, I couldn't find cervical cancer information at the moment."), sender, phone_id)
                
                ask_cervical_more_info(sender, phone_id)
            save_single_user_state(sender)
            return

        elif prompt_lower in ["2", "specific", "question", "questions", "mubvunzo", "umbuzo", "funso"]:
            if state.get("topic") == "maternal":
                state["step"] = "maternal_question_choice"
                q_map = {
                    "shona": (
                        "Sarudza mubvunzo:\n"
                        "1. Ndezvikaita zviratidzo zvepamuviri?\n"
                        "2. Ndeapi marairiro ezvokudya?\n"
                        "3. Ndingafanire kuona chiremba riini?"
                    ),
                    "ndebele": (
                        "Khetha umbuzo:\n"
                        "1. Ngabe yiziphi izimpawu zesisu?\n"
                        "2. Ngabe yimaphi amathiphu okudla?\n"
                        "3. Ngabe kufanele ngibone udokotela nini?"
                    ),
                    "chinyanja": (
                        "Sankhani funso:\n"
                        "1. Ndi zizindikiro zotani za pakati?\n"
                        "2. Ndi malangizo otani okudya?\n"
                        "3. Ndingafunire kuona dokotala liti?"
                    ),
                    "lozi": (
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Zibonelelo ze ku imelela mwana zezi ntini?\n"
                        "2. Ni maano a ku nwa zintu za bupilo a ka landelwa?\n"
                        "3. Nini nka ya kwa dokotela?"
                    ),
                    "tonga": (
                        "Sarudza mubvuzo:\n"
                        "1. Zizyo zyakubaa mwana nzi?\n"
                        "2. Malangizo aakudya nzi?\n"
                        "3. Ndingafwile kubona dokotela lili?"
                    ),
                    "bemba": (
                        "Sala ilipusho:\n"
                        "1. Ifyo balenanga pa nkundi nshi?\n"
                        "2. Amabumba ya kulya nshi?\n"
                        "3. Ngatuke kwa dokota liti?"
                    ),
                }
                send(q_map.get(lang, (
                    "You can choose a question or ask any of your own.\n"
                    "1. What are common pregnancy symptoms?\n"
                    "2. What nutrition tips should I follow?\n"
                    "3. When should I see a doctor?"
                )), sender, phone_id)
            elif state.get("topic") == "cervical":
                state["step"] = "cervical_question_choice"
                cq_map = {
                    "shona": (
                        "Sarudza mubvunzo:\n"
                        "1. Chii chinonzi cervical cancer?\n"
                        "2. Ndezvipi zviratidzo zvekutanga zvecervical cancer?\n"
                        "3. Chii chinokonzera cervical cancer?"
                    ),
                    "ndebele": (
                        "Khetha umbuzo:\n"
                        "1. Yini i-cervical cancer?\n"
                        "2. Ngabe yiziphi izimpawu zokuqala ze-cervical cancer?\n"
                        "3. Yini ebangela i-cervical cancer?"
                    ),
                    "chinyanja": (
                        "Sankhani funso:\n"
                        "1. Ndi chiyani cervical cancer?\n"
                        "2. Ndi zizindikiro zotani zoyamba za cervical cancer?\n"
                        "3. Ndi chiyani chimayambitsa cervical cancer?"
                    ),
                    "lozi": (
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Kankere ya sibete sa bomme ki yini?\n"
                        "2. Zibonelelo za kutanga za kankere ya sibete zezi ntini?\n"
                        "3. Zini zi bakela kankere ya sibete?"
                    ),
                    "tonga": (
                        "Sarudza mubvuzo:\n"
                        "1. Kansa ya mulomo wa cibeleko nzi?\n"
                        "2. Zizyo zyakutanga zya kansa ya mulomo wa cibeleko nzi?\n"
                        "3. Chiyambitsa kansa ya mulomo wa cibeleko nzi?"
                    ),
                    "bemba": (
                        "Sala ilipusho:\n"
                        "1. Kansa ya cibeleshi nshi?\n"
                        "2. Ifyo balenanga mu kutanga kwa kansa ya cibeleshi nshi?\n"
                        "3. Chiyambisha kansa ya cibeleshi nshi?"
                    ),
                }
                send(cq_map.get(lang, (
                    "You can choose a question or ask any of your own.\n"
                    "1. What is cervical cancer?\n"
                    "2. What are the early symptoms of cervical cancer?\n"
                    "3. What causes cervical cancer?"
                )), sender, phone_id)
            save_single_user_state(sender)
            return

        else:
            invalid_map = {
                "shona": "Pindura ne '1' kuti uwane ruzivo kana '2' kuti ubvunze mibvunzo.",
                "ndebele": "Phendula ngo-'1' ukuze uthole ulwazi noma '2' ukuze ubuze imibuzo.",
                "chinyanja": "Yankhani ndi '1' kuti mupeze zambiri kapena '2' kuti mufunse mafunso.",
                "tonga": "Ndapota pindula na '1' ku lwisisa zintu ka bonya noma '2' ku mubuzo wa nene",
                "bemba": "Yasuka na '1' ukufuna ubunga kana '2' ukufuna ilipusho.",
                "lozi": "Arabela ka '1' ku fumana litaba kamba '2' ku buza lipuzo.",
            }
            send(invalid_map.get(lang, "Please reply '1' for information or '2' for questions."), sender, phone_id)
            return

    if state.get("step") == "ask_week":
        try:
            week = int(re.sub(r"\D", "", prompt_lower))
            if 1 <= week <= 40:
                info_text = get_pregnancy_data(lang)
                if lang == "shona":
                    pattern = rf"\*Vhiki {week}:.*?(?=\*Vhiki {week+1}:|\Z)"
                elif lang == "ndebele":
                    pattern = rf"\*Iviki {week}:.*?(?=\*Iviki {week+1}:|\Z)"
                elif lang == "chinyanja":
                    pattern = rf"\*Sabata {week}:.*?(?=\*Sabata {week+1}:|\Z)"
                elif lang == "lozi":
                    pattern = rf"\*Sunda {week}:.*?(?=\*Sunda {week+1}:|\Z)"
                elif lang == "bemba":
                    pattern = rf"\*Umulungu {week}:.*?(?=\*Umulungu {week+1}:|\Z)"
                elif lang == "tonga":
                    pattern = rf"\*Nhwiiiki {week}:.*?(?=\*Nhwiiiki {week+1}:|\Z)"
                else:
                    pattern = rf"\*Week {week}:.*?(?=\*Week {week+1}:|\Z)"
                    
                match = re.search(pattern, info_text, re.S)
                if match:
                    header_map = {
                        "shona": f"Ruzivo rwe *Vhiki {week}:*\n\n",
                        "ndebele": f"Ulwazi lwe *Iviki {week}:*\n\n",
                        "chinyanja": f"Zambiri za *Sabata {week}:*\n\n",
                        "lozi": f"Yezi zintu za lwisisa ka bonya ku *Sunda {week}:*\n\n",
                        "bemba": f"Icibeela ca *Mulungu {week}:*\n\n",
                        "tonga": f"Cibeela ca *Nhwiiiki {week}:*\n\n",
                    }
                    header = header_map.get(lang, f"Here's information for *Week {week}:*\n\n")
                    send(f"{header}{match.group(0)}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    no_week_map = {
                        "shona": "Hapana ruzivo rwevhiki iyi.",
                        "ndebele": "Alukho ulwazi lwaleviki.",
                        "chinyanja": "Palibe zambiri za sabata ili.",
                        "lozi": "Sina zintu za ku fumwa ka viki ye.",
                        "bemba": "Tapali icibeela ca umulungu wu.",
                        "tonga": "Tana cibeela ca nhwiiiki iyi.",
                    }
                    send(no_week_map.get(lang, "No data available for that week."), sender, phone_id)
                    ask_another_week(sender, phone_id)
        except ValueError:
            invalid_week_map = {
                "shona": "Ndapota pinda nhamba chaiyo yevhiki kubva pa 1 kusvika pa 40.",
                "ndebele": "Sicela ufake inombolo yeviki evumelekile ephakathi kuka-1 no-40.",
                "chinyanja": "Chonde lowetsani nambala yoyenera ya sabata kuchokera pa 1 mpaka 40.",
                "lozi": "Ndapota faka linomolo la viki le li le ka 1 ku ya ka 40.",
                "bemba": "Napapita, ingisha umulungu ukufuma pa 1 ukufika pa 40.",
                "tonga": "Ndatola, ingila nhwiiiki kuzwa 1 kusika 40.",
            }
            send(invalid_week_map.get(lang, "Please enter a valid week number between 1 and 40."), sender, phone_id)
            ask_another_week(sender, phone_id)
        return  

    if state.get("step") == "maternal_question_choice":
        if prompt_lower in ["1", "symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
            sym_map = {
                "shona": "Zviratidzo zvepamuviri zvinosanganisira kusvotwa, kuneta, kuvava mazamu, uye kuchinja mweya.",
                "ndebele": "Izimpawu zesisu zihlanganisa isicanucanu, ukukhathala, ubuhlungu bezebelé, nokushintsha kwemizwa.",
                "chinyanja": "Zizindikiro za pakati zimaphatikizapo kusanza, kulemba, kubvutika mabele, ndi kusintha kwa maganizo.",
                "lozi": "Limpande ze twayelehileng za buimana li akaretsa ho nyekeloa ke pelo, kukhathala, kubaba kwa matete ni kupotoloka kwa maikuto.",
                "tonga": "Zizyo zyakubaa mwana nzinzi kusanza, kukola, kuuma mabbele, anilizyo kusanduka.",
                "bemba": "Ifyo balenanga pa nkundi fifikwata ulwimpa, ubulanda, ukulwala amabere, no kusanduka kwa malinganya.",
            }
            send(sym_map.get(lang, "Common pregnancy symptoms include nausea, fatigue, breast tenderness, and mood swings."), sender, phone_id)
    
        elif prompt_lower in ["2", "nutrition", "zvokudya", "ukudla", "kudya"]:
            nut_map = {
                "shona": "Marairiro ezvokudya: Idya chikafu chakaringana, wedzera folic acid uye iron, uye nwa mvura yakawanda.",
                "ndebele": "Amathiphu okudla: Yidla ukudla okunempilo, khulisa i-folic acid ne-iron, futhi uhlale unamandla.",
                "chinyanja": "Malangizo okudya: Idyani chakudya chabwino, onjezerani folic acid ndi iron, ndipo muzikhala ndi madzi.",
                "lozi": "Litaba za swakudya: Ja swakudya se se lekalekanang, engetsa kufumana folic acid ni iron, mi u nne u nwa mezi a mangi.",
                "tonga": "Malangizo aakudya: Lya chakudya chakwe, engesha folic acid ni iron, anilizyo nwa maanzi amanji.",
                "bemba": "Amabumba ya kulya: Lya ifya kulya ifya balanso, engesha folic acid ni iron, kabili nwa amenshi ayengi.",
            }
            send(nut_map.get(lang, "Nutrition tips: Eat balanced meals, increase folic acid and iron intake, and stay hydrated."), sender, phone_id)
    
        elif prompt_lower in ["3", "doctor", "chiremba", "udokotela", "dokotala"]:
            doc_map = {
                "shona": "Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.",
                "ndebele": "Iya kudokotela uma unobuhlungu obukhulu, ukuphuma kwegazi okukhulu, noma imfiva ephezulu.",
                "chinyanja": "Pitani kudokotala ngati muli ndi kupweteka kwakukulu, kutuluka magazi ambiri, kapena malungo apamwamba.",
                "lozi": "Bona ngaka kapili ha u ka ba ni buhlungu bo boholo, kuelwa mali a mangi, kamba mufufutso o mutuna.",
                "tonga": "Bona dokotela kapeli naa uli a kupwetekwa kwakuuluka, kutuluka magazi amanji, naa malungo apamwamba.",
                "bemba": "Enda kwa dokota kapili naa uli na ubunono bwakucila, ukutuluka nko yamene, kamba ifufumiti ifyakucila.",
            }
            send(doc_map.get(lang, "See a doctor immediately if you experience severe pain, heavy bleeding, or high fever."), sender, phone_id)
    
        else:
            _send_thinking(sender, phone_id, lang)
            gemini_response = ask_gemini(prompt, lang)  
            send(gemini_response, sender, phone_id)
        
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    if state.get("step") == "cervical_question_choice":
        if prompt_lower in ["1", "what is it", "what is cervical cancer", "chii", "yini", "chiyani"]:
            cc_what_map = {
                "shona": "Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.",
                "ndebele": "I-cervical cancer yisifo se-cervix, ingxenye engezansi yesibeletho ehlobene nesibeletho. Yisifo somhlaza sesibili esivame kakhulu emhlabeni wonke futhi yisifo esivame kakhulu kwabesifazane eZambia. Isifo esingavinjwa futhi singelapheka, ikakhulukazi uma sitholakala ngokushesha.",
                "chinyanja": "Cervical cancer ndi matenda a cervix, gawo lotsika la chibereko lomwe limagwirizana ndi chibereko. Ndimatenda a kansa wachiwiri omwe amapezeka kwambiri padziko lapansi ndipo ndi omwe amachitika kwambiri kwa amayi ku Zambia. Matenda omwe angapweke ndi opatsirika, makamaka akadziwika msanga.",
                "lozi": "Kansa ya mulomo wa popelo ki malwale a mulomo wa popelo, sipande sa fafasi sa popelo se si kopanya kwa mukutu wa botsadi. Ki kansa ya bobeli e atile hahulu kwa basali mwa lifasi kaufela, mi ki yona e atile hahulu kwa basali mwa Zambia. Ki malwale a ka thibelwa ni ku alafiwa, haholoholo ha a lemohuoa kapili.",
                "tonga": "Kansa ya mulomo wa cibeleko ndi matenda a mulomo wa cibeleko, chipande chakusika cha cibeleko chomwe chimayambana ndi cibeleko. Ndi matenda a kansa wachiwiri omwe amapezeka kwambiri padziko lapansi.",
                "bemba": "Kansa ya cibeleshi ndi ubwafya bwa mulomo wa cibeleshi. Ndi ubwafya bwachibili ubwafumina ukufuma ku kansa panse yonse.",
            }
            send(cc_what_map.get(lang, "Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. It is the second most common female malignancy worldwide and the most common in females in Zambia. It is a preventable and treatable disease, especially when detected early."), sender, phone_id)

        elif prompt_lower in ["2", "symptoms", "early symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
            cc_sym_map = {
                "shona": "Mumatanho ekutanga, cervical cancer kazhinji haina zviratidzo zvinooneka. Ndokusaka kuongororwa nguva nenguva kwakakosha. Sezvo cancer ichikura, zviratidzo zvinogona kusanganisira kubuda ropa kusingawanzo, kubuda kwezvipembenene zvinonhuwa, kana kurwadziwa panguva yekuita bonde.",
                "ndebele": "Ezitebhisini zokuqala, i-cervical cancer ivamise ukungabi nezimpawu ezibonakalayo. Yingakho ukuhlolwa ngesikhathi esithile kubalulekile. Njengoba umhlaza ukhula, izimpawu zingahlanganisa ukuphuma kwegazi okungajwayelekile, ukuphuma kokomkhando olunephunga elibi, noma ubuhlungu ngesikhathi sokwenza ucansi.",
                "chinyanja": "M'magawo oyamba, cervical cancer imayambira mosazindikika. Ndi chifukwa chake kuyezetsa nthawi ndi nthawi ndi kofunikira. Pomwe kansa ikukula, zizindikiro zingakhale kutuluka magazi osayembekezereka, kutuluka kwa chinyezi choipa, kapena kupweteka panthawi ya kugonana.",
                "lozi": "Ka nako ya makalelo, kansa ya mulomo wa sibeleko ha i na mabonelo a bonahala. Ki sona se si ama ku lekolwa ka linako za nako ku ba kwa butokwa. Ha kansa i hula, mabonelo a kona ku akaretsa kuelwa mali ka linako ze sa lebelelwi, ku zwahela kwa tumelo ye nuna, kamba buhlungu bo ba teñi ha ku eza za bunde.",
                "tonga": "Mukutanga kwa matenda, kansa ya mulomo wa cibeleko imaziyizya mosazindikika. Ndi chifukwa chake kuyezetsa nthawi ndi nthawi ndi kofunikira.",
                "bemba": "Mu nsanga ya imituntumuko, kansa ya cibeleshi ifwilika ukuba takuli ifyo balenanga ifilumba. Ndi ifyo ifikoshi ukuyeshiwa nthawi na nthawi.",
            }
            send(cc_sym_map.get(lang, "In its early stages, cervical cancer often has no noticeable symptoms. This is why regular screening is so important. As the cancer progresses, symptoms may include unusual vaginal bleeding (between periods, after sex, or after menopause), foul-smelling vaginal discharge, or pain during sexual intercourse."), sender, phone_id)

        elif prompt_lower in ["3", "causes", "what causes it", "chikonzero", "izimbangela", "zoyambitsa"]:
            cc_cause_map = {
                "shona": "Kazhinji, cervical cancer inokonzerwa nehutachiona husingaperi hweHuman Papilloma Virus (HPV). HPV ihutachiona hwakajairika, hunotapuriranwa nekusangana pabonde. Kunyange immune system yemuviri ichibvisa hutachiona muvanhu vazhinji, hutachiona husingaperi hunogona kukonzera shanduko yamasero inogona kuzopedzisira yaita cancer.",
                "ndebele": "Ezimeni zonke, i-cervical cancer ibangelwa ukutheleleka okungapheli kwe-Human Papilloma Virus (HPV). I-HPV igciwane elivamile, elidluliselwa ngocansi. Ngenkathi amasosha omzimba emuncela igciwane kubantu abaningi, ukutheleleka okungapheli kungaholela ekushintsheni kwamaseli okungajwayelekile okungase igcine kube umhlaza.",
                "chinyanja": "M'magawo onse, cervical cancer imayambitsidwa ndi matenda osatha a Human Papilloma Virus (HPV). HPV ndi matenda amene amapezeka kwambiri, omwe amatengedwa pogonana. Pomwe immune system ya thupi imatulutsa matenda mwa anthu ambiri, matenda osatha angayambitse kusintha kwa maselo komwe kungatheka kukhala kansa.",
                "lozi": "Mwa mikwa kaufela, kansa ya mulomo wa sibeleko i bakiwa ki kulwala ka nako ye telele kwa Human Papilloma Virus (HPV). HPV ki bulwasi bo bu atile hahulu, bo bu fetisezwa ka ku eza za bunde. Niha mili wa mutu u fanga bulwasi ku batu ba bañata, ku lwala ka nako ye telele ku kona ku leza licinceho za liseli ze si za twanelo ze kona ku isa kwa kansa.",
                "tonga": "Mu nyengo yonse, kansa ya mulomo wa cibeleko imayambitsidwa ndi matenda osatha a HPV. HPV ndi matenda amene amapezeka kwambiri, omwe amatengedwa pogonana.",
                "bemba": "Mu nsanga yonse, kansa ya cibeleshi iyambilisha na ubwafya bwa nthawi yonse ya HPV (Human Papilloma Virus). HPV ni ubwafya ubutangwa pangono, ubutampwa ku kupanga umwenda.",
            }
            send(cc_cause_map.get(lang, "In almost all cases, cervical cancer is caused by persistent infection with the Human Papilloma Virus (HPV). HPV is a very common, sexually transmitted virus. While the body's immune system clears the virus in most people, a persistent infection can lead to abnormal cell changes that may eventually develop into cancer."), sender, phone_id)

        else:
            _send_thinking(sender, phone_id, lang)
            gemini_response = ask_gemini_cancer(prompt, lang)  
            send(gemini_response, sender, phone_id)
            
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    maternal_keywords = ["pamuviri", "pakati", "pregnancy", "pregnant", "baby", "maternal", "nhumbu"]
    question_words = ["what", "how", "when", "why", "can", "should", "kuti", "sei", "ngani", "kodi", "bwanji", "chifukwa", "ndeipi", "ndiani", "nzira", "zviratidzo", "zizindikiro"]
    
    is_direct_question = (
        any(keyword in prompt_lower for keyword in maternal_keywords) and 
        any(question_word in prompt_lower for question_word in question_words)
    )
    
    if is_direct_question:
        _send_thinking(sender, phone_id, lang)
        gemini_response = ask_gemini(prompt, lang)
        send(gemini_response, sender, phone_id)
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    gemini_reply = ask_gemini_general(prompt, lang)
    send(gemini_reply, sender, phone_id)
    _send_more_questions(sender, phone_id, lang)
    
    state["step"] = "general_followup"
    save_single_user_state(sender)
    return
    

def _ask_purchase_interest(sender, phone_id, lang):
    """Ask the user if they'd like to purchase products, then show categories + samples."""
    state = user_states[sender]
    ask_map = {
        "shona": "Ungada here kutenga zvimwe zvezvigadzirwa zvedu? ",
        "ndebele": "Ungathanda ukuthenga noma yimuphi imikhiqizo yethu? ",
        "chinyanja": "Kodi mukufuna kugula zinthu zina mu zithu zathu? ",
        "tonga": "Ungafuna kugula zintu zina mu zinthu zyeesu? ",
        "bemba": "Ufuna ukugula imisansa yeesu? ",
        "lozi": "Kana u bata ku landa swakupila sa luna? ",
    }
    send(ask_map.get(lang, "Would you like to purchase any of our products?"), sender, phone_id)
    state["step"] = "shop_interest"
    save_single_user_state(sender)


def _send_shop_categories(sender, phone_id, lang):
    """Send category list with 1-2 sample products each."""
    lines = []
    header_map = {
        "shona": "🛒 Makategi eZvigadzirwa:\n",
        "ndebele": "🛒 Imigqa Yemikhiqizo:\n",
        "chinyanja": "🛒 Mitundu ya Zinthu:\n",
        "tonga": "🛒 Mitundu ya Zinthu:\n",
        "bemba": "🛒 Imibeko ya Imisansa:\n",
        "lozi": "🛒 Mibeko ya Swakupila:\n",
    }
    lines.append(header_map.get(lang, "🛒 Product Categories:\n"))

    for idx, (cat_name, items) in enumerate(products_by_category.items(), 1):
        lines.append(f"*{idx}. {cat_name}*")
        for item in items[:2]:
            lines.append(f"   • {item['name']} — {item['price']}")
        if len(items) > 2:
            more_map = {
                "shona": f"   ...uye zvimwe {len(items)-2}",
                "ndebele": f"   ...namanye a-{len(items)-2}",
                "chinyanja": f"   ...ndi ena {len(items)-2}",
                "tonga": f"   ...na zina {len(items)-2}",
                "bemba": f"   ...na fimo {len(items)-2}",
                "lozi": f"   ...ni ze ñwi {len(items)-2}",
            }
            lines.append(more_map.get(lang, f"   ...and {len(items)-2} more"))
        lines.append("")

    prompt_map = {
        "shona": "Tumira nhamba yekategi kuti uone zvigadzirwa zvose, kana udza zita rechigadzirwa chaunoda kutenga.",
        "ndebele": "Thumela inombolo yomugqa ukuze ubone wonke umkhiqizo, noma sitshele igama lomkhiqizo ofuna ukuthenga.",
        "chinyanja": "Tumizani nambala ya gulu kuti muone zinthu zonse, kapena uzani dzina la chinthu mukufuna kugula.",
        "tonga": "Tuma nambala ya gulu kuti uone zinthu zonse, kapena umbe dzina la chinthu cha kugula.",
        "bemba": "Tuma inomba ya icigaba ukufuna ukubona imisansa yonse, noma ulanda ishina lya imisansa ufuna ukugula.",
        "lozi": "Lumeza nomolo ya sibaka ku bona swakupila kaufela, kamba u bulele libizo la swakupila u bata ku landa.",
    }
    lines.append(prompt_map.get(lang, "Send the category number to see all products, or tell us the name of the product you'd like to order."))

    send("\n".join(lines), sender, phone_id)
    state = user_states[sender]
    state["step"] = "shop_browse"
    save_single_user_state(sender)


def _interpret_shop_intent(prompt_lower):
    """Return 'browse', 'decline', or 'unknown' based on flexible intent matching."""
    browse_signals = [
        "yes", "yeah", "yep", "please", "sure", "ok", "okay", "alright",
        "ehe", "hongu", "ndizvo", "inde", "yebo",
        "product", "products", "what do you have", "what have you got",
        "show me", "see", "available", "list", "categories", "what can i",
        "buy", "order", "purchase", "get", "want", "need", "looking for",
        "zvigadzirwa", "zvinhu", "imikhiqizo", "zinthu", "imisansa", "swakupila",
    ]
    decline_signals = [
        "no", "nah", "nope", "not really", "not now", "later", "goodbye", "bye",
        "hapana", "kwete", "aiwa", "a'a", "ayi", "cha",
    ]
    if any(s in prompt_lower for s in browse_signals):
        return "browse"
    if any(s in prompt_lower for s in decline_signals):
        return "decline"
    return "unknown"


def handle_shop_interest(sender, prompt, phone_id):
    """Handle flexible intent to browse/purchase products."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        bye_map = {
            "shona": "Zvakanaka! Iva nezuva rakanaka. Tanga patsva nekuti 'hesi' kana uine mimwe mibvunzo.",
            "ndebele": "Kulungile! Ube nosuku oluhle. Qala kabusha ngo-'unjani' uma uneminye imibuzo.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Yambani ndi 'muli bwanji' ngati muli ndi mafunso.",
            "tonga": "Chabwino! Mube abusiku bwabwino. Yambani ndi 'mwabuka buti' ngaa muli ne mafunso.",
            "bemba": "Cino cino! Mubelele bwino. Tambuleni ndi 'mwaiseni' nga muli na ifyafyala.",
            "lozi": "Ho lokile! Mube ni lizazi le linde. Qalisa ndi 'mwa bona' ha mu na lipuzo.",
        }
        send(bye_map.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id)
        reset_conversation(sender)
    else:
        # Unknown intent — show categories, most helpful default
        _send_shop_categories(sender, phone_id, lang)


def handle_shop_browse(sender, prompt, phone_id):
    """Handle category number or product name selection during browsing."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    categories = list(products_by_category.keys())

    # Check if user typed a category number
    if prompt_lower.isdigit():
        idx = int(prompt_lower) - 1
        if 0 <= idx < len(categories):
            cat_name = categories[idx]
            all_items = products_by_category[cat_name]
            lines = [f"🏥 *{cat_name}*\n"]
            for i, item in enumerate(all_items, 1):
                lines.append(f"{i}. {item['name']}")
                lines.append(f"   💰 {item['price']} | 📦 {item['availability']}")
                lines.append(f"   {item['description']}\n")

            order_map = {
                "shona": "\nUngada here kuodha chimwe chezvigadzirwa izvi? Pindura 'hongu' uye udza zita rechigadzirwa, kana 'aiwa'.",
                "ndebele": "\nUngathanda ukuodha noma yimuphi yale mikhiqizo? Phendula 'yebo' ubese usitshele igama lomkhiqizo, noma 'cha'.",
                "chinyanja": "\nKodi mukufuna kugula chinthu cha zinthu izi? Yankha 'inde' ndipo uzani dzina la chinthu, kapena 'ayi'.",
                "tonga": "\nUngafuna kugula chintu cha zinthu izi? Yankha 'inde' ndipo umbe dzina la chinthu, kapena 'ayi'.",
                "bemba": "\nUfuna ukugula imisansa iyi? Yasuka 'inde' ulanda ishina lya imisansa, noma 'ayi'.",
                "lozi": "\nKana u bata ku landa se si liñwi sa swakupila se? Arabela 'inde' u bulele libizo, kamba 'ayi'.",
            }
            lines.append(order_map.get(lang, "\nWould you like to order any of these products? Reply 'yes' and tell us the product name, or 'no'."))
            send("\n".join(lines), sender, phone_id)
            state["step"] = "shop_order_decision"
            state["shop_category"] = cat_name
            save_single_user_state(sender)
            return
        else:
            invalid_map = {
                "shona": f"Nhamba isiriyo. Ndapota sarudza pakati pa 1 ne {len(categories)}.",
                "ndebele": f"Inombolo engavumelekile. Ngicela ukhethe phakathi kuka-1 no-{len(categories)}.",
                "chinyanja": f"Nambala yolakwika. Chonde sankhani pakati pa 1 ndi {len(categories)}.",
                "tonga": f"Nambala yolakwika. Ndatola, sankhani pakati pa 1 ndi {len(categories)}.",
                "bemba": f"Inomba yolakwika. Napapata, sala pakati pa 1 na {len(categories)}.",
                "lozi": f"Nomolo e si ya. Ndapota, khetha ku zwana 1 ku ya ku {len(categories)}.",
            }
            send(invalid_map.get(lang, f"Invalid number. Please choose between 1 and {len(categories)}."), sender, phone_id)
            return

    # Check if user typed a product name — treat as direct order intent
    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
        return

    # Otherwise re-prompt
    _send_shop_categories(sender, phone_id, lang)


def handle_shop_order_decision(sender, prompt, phone_id):
    """After browsing a category, did user want to order?"""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]

    if any(r in prompt_lower for r in yes_responses):
        # Ask which product by name
        ask_which_map = {
            "shona": "Zvakanaka! Nyora zita rechigadzirwa chaunoda kuodha.",
            "ndebele": "Kulungile! Bhala igama lomkhiqizo ofuna ukuwodha.",
            "chinyanja": "Chabwino! Lemba dzina la chinthu chomwe mukufuna kugula.",
            "tonga": "Chabwino! Lema dzina la chinthu cha kugula.",
            "bemba": "Cino cino! Lemba ishina lya imisansa ufuna ukugula.",
            "lozi": "Ho lokile! Ñola libizo la swakupila u bata ku landa.",
        }
        send(ask_which_map.get(lang, "Great! Please type the name of the product you'd like to order."), sender, phone_id)
        state["step"] = "shop_product_name"
        save_single_user_state(sender)

    elif any(r in prompt_lower for r in no_responses):
        # Go back to categories
        see_more_map = {
            "shona": "Zvakanaka! Ungada here kuona mamwe makategi? ",
            "ndebele": "Kulungile! Ungathanda ukubona eminye imigqa? ",
            "chinyanja": "Chabwino! Kodi mukufuna kuona mitundu ina? ",
            "tonga": "Chabwino! Ungafuna kuona mitundu ina? ",
            "bemba": "Cino cino! Ufuna ukubona imibeko ina? ",
            "lozi": "Ho lokile! Kana u bata ku bona mibeko ina? ",
        }
        send(see_more_map.get(lang, "Alright! Would you like to see other categories?"), sender, phone_id)
        state["step"] = "shop_more_categories"
        save_single_user_state(sender)
    else:
        # Maybe they typed a product name directly
        all_products_flat = [p for items in products_by_category.values() for p in items]
        matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            _ask_quantity(sender, phone_id, lang, matched["name"])
        else:
            ask_yes_no_map = {
                "shona": "Ndapota pindura 'hongu' kana 'aiwa'.",
                "ndebele": "Ngicela uphendule 'yebo' noma 'cha'.",
                "chinyanja": "Chonde yankha 'inde' kapena 'ayi'.",
                "tonga": "Ndatola, yankha 'inde' kapena 'ayi'.",
                "bemba": "Napapata, yasuka 'inde' kapena 'ayi'.",
                "lozi": "Ndapota, arabela 'inde' kamba 'ayi'.",
            }
            send(ask_yes_no_map.get(lang, "Please reply 'yes' or 'no'."), sender, phone_id)


def handle_shop_product_name(sender, prompt, phone_id):
    """User typed a product name to order."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)

    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
    else:
        not_found_map = {
            "shona": "Handina kuwana chigadzirwa ichocho. Ndapota nyora zita rakanangana rechigadzirwa kubva kumureza.",
            "ndebele": "Angitholi umkhiqizo lowo. Ngicela ubhale igama elicacile lomkhiqizo uvela ohlwini.",
            "chinyanja": "Sindipeza chinthu ichi. Chonde lemba dzina loyenera la chinthu kuchokera pamndandanda.",
            "tonga": "Tana kupeza chinthu ichi. Ndatola, lema dzina loyenera la chinthu kuchokera pamndandanda.",
            "bemba": "Nshasangile imisansa iyo. Napapata, lemba ishina leyenera lya imisansa ku mendenda.",
            "lozi": "Ha ni fumani swakupila se. Ndapota, ñola libizo le li nepahala la swakupila ku luhelo.",
        }
        send(not_found_map.get(lang, "I couldn't find that product. Please type the exact product name from the list."), sender, phone_id)


def handle_shop_more_categories(sender, prompt, phone_id):
    """Handle flexible intent after asking if user wants to see more categories."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        bye_map = {
            "shona": "Zvakanaka! Iva nezuva rakanaka. Tanga patsva nekuti 'hesi' kana uine mimwe mibvunzo.",
            "ndebele": "Kulungile! Ube nosuku oluhle. Qala kabusha ngo-'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Yambani ndi 'muli bwanji'.",
            "tonga": "Chabwino! Mube abusiku bwabwino. Yambani ndi 'mwabuka buti'.",
            "bemba": "Cino cino! Mubelele bwino. Tambuleni ndi 'mwaiseni'.",
            "lozi": "Ho lokile! Mube ni lizazi le linde. Qalisa ndi 'mwa bona'.",
        }
        send(bye_map.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id)
        reset_conversation(sender)
    else:
        _send_shop_categories(sender, phone_id, lang)


def _ask_quantity(sender, phone_id, lang, product_name):
    """Ask the user for quantity of the selected product."""
    state = user_states[sender]
    qty_map = {
        "shona": f"Zvakanaka! Mungada mangani e *{product_name}*?",
        "ndebele": f"Kulungile! Ufuna izinga elingakanani le *{product_name}*?",
        "chinyanja": f"Chabwino! Mukufuna kuchulukitsa *{product_name}* kangati?",
        "tonga": f"Chabwino! Mukufuna kuchulukitsa *{product_name}* kangati?",
        "bemba": f"Cino cino! Ufuna ukugula *{product_name}* ingi buti?",
        "lozi": f"Ho lokile! U bata ku landa *{product_name}* hañata mañi?",
    }
    send(qty_map.get(lang, f"Great! How many *{product_name}* would you like?"), sender, phone_id)
    state["step"] = "shop_quantity"
    save_single_user_state(sender)


def handle_shop_quantity(sender, prompt, phone_id):
    """Capture quantity and then ask for address."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.strip()

    # Try to extract a number
    qty_match = re.search(r"\d+", prompt_lower)
    if qty_match:
        qty = int(qty_match.group())
        state["shop_quantity"] = qty
        addr_map = {
            "shona": "Ndatenda! Ndapota tipa kero yako yekuendesa (guta, nharaunda, uye chero mamwe mashoko akakurudzira kubatsira).",
            "ndebele": "Ngiyabonga! Ngicela unike ikheli lakho lokuhambisa (idolobha, indawo, noma yiluphi ulwazi olwengeziwe olulusizo).",
            "chinyanja": "Zikomo! Chonde tipatseni adilesi yanu yokumanga (mzinda, dera, ndi chilichonse china chopindulitsa).",
            "tonga": "Twatotela! Ndatola, tipeni adilesi yanu yokumanga (mzinda, dera, ndi chilichonse china).",
            "bemba": "Natotela! Napapata, mpeele aderesi yenu ya kupeleka (tawuni, cifungo, kabili fimo ifyalumo).",
            "lozi": "Ndalumba! Ndapota, nipe aderesi ya hao ya ku alafa (tauni, sibaka, ni ze ñwi ze thusang).",
        }
        send(addr_map.get(lang, "Thank you! Please provide your delivery address (town, area, and any additional helpful details)."), sender, phone_id)
        state["step"] = "shop_address"
        save_single_user_state(sender)
    else:
        invalid_qty_map = {
            "shona": "Ndapota pinda nhamba (semuenzaniso: 1, 2, 3).",
            "ndebele": "Ngicela ufake inombolo (isibonelo: 1, 2, 3).",
            "chinyanja": "Chonde lowetsani nambala (mwachitsanzo: 1, 2, 3).",
            "tonga": "Ndatola, ingila nambala (semuenzaniso: 1, 2, 3).",
            "bemba": "Napapata, ingisha inomba (semuenzaniso: 1, 2, 3).",
            "lozi": "Ndapota, kenya nomolo (semuenzaniso: 1, 2, 3).",
        }
        send(invalid_qty_map.get(lang, "Please enter a number (e.g. 1, 2, 3)."), sender, phone_id)


def _save_orders_to_redis(sender, cart, address):
    """Persist all cart items as individual order records in Redis."""
    if not redis_client:
        return
    user_id = user_states[sender].get("user_id", sender)
    for item in cart:
        order = {
            "user_id": user_id,
            "sender": sender,
            "product": item["product"],
            "price": item["price"],
            "quantity": item["quantity"],
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }
        try:
            order_key = f"orders:{sender}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            redis_client.set(order_key, json.dumps(order))
            logging.info(f"Order saved: {order_key} -> {order}")
        except Exception as e:
            logging.error(f"Error saving order: {e}")


def _send_order_confirmation(sender, phone_id, lang, cart, address):
    """Send a full order summary and farewell."""
    def build_lines(header, addr_label, closing):
        parts = [header, ""]
        for item in cart:
            parts.append(f"  📦 {item['product']} x{item['quantity']} — {item['price']}")
        parts.append("")
        parts.append(f"📍 {addr_label}: {address}")
        parts.append(closing)
        return "\n".join(parts)

    msg_map = {
        "shona":     build_lines("✅ *Odha Yakugamuchirwa!*",    "Kero",    "Tichakubata munguva pfupi. Ndatenda! 😊"),
        "ndebele":   build_lines("✅ *Ioda Ikugunyazwe!*",       "Ikheli",  "Sizokuthinta masinyane. Ngiyabonga! 😊"),
        "chinyanja": build_lines("✅ *Dongosolo Lasinthidwa!*",  "Adilesi", "Tidzakuumbanani posachedwapa. Zikomo! 😊"),
        "tonga":     build_lines("✅ *Dongosolo Lasinthidwa!*",  "Adilesi", "Tinkuumba posachedwapa. Twatotela! 😊"),
        "bemba":     build_lines("✅ *Icigula Cachitwa!*",       "Aderesi", "Tukakuumba mu kufupifupi. Natotela! 😊"),
        "lozi":      build_lines("✅ *Landa Le Li Amuhezwi!*",   "Aderesi", "Lu ta ku ama ka nako ye nyinyani. Ndalumba! 😊"),
    }
    default = build_lines("✅ *Order Confirmed!*", "Delivery address", "We'll be in touch shortly. Thank you! 😊")
    send(msg_map.get(lang, default), sender, phone_id)




def handle_shop_address(sender, prompt, phone_id):
    """Capture delivery address, add current item to cart, ask if user wants anything else."""
    state = user_states[sender]
    lang = state["language"]
    address = prompt.strip()

    state["shop_address"] = address

    # Append current item to cart
    cart = state.setdefault("cart", [])
    cart.append({
        "product": state.get("shop_selected_product", "Unknown"),
        "price":   state.get("shop_selected_price", "N/A"),
        "quantity": state.get("shop_quantity", 1),
    })

    more_map = {
        "shona":     "✅ Zvakanaka! Ndakuwanira. Ungada here kuwedzera chimwe chigadzirwa kuodha yako?",
        "ndebele":   "✅ Kulungile! Ngikuqoqele. Ungathanda ukwengeza umkhiqizo ku-odha yakho?",
        "chinyanja": "✅ Chabwino! Ndakujambulani. Kodi mukufuna kuwonjezera chinthu china ku dongosolo lanu?",
        "tonga":     "✅ Chabwino! Ndakujambulani. Ungafuna kuwonjezera chinthu china ku dongosolo lako?",
        "bemba":     "✅ Cino cino! Nalilembele. Ufuna ukuwongesha imisansa ina ku icigula cako?",
        "lozi":      "✅ Ho lokile! Na ku ñolela. Kana u bata ku yema swakupila si liñwi ku landa la hao?",
    }
    send(more_map.get(lang, "✅ Got it! Would you like to add anything else to your order?"), sender, phone_id)
    state["step"] = "shop_add_more"
    save_single_user_state(sender)


def handle_shop_add_more(sender, prompt, phone_id):
    """Handle 'anything else?' — either add more items or finalise the order."""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    # Check if they directly named a product
    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next(
        (p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()),
        None
    )
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"]   = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
        return

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        cart    = state.get("cart", [])
        address = state.get("shop_address", "")
        _save_orders_to_redis(sender, cart, address)
        _send_order_confirmation(sender, phone_id, lang, cart, address)
        reset_conversation(sender)
    else:
        clarify_map = {
            "shona":     "Ungada here kuwedzera chimwe? Pindura 'hongu' kuona zvigadzirwa, kana 'kwete' kugadzirisa odha yako.",
            "ndebele":   "Ungathanda ukwengeza okunye? Phendula 'yebo' ukuze ubone imikhiqizo, noma 'cha' ukuqeda i-odha.",
            "chinyanja": "Kodi mukufuna kuwonjezera china? Yankha 'inde' kuona zinthu, kapena 'ayi' kumaliza dongosolo.",
            "tonga":     "Ungafuna kuwonjezera china? Yankha 'inde' kuona zinthu, kapena 'ayi' kumaliza dongosolo.",
            "bemba":     "Ufuna ukuwongesha fimo? Yasuka 'inde' ukubona imisansa, noma 'ayi' ukumalisha icigula.",
            "lozi":      "Kana u bata ku yema se si liñwi? Arabela 'inde' ku bona swakupila, kamba 'ayi' ku feza landa.",
        }
        send(clarify_map.get(lang, "Would you like to add anything else? Reply 'yes' to browse or 'no' to finalise."), sender, phone_id)



def handle_purchase_response(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    
    if any(response in prompt_lower for response in no_responses):
        bye_map = {
            "shona": "Ndatenda! Iva nezuva rakanaka. Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.",
            "ndebele": "Ngiyabonga! Ube nosuku oluhle. Uma uneminye imibuzo, qala ingxoxo entsha ngo-'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.",
            "tonga": "Twatotela! Mube abusiku bwabwino. Ngaa muli ne mafunso yami, yambani ponena 'mwabuka buti'.",
            "bemba": "Natotela! Mubelele bwino. Nga muli na ifyafyala ifyalumo, tambuleni ponena 'mwaiseni'.",
            "lozi": "Ndalumba! Mube ni lizazi le linde. Ha mu na lipuzo le linwi, qalisa ka ku bulela 'mwa bona'.",
        }
        send(bye_map.get(lang, "Thank you! Have a nice day. If you have more questions, start over by saying 'hi'."), sender, phone_id)
        reset_conversation(sender)
        return
        
    elif any(response in prompt_lower for response in yes_responses):
        topic = state.get("topic")
        
        if topic == "maternal":
            maternal_products = extract_products_by_category("Maternal Health")
            if maternal_products:
                products_text = format_products_for_display(maternal_products, lang)
                send(products_text, sender, phone_id)
            else:
                no_prod_map = {
                    "shona": "Ndine urombo, hapana zvigadzirwa zvehutano hwepamuviri zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.",
                    "ndebele": "Uxolo, azikho izinto zokunakekela isisu ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.",
                    "tonga": "Ndatola, tana zinthu za bupilo bwa kubaa mwana zitholakalaya seesilizyo. Tinkusinja kufuna kuswiilila bukliniki bweesu.",
                    "bemba": "Natapa, tapali imisansa ya buumi bwa nkashi itholakalayo seesilizyo. Tikusinja ukuya ku kliniki yeesu.",
                    "lozi": "Ni maswabi, ha ku na swakupila swa buimana se si fumaneha cwale. Lu ku susueza ku ya kwa kiliniki ya luna.",
                }
                send(no_prod_map.get(lang, "Sorry, no maternal health products are currently available. We recommend visiting our clinic for more information."), sender, phone_id)
                
        elif topic == "cervical":
            cervical_products = extract_products_by_category("Cervical Cancer")
            if cervical_products:
                products_text = format_products_for_display(cervical_products, lang)
                send(products_text, sender, phone_id)
            else:
                no_cerv_map = {
                    "shona": "Ndine urombo, hapana zvigadzirwa zvecervical cancer zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.",
                    "ndebele": "Uxolo, azikho izinto zokuvikela isilonda somlomo wesibeletho ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.",
                    "tonga": "Ndatola, tana zinthu za kansa ya mulomo wa cibeleko zitholakalaya seesilizyo. Tinkusinja kufuna kuswiilila bukliniki bweesu.",
                    "bemba": "Natapa, tapali imisansa ya kansa ya cibeleshi itholakalayo seesilizyo. Tikusinja ukuya ku kliniki yeesu.",
                    "lozi": "Ni maswabi, ha ku na swakupila swa kankere ya mulomo wa sibeleko se si fumaneha cwale. Lu ku susueza ku ya kwa kiliniki ya luna.",
                }
                send(no_cerv_map.get(lang, "Sorry, no cervical cancer products are currently available. We recommend visiting our clinic for more information."), sender, phone_id)
        else:
            general_products = extract_products_by_category("General")
            if general_products:
                products_text = format_products_for_display(general_products, lang)
                send(products_text, sender, phone_id)
            else:
                gen_map = {
                    "shona": "Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo.",
                    "ndebele": "Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane.",
                    "tonga": "Twatotela! Tinkuumba mukati wa maminiti mashoma kuti mupate zambiri.",
                    "bemba": "Natotela! Tukakuumba mukati wa maminiti ashono ukufuna ifyalumo fyalumo.",
                    "lozi": "Ndalumba! Lu ta ku ama ka nako ye nyinyani kuli lu file litaba ze ñwi.",
                }
                send(gen_map.get(lang, "Thank you! We'll contact you shortly for more details."), sender, phone_id)
        
        proceed_map = {
            "shona": "Ungada here kuenderera mberi nekutenga chimwe chezvigadzirwa izvi? ",
            "ndebele": "Ungathanda ukuqhubeka nokuthenga noma yini yale mikhiqizo? ",
            "chinyanja": "Kodi mukufuna kupitiriza kugula chinthu cha zinthu izi? ",
            "tonga": "Ungafuna kupitiriza kugula chintu cha zinthu izi? ",
            "bemba": "Ufuna ukupitiliza ukugula chintu cha imisansa iyi? ",
            "lozi": "Kana u bata ku zwelapili ku landa se si liñwi sa swakupila se? ",
        }
        send(proceed_map.get(lang, "Would you like to proceed with purchasing any of these products? "), sender, phone_id)
        
        state["step"] = "confirm_purchase"
        save_single_user_state(sender)
        
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kutenga zvigadzirwa? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuthenga imikhiqizo? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kugula zinthu? ",
            "tonga": "Tana kumva. Yankhani chonde: Ungafuna kugula zinthu? ",
            "bemba": "Nsasumina. Yasuka chonde: Ufuna ukugula imisansa? ",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku landa swakupila? ",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to purchase products?  "), sender, phone_id)


def handle_purchase_confirmation(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    
    if any(response in prompt_lower for response in no_responses):
        alright_map = {
            "shona": "Zvakanaka. Tinokutendai! Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.",
            "ndebele": "Kulungile. Ngiyabonga! Uma uneminye imibuzo, qala kabusha ngokuthi 'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.",
            "tonga": "Chabwino. Twatotela! Ngaa muli ne mafunso yami, yambani ponena 'mwabuka buti'.",
            "bemba": "Cino cino. Natotela! Nga muli na ifyafyala ifyalumo, tambuleni ponena 'mwaiseni'.",
            "lozi": "Ho lokile. Ndalumba! Ha mu na lipuzo le linwi, qalisa ka ku bulela 'mwa bona'.",
        }
        send(alright_map.get(lang, "Alright. Thank you! If you have more questions, start over by saying 'hi'."), sender, phone_id)
        reset_conversation(sender)
        
    elif any(response in prompt_lower for response in yes_responses):
        thanks_map = {
            "shona": "Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo nezvekutenga.",
            "ndebele": "Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane ngokuthenga.",
            "chinyanja": "Zikomo! Tidzakuumbanani posachedwapa kuti mupeze zambiri zokhudza kugula.",
            "tonga": "Twatotela! Tinkuumba posachedwapa kuti mupate zambiri zokhudza kugula.",
            "bemba": "Natotela! Tukakuumba mu kufupifupi ukufuna ifyalumo pa kugula.",
            "lozi": "Ndalumba! Lu ta ku ama ka nako ye nyinyani kuli lu fe litaba ze ñwi ka za ku landa.",
        }
        send(thanks_map.get(lang, "Thank you! We'll contact you shortly for more details about your purchase."), sender, phone_id)
        reset_conversation(sender)
        
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuenderera mberi nekutenga? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokuthenga? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupitiriza kugula? ",
            "tonga": "Tana kumva. Yankhani chonde: Ungafuna kupitiriza kugula? ",
            "bemba": "Nsasumina. Yasuka chonde: Ufuna ukupitiliza ukugula? ",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku zwelapili ku landa? ",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to proceed with purchasing?  "), sender, phone_id)


def extract_products_by_category(category_name):
    try:
        return products_by_category.get(category_name, [])
    except Exception as e:
        logging.error(f"Error extracting products for category {category_name}: {e}")
        return []

def format_products_for_display(products_list, lang):
    if not products_list:
        empty_map = {
            "shona": "Hapana zvigadzirwa zvazvino onekwa.",
            "ndebele": "Azikho imikhiqizo etholakalayo okwamanje.",
            "chinyanja": "Palibe zinthu zitholakalayo pakali pano.",
            "tonga": "Tana zinthu zitholakalaya seesilizyo.",
            "bemba": "Tapali imisansa itholakalayo seesilizyo.",
            "lozi": "Ha ku na swakupila se si fumaneha cwale.",
        }
        return empty_map.get(lang, "No products currently available.")
    
    header_map = {
        "shona": "🏥 Zvigadzirwa Zvehutano:\n\n",
        "ndebele": "🏥 Imikhiqizo Yezempilo:\n\n",
        "chinyanja": "🏥 Zinthu za Thanzo:\n\n",
        "tonga": "🏥 Zinthu za Bupilo:\n\n",
        "bemba": "🏥 Imisansa ya Buumi:\n\n",
        "lozi": "🏥 Swakupila:\n\n",
    }
    products_text = header_map.get(lang, "🏥 Health Products:\n\n")
    
    for i, product in enumerate(products_list, 1):
        name = product.get('name', 'Unknown Product')
        price = product.get('price', 'Price not available')
        availability = product.get('availability', 'Availability not specified')
        
        if lang == "shona":
            products_text += f"{i}. {name}\n   💰 Mutengo: {price}\n   📦 Kuwanikwa: {availability}\n\n"
        elif lang == "ndebele":
            products_text += f"{i}. {name}\n   💰 Inani: {price}\n   📦 Ukutholakala: {availability}\n\n"
        elif lang == "chinyanja":
            products_text += f"{i}. {name}\n   💰 Mtengo: {price}\n   📦 Kupezeka: {availability}\n\n"
        elif lang == "tonga":
            products_text += f"{i}. {name}\n   💰 Mtengo: {price}\n   📦 Kutholakalaya: {availability}\n\n"
        elif lang == "bemba":
            products_text += f"{i}. {name}\n   💰 Intengo: {price}\n   📦 Ukutholakalaya: {availability}\n\n"
        elif lang == "lozi":
            products_text += f"{i}. {name}\n   💰 Teko: {price}\n   📦 Ku fumanehanga: {availability}\n\n"
        else:
            products_text += f"{i}. {name}\n   💰 Price: {price}\n   📦 Availability: {availability}\n\n"
    
    select_map = {
        "shona": "Sarudza chirongwa nekuudza nhamba yacho.",
        "ndebele": "Khetha umkhiqizo ngokutshela inombolo yayo.",
        "chinyanja": "Sankhani chinthu ponena nambala yake.",
        "tonga": "Sankhani chinthu ponena nambala yake.",
        "bemba": "Sala imisansa ka kulanda inomba yayo.",
        "lozi": "U khethe swakupila ka ku bulela nomolo ya sona.",
    }
    products_text += select_map.get(lang, "Select a product by telling us the number.")
    
    return products_text

def handle_conversation_state(sender, prompt, phone_id):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()

    # ── UNIVERSAL greeting + reset intercept ──────────────────────────────────
    # This fires BEFORE any step routing so a greeting at ANY step always resets
    # cleanly — prevents stale topic/step from contaminating the new conversation.
    ALL_GREETINGS = [
        "hi", "hello", "hey", "hie", "hi there",
        "good morning", "good afternoon", "good evening",
        # Shona
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        # Ndebele
        "sawubona", "salibonani",
        # Chinyanja
        "moni", "muli bwanji",
        # Tonga
        "mwabuka", "mwabuka buti", "mwalandwa", "mwalandwa buti",
        # Bemba
        "mwaiseni", "muli shani",
        # Lozi
        "mwa bona",
    ]
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "reset", "help"]

    is_greeting = any(
        prompt_lower == w or re.search(rf"\b{re.escape(w)}\b", prompt_lower)
        for w in ALL_GREETINGS
    )
    is_reset = any(kw in prompt_lower for kw in reset_keywords)

    # Only skip during language_detection and registration
    current_step_pre = state.get("step")
    if (is_greeting or is_reset) and current_step_pre not in ["language_detection", "registration"]:
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "tonga": "Moni! Ndingamwafwa shani ilelo?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        return
    # ─────────────────────────────────────────────────────────────────────────

    current_step = state.get("step")
    
    if current_step == "language_detection" and state.get("first_message", True):
        handle_language_detection(sender, prompt, phone_id)
    elif current_step == "registration":
        handle_registration(sender, prompt, phone_id)
    elif current_step in ["ask_another_week", "cervical_more_info", "cervical_question_number", "keep_learning", "follow_up"]:
        if current_step == "ask_another_week":
            handle_another_week(sender, prompt, phone_id)
        elif current_step == "cervical_more_info":
            handle_cervical_more_info(sender, prompt, phone_id)
        elif current_step == "cervical_question_number":
            handle_cervical_question_number(sender, prompt, phone_id)
        elif current_step == "keep_learning":
            handle_keep_learning(sender, prompt, phone_id)
        elif current_step == "follow_up":
            handle_follow_up(sender, prompt, phone_id)
    elif current_step == "product_inquiry":
        handle_purchase_response(sender, prompt, phone_id)
    elif current_step == "confirm_purchase":
        handle_purchase_confirmation(sender, prompt, phone_id)
    elif current_step == "shop_interest":
        handle_shop_interest(sender, prompt, phone_id)
    elif current_step == "shop_browse":
        handle_shop_browse(sender, prompt, phone_id)
    elif current_step == "shop_order_decision":
        handle_shop_order_decision(sender, prompt, phone_id)
    elif current_step == "shop_product_name":
        handle_shop_product_name(sender, prompt, phone_id)
    elif current_step == "shop_more_categories":
        handle_shop_more_categories(sender, prompt, phone_id)
    elif current_step == "shop_add_more":
        handle_shop_add_more(sender, prompt, phone_id)
    elif current_step == "shop_quantity":
        handle_shop_quantity(sender, prompt, phone_id)
    elif current_step == "shop_address":
        handle_shop_address(sender, prompt, phone_id)
    elif current_step == "general_followup":
        handle_general_followup(sender, prompt, phone_id)
        return
    elif current_step == "general_question":
        lang = state["language"]
        reply = ask_gemini_general(prompt, lang)
        send(reply, sender, phone_id)
        _send_more_questions(sender, phone_id, lang)
        state["step"] = "general_followup"
        save_single_user_state(sender)
        return
    else:
        handle_main_menu(sender, prompt, phone_id)
        

def ask_cervical_more_info(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    more_map = {
        "shona": "Ungada here kuwana rumwe ruzivo rwe cervical cancer? ",
        "ndebele": "Ungathanda ukuthola eminye imininingwane nge-cervical cancer? ",
        "chinyanja": "Kodi mukufuna kupeza zambiri za cervical cancer?",
        "tonga": "Ungafuna kupata zambiri za kansa ya mulomo wa cibeleko?",
        "bemba": "Ufuna ukupata ifyalumo fyalumo pa kansa ya cibeleshi?",
        "lozi": "Kana u bata ku fumana litaba ze ñwi ka za kankere ya mulomo wa sibeleko?",
    }
    send(more_map.get(lang, "Would you like to get more information about cervical cancer?  "), sender, phone_id)
    
    state["step"] = "cervical_more_info"
    save_single_user_state(sender)

def ask_cervical_question_number(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    num_map = {
        "shona": "Pinda nhamba yemubvunzo kubva pa 1 kusvika pa 100:",
        "ndebele": "Faka inombolo yombuzo kusuka ku-1 kuya ku-100:",
        "chinyanja": "Lowetsani nambala ya funso kuchokera pa 1 mpaka 100:",
        "tonga": "Ingila nambala ya mwaambo kuzwa 1 kusika 100:",
        "bemba": "Ingisha inomba ya ilipusho ukufuma pa 1 ukufika pa 100:",
        "lozi": "Kenya nomolo ya lipuzo ku zwana 1 ku ya ku 100:",
    }
    send(num_map.get(lang, "Enter a question number from 1 to 100:"), sender, phone_id)
    
    state["step"] = "cervical_question_number"
    save_single_user_state(sender)

def ask_keep_learning(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    keep_map = {
        "shona": "Ungada here kuramba uchidzidza zvimwe zvinhu zve cervical cancer? ",
        "ndebele": "Ungathanda ukuqhubeka nokufunda ezinye izindaba ze-cervical cancer? ",
        "chinyanja": "Kodi mukufuna kupitiriza kuphunzira zina zambiri za cervical cancer?",
        "tonga": "Ungafuna kupitiriza kuphunzira zambiri za kansa ya mulomo wa cibeleko?",
        "bemba": "Ufuna ukupitiliza ukulearela ifyalumo pa kansa ya cibeleshi?",
        "lozi": "Kana u bata ku zwelapili ku ithuta litaba ze ñwi ka za kankere ya mulomo wa sibeleko?",
    }
    send(keep_map.get(lang, "Would you like to keep learning more about cervical cancer?  "), sender, phone_id)
    
    state["step"] = "keep_learning"
    save_single_user_state(sender)

def handle_cervical_more_info(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi"]
    
    if any(response in prompt_lower for response in yes_responses):
        ask_cervical_question_number(sender, phone_id)
    elif any(response in prompt_lower for response in no_responses):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuwana rumwe ruzivo? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuthola eminye imininingwane? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupeza zambiri?",
            "tonga": "Tana kumva. Yankhani chonde: Ungafuna kupata zambiri?",
            "bemba": "Nsasumina. Yasuka chonde: Ufuna ukupata ifyalumo fyalumo?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku fumana litaba ze ñwi?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to get more information?  "), sender, phone_id)

def handle_cervical_question_number(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    try:
        question_num = int(re.sub(r"\D", "", prompt))
        if 1 <= question_num <= 100:
            data_tuple = get_cervical_data(lang)
            
            question_found = False
            for i, item in enumerate(data_tuple):
                if f"*Question {question_num}:" in str(item):
                    question_content = str(item)
                    
                    if i + 1 < len(data_tuple) and "Answer" in str(data_tuple[i + 1]):
                        question_content += "\n" + str(data_tuple[i + 1])
                    
                    send(question_content, sender, phone_id)
                    question_found = True
                    ask_keep_learning(sender, phone_id)
                    break
            
            if not question_found:
                not_found_map = {
                    "shona": f"Ndine urombo, handina kuwana mubvunzo wenhamba {question_num}. Edza imwe nhamba kubva pa 1 kusvika pa 100.",
                    "ndebele": f"Uxolo, angikutholanga umbuzo wenombolo {question_num}. Zama enye inombolo kusuka ku-1 kuya ku-100.",
                    "chinyanja": f"Pepani, sindinapeze funso la nambala {question_num}. Yesani nambala ina kuchokera pa 1 mpaka 100.",
                    "tonga": f"Ndatola, tana kupata mwaambo wa nambala {question_num}. Lingenya nambala inzwi kuzwa 1 kusika 100.",
                    "bemba": f"Natapa, nshasangile ilipusho lya inomba {question_num}. Esheni inomba inzwi ukufuma pa 1 ukufika pa 100.",
                    "lozi": f"Ni maswabi, ha ni fumani lipuzo la nomolo {question_num}. Linge nomolo ye nzwi ku zwana 1 ku ya ku 100.",
                }
                send(not_found_map.get(lang, f"Sorry, I couldn't find question number {question_num}. Please try another number from 1 to 100."), sender, phone_id)
                ask_cervical_question_number(sender, phone_id)
        else:
            range_map = {
                "shona": "Ndapota pinda nhamba kubva pa 1 kusvika pa 100 chete.",
                "ndebele": "Sicela ufake inombolo ephakathi kuka-1 no-100 kuphela.",
                "chinyanja": "Chonde lowetsani nambala kuchokera pa 1 mpaka 100 basi.",
                "tonga": "Ndatola, ingila nambala kuzwa 1 kusika 100 fye.",
                "bemba": "Napapita, ingisha inomba ukufuma pa 1 ukufika pa 100 fye.",
                "lozi": "Ndapota, kenya nomolo ku zwana 1 ku ya ku 100 feela.",
            }
            send(range_map.get(lang, "Please enter a number between 1 and 100 only."), sender, phone_id)
            ask_cervical_question_number(sender, phone_id)
            
    except ValueError:
        invalid_map = {
            "shona": "Ndapota pinda nhamba chaiyo kubva pa 1 kusvika pa 100.",
            "ndebele": "Sicela ufake inombolo evumelekile ephakathi kuka-1 no-100.",
            "chinyanja": "Chonde lowetsani nambala yoyenera kuchokera pa 1 mpaka 100.",
            "tonga": "Ndatola, ingila nambala yoyenera kuzwa 1 kusika 100.",
            "bemba": "Napapita, ingisha inomba iyenera ukufuma pa 1 ukufika pa 100.",
            "lozi": "Ndapota, kenya nomolo ye nepahezi ku zwana 1 ku ya ku 100.",
        }
        send(invalid_map.get(lang, "Please enter a valid number between 1 and 100."), sender, phone_id)
        ask_cervical_question_number(sender, phone_id)

def handle_keep_learning(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi"]
    
    if any(response in prompt_lower for response in yes_responses):
        ask_cervical_question_number(sender, phone_id)
    elif any(response in prompt_lower for response in no_responses):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuramba uchidzidza? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokufunda? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupitiriza kuphunzira?",
            "tonga": "Tana kumva. Yankhani chonde: Ungafuna kupitiriza kuphunzira?",
            "bemba": "Nsasumina. Yasuka chonde: Ufuna ukupitiliza ukulearela?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku zwelapili ku ithuta?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to keep learning?  "), sender, phone_id)

def ask_another_week(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    another_map = {
        "shona": "Ungada here kudzidza nezve mamwe mavhiki epamuviri? ",
        "ndebele": "Ungathanda ukufunda ngamanye amaviki okukhulelwa? ",
        "chinyanja": "Kodi mukufuna kudziwa za masabata ena a pakati?",
        "tonga": "Ungafuna kudziwa za manhwiiiki eni a kubaa mwana?",
        "bemba": "Ufuna ukuishiba pa myeshi iyengi ya pa nkundi?",
        "lozi": "Kana u bata ku ithuta ka za maviki a manwi a buimana?",
    }
    send(another_map.get(lang, "Would you like to learn about other pregnancy weeks?  "), sender, phone_id)
    
    state["step"] = "ask_another_week"
    save_single_user_state(sender)


def handle_another_week(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi"]
    
    if any(response in prompt_lower for response in yes_responses):
        state["step"] = "ask_week"
        week_map = {
            "shona": "Ndapota isa vhiki re pamuviri ",
            "ndebele": "Sicela ufake iviki lokukhulelwa ",
            "chinyanja": "Chonde lowetsani sabata la pakati ",
            "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
            "tonga": "Ndatola, ingila nhwiiiki ya kubeleka mwana ",
            "bemba": "Napapita, ingisha umulungu wa pa nkundi ",
        }
        send(week_map.get(lang, "Please enter your pregnancy week number "), sender, phone_id)
        save_single_user_state(sender)
        
    elif any(response in prompt_lower for response in no_responses):
        state["step"] = "product_inquiry"
        state["topic"] = "maternal"
        
        prod_offer_map = {
            "shona": "Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri? Tinopa:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits",
            "ndebele": "Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu? Sinakho:\n- Ama-Prenatal Vitamins\n- Izinto zokuhlola isisu\n- Amakhithi okunakekela isisu",
            "chinyanja": "Zikomo! Kodi mukufuna kugula zinthu za Thanzi la Amayi? Tili ndi:\n- Mavitamini a Prenatal\n- Zoyezera pakati\n- Makiti a Thanzi la Amayi",
            "tonga": "Twatotela! Ungafuna kugula zinthu za bupilo bwa kubaa mwana? Tili na:\n- Mavitamini a Prenatal\n- Zoyezera kubaa mwana\n- Makiti a Bupilo bwa Kubaa Mwana",
            "bemba": "Natotela! Ufuna ukugula imisansa ya buumi bwa nkashi? Tuli na:\n- Mavitamini a Prenatal\n- Ifyoyeshamo pa nkundi\n- Makiti ya Buumi bwa Nkashi",
            "lozi": "Ndalumba! Kana u bata ku landa swakupila swa buimana? Lu na:\n- Mavitamini a Prenatal\n- Swakutatuba buimana\n- Makiti a Buimana",
        }
        send(prod_offer_map.get(lang, "Thank you! Would you like to purchase maternal health products? We offer:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits"), sender, phone_id)
        save_single_user_state(sender)
        
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kudzidza nezve mamwe mavhiki? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukufunda ngamanye amaviki? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kudziwa za masabata ena?",
            "tonga": "Tana kumva. Yankhani chonde: Ungafuna kudziwa za manhwiiiki eni?",
            "bemba": "Nsasumina. Yasuka chonde: Ufuna ukuishiba pa myeshi iyengi?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku ithuta ka za maviki a manwi?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to learn about other weeks?  "), sender, phone_id)


# ─────────────────────────────────────────────
#  FIX: All Gemini functions now use stateless
#  per-call generation — NO shared global convo
# ─────────────────────────────────────────────

def _get_lang_enforce(lang: str) -> str:
    return {
        "shona":     "Pindura muchiShona chete. Usashandise Chirungu.",
        "ndebele":   "Phendula ngesiNdebele kuphela. Ungasebenzisi isiNgisi.",
        "chinyanja": "Yankhani mu Chichewa/Chinyanja basi. Osagwiritsa ntchito Chingerezi.",
        "lozi":      "Arabela ka Silozi feela. U se ke wa sebelisa Siingelesi.",
        "bemba":     "Yasuka mu Chibemba fye. Ushatumishe Cingeleshi.",
        "tonga":     "Mupandule mu Chitonga chete. Musagwisye Ciingelezi.",
    }.get(lang, "Respond in English only.")


def _get_fallback(lang: str) -> str:
    return {
        "shona":     "Pane dambudziko pakupindura mubvunzo wako.",
        "ndebele":   "Kunenkinga ekuphenduleni umbuzo wakho.",
        "chinyanja": "Pali vuto popanga yankho la funso lanu.",
        "tonga":     "Kwakali zyuuno mu kupandula mwaambo wako.",
        "bemba":     "Kuli ubukopo mu kuyasuka ilipusho lyobe.",
        "lozi":      "Ku na bothata ka ku arabela lipuzo la hao.",
    }.get(lang, "Sorry, there was a problem getting an answer.")


def ask_gemini(question: str, lang: str = "english") -> str:
    """
    Maternal health Gemini call.
    FIX: uses stateless generate_content (no shared chat history).
    FIX: language enforce instruction placed at END of prompt.
    """
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano hwepamuviri. "
            "Pindura mubvunzo uyu muShona yakajeka, yakapfava, uye ine ruzivo rwezvehutano:\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo yesisu. "
            "Phendula lo mbuzo ngesiNdebele esicacile, esilula, futhi enolwazi lwezempilo:\n\n"
        ),
        "chinyanja": (
            "Ndine mphungu wa Thanzi la Amayi. "
            "Yankhani funso ili m'Chinyanja moyenera, mosavuta, komanso moli ndi umanyambazi wa Thanzi la Amayi:\n\n"
        ),
        "lozi": (
            "Ki muthusi wa za mapilo wa buimana. "
            "Alaba lipuzo le ka Silozi se si nepahezi, se si nolofetse, ni se si na ni bupilo:\n\n"
        ),
        "bemba": (
            "Ndi kapyunga wa buumi bwa bana. "
            "Yasuka ilipusho lyi mu Chibemba icasalangana, icapepa, na icali na ubusuma bwa buumi:\n\n"
        ),
        "tonga": (
            "Ndi mweenzinyina wa buumi bwa kubaa mwana. "
            "Mupandule mwaambo wu mu Chitonga chakweelela, chiswiipe, komanso chili a cibelesyo ca buumi:\n\n"
        ),
    }.get(lang, (
        "You are a maternal health assistant. "
        "Answer the following question clearly, simply, and with accurate health information:\n\n"
    ))

    # FIX: language enforcement moved to END of prompt
    prompt = f"{instruction_body}{question}\n\n{lang_enforce}"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt)
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini] Blocked/empty lang={lang}: {ve}")
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini Error] {type(e).__name__}: {e}")
        return fallback


def ask_gemini_cancer(question: str, lang: str = "english") -> str:
    """
    Cervical cancer Gemini call.
    FIX: uses stateless generate_content (no shared chat history).
    FIX: language enforce instruction placed at END of prompt.
    """
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano hwegomarara rechibereko. "
            "Pindura mubvunzo uyu muShona yakajeka uye yakapfava:\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo yomhlaza wesibeletho. "
            "Phendula lo mbuzo ngesiNdebele esicacile futhi esilula:\n\n"
        ),
        "chinyanja": (
            "Ndine mphungu wa thanzi la kansa ya chibereko. "
            "Yankhani funso ili mu Chinyanja momveka bwino komanso mwaulemu:\n\n"
        ),
        "lozi": (
            "Ki muthusi wa za kansa ya mulomo wa sibeleko. "
            "Alaba lipuzo le ka Silozi se si bonahala hande ni se si nolofetse:\n\n"
        ),
        "bemba": (
            "Ndi kapyunga wa kansa ya cibeleshi. "
            "Yasuka ilipusho lyi mu Chibemba icamoneka bwino kabili icapepuka:\n\n"
        ),
        "tonga": (
            "Ndi mweenzinyina wa kansa ya mulomo wa cibeleko. "
            "Mupandule mwaambo wu mu Chitonga chakweelela alimwi chiswiipe:\n\n"
        ),
    }.get(lang, (
        "You are a cervical cancer health assistant. "
        "Answer the following question clearly and simply in English:\n\n"
    ))

    # FIX: language enforcement moved to END of prompt
    prompt = f"{instruction_body}{question}\n\n{lang_enforce}"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt)
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini_cancer] Blocked/empty lang={lang}: {ve}")
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini_cancer Error] {type(e).__name__}: {e}")
        return fallback


def ask_gemini_general(question: str, lang: str) -> str:
    """
    General Gemini call.
    FIX: uses stateless generate_content (no shared chat history).
    FIX: language enforce instruction placed at END of prompt.
    """
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)

    company_address = "No. 50 Lunsemfwa Rd, Kalundu, Lusaka, Zambia"
    company_email   = "hello@dawa-health.com"
    company_website = "https://dawa-health.com/"
    company_phone   = "+260 571 376 677"

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano ane hunyanzvi muhutano hwevakadzi vane pamuviri uye gomarara remuromo wechibereko. "
            "Pindura mubvunzo wemushandisi uchishandisa ruzivo rwechokwadi. "
            "USATANGE nemitsara yakaita sekuti Zvakanaka, Hongu, Hezvino. "
            "Tanga zvakananga nemhinduro. "
            "Pedzisa nekuyambira kupfupi kunoti ruzivo urwu harutsivi kuongororwa nachiremba.\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo ochwepheshile ogxile kwezempilo yabomama abakhulelweyo kanye lomdlavuza womlomo wesibeletho. "
            "Phendula umbuzo womsebenzisi usebenzisa ulwazi lwezempilo oluqondileyo. "
            "UNGAKALI ngemisho efana lokuthi Kulungile, Yebo, Nakhu. "
            "Qalisa masinyane ngempendulo uqobo. "
            "Qedisa ngesexwayiso esifitshane esithi ulwazi lolu aluthathi indawo yokuhlolwa ngudokotela.\n\n"
        ),
        "chinyanja": (
            "Ndinu mthandizi wa zaumoyo wa akatswiri pa zaumoyo wa amayi apakati komanso khansa ya chiberekero. "
            "Yankhani funso la wogwiritsa ntchito pogwiritsa ntchito chidziwitso cholondola. "
            "MUSAYAMBE ndi mawu ngati Chabwino, Inde, Nazi. "
            "Yambani mwachindunji ndi yankho. "
            "Malizitsani ndi chenjezo chachidule chonena kuti chidziwitsochi sichimalowa m'malo mwa kuyezetsa kwa dokotala.\n\n"
        ),
        "tonga": (
            "Muli mweenzinyina wa buumi mu buumi bwa banakazi abali mu buumi bwa kubusya mwana alimwi ne ndenda ya mulomo wa cibeleko. "
            "Tangi mpoonya mpoonyo ku mpendulo. "
            "Malizya a kusinsimuna kufwaafwi kuti ulwazi ulu talusanduki ku lwandano lwa dokotela.\n\n"
        ),
        "bemba": (
            "Uli kapyunga wa buumi uwashintilila pa buumi bwa banakashi abali ne fumo pamo ne kansa ya mulomo wa cibeleshi. "
            "Tambilila ku lyasuko mwachindunji. "
            "Pwishisheni ne cilembelo cipepa icilelanda ati ubu busuma tabusendapo icifulo ca kuyeshiwa kuli dokota.\n\n"
        ),
        "lozi": (
            "Mu muthusi wa za mapilo wa bucwani ya iketile hahulu ku mapilo a basali baimana ni kankere ya mulomo wa sibeleko. "
            "Qalisa hanghang ka karabo. "
            "Felelisa ka temoso ye nyinyani ye e re ziboho ze ha zi nkeleli sibaka sa ku lekolwa ki dokota.\n\n"
        ),
    }.get(lang, (
        "You are a professional health assistant specializing in maternal health and cervical cancer for Dawa Health. "
        "Answer the user question using correct and evidence-based health information. "
        "DO NOT start with phrases like Okay, Sure, or Let me explain. "
        "Start directly with the answer. "
        "Include a brief disclaimer at the end stating that this information does not replace a doctor evaluation. "
        "If asked about home visits, Dawa Health clinicians do home visits. "
        f"Contact: email={company_email}, phone={company_phone}, address={company_address}, website={company_website}.\n\n"
    ))

    # FIX: language enforcement moved to END of prompt
    prompt = f"{instruction_body}{question}\n\n{lang_enforce}"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt)
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini_general] Blocked/empty lang={lang}: {ve}")
            try:
                finish = response.candidates[0].finish_reason
                logging.warning(f"[ask_gemini_general] Finish reason: {finish}")
            except Exception:
                pass
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini_general Error] {type(e).__name__}: {e}")
        return fallback


def handle_ask_week(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    try:
        week_num = int(re.sub(r"\D", "", prompt))
        if 1 <= week_num <= 40:
            pregnancy_info = get_pregnancy_data(lang)
            info_text = str(pregnancy_info)
            
            logging.info(f"Searching for week {week_num} in {lang} data")
            
            if lang == "ndebele":
                week_found = False
                lines = info_text.split('\n')
                response_lines = []
                
                ndebele_week_names = {
                    1: "yokuqala", 2: "yesibili", 3: "yesithathu", 4: "yesine", 
                    5: "yesihlanu", 6: "yesithupha", 7: "yesikhombisa", 8: "yesishiyagalombili",
                    9: "yesishiyagalolunye", 10: "yetshumi", 11: "yetshumi nanye", 12: "yetshumi nambili",
                    13: "yetshumi nantathu", 14: "yetshumi nane", 15: "yetshumi nanhlanu",
                    16: "yetshumi nesithupha", 17: "yetshumi nesikhombisa", 18: "yetshumi nesishiyagalombili",
                    19: "yetshumi nesishiyagalolunye", 20: "lamashumi amabili", 21: "lamashumi amabili nesinye",
                    22: "lamashumi amabili nambili", 23: "lamashumi amabili nantathu", 24: "lamashumi amabili nane",
                    25: "lamashumi amabili nanhlanu", 26: "lamashumi amabili nesithupha", 27: "lamashumi amabili nesikhombisa",
                    28: "lamashumi amabili nesishiyagalombili", 29: "lamashumi amabili nesishiyagalolunye", 30: "lamashumi amathathu",
                    31: "lamashumi amathathu nesinye", 32: "lamashumi amathathu nambili", 33: "lamashumi amathathu nantathu",
                    34: "lamashumi amathathu nane", 35: "lamashumi amathathu nanhlanu", 36: "lamashumi amathathu nesithupha",
                    37: "lamashumi amathathu nesikhombisa", 38: "lamashumi amathathu nesishiyagalombili", 
                    39: "lamashumi amathathu nesishiyagalolunye", 40: "lamashumi amane"
                }
                
                week_name = ndebele_week_names.get(week_num, f"#{week_num}")
                
                patterns = [
                    f"*Iviki {week_name}:",
                    f"*Iviki {week_name} :",
                    f"*Iviki {week_name}*",
                    f"*Iviki {week_name} (",
                    f"*Iviki {week_name}(",
                    f"*Iviki {week_name} :*"
                ]
                
                found_start = False
                for i, line in enumerate(lines):
                    if not found_start:
                        for pattern in patterns:
                            if pattern in line:
                                found_start = True
                                response_lines.append(line)
                                break
                    else:
                        if i + 1 < len(lines) and any(
                            f"*Iviki {ndebele_week_names.get(w, '')}" in lines[i + 1] 
                            for w in range(week_num + 1, min(week_num + 5, 41))
                        ):
                            break
                        elif line.strip().startswith('*Iviki ') and week_name not in line:
                            break
                        elif line.strip() and not line.strip().startswith('***'):
                            response_lines.append(line)
                
                if response_lines:
                    week_info = '\n'.join(response_lines)
                    week_info = re.sub(r'\*Iviki [^*]*$', '', week_info)
                    send(f"Ulwazi lwe *Iviki {week_num}:*\n\n{week_info}", sender, phone_id)
                    week_found = True
                    ask_another_week(sender, phone_id)
                
                if not week_found:
                    week_patterns = [f"Week {week_num}", f"({week_num})", f": {week_num}:"]
                    for pattern in week_patterns:
                        if pattern in info_text:
                            start_idx = info_text.find(pattern)
                            end_idx = min(start_idx + 2000, len(info_text))
                            next_week = min([info_text.find(f"Week {w}", start_idx + 10) for w in range(week_num + 1, 41) if info_text.find(f"Week {w}", start_idx + 10) != -1] or [end_idx])
                            section = info_text[start_idx:next_week].strip()
                            send(f"Ulwazi lwe Iviki {week_num}:\n\n{section}", sender, phone_id)
                            week_found = True
                            ask_another_week(sender, phone_id)
                            break
                
                if not week_found:
                    send(f"Uxolo, angikutholanga ulwazi lweviki {week_num}. Zama elinye iviki kusuka ku-1 kuya ku-40.", sender, phone_id)
            
            elif lang == "chinyanja":
                pattern = rf"\*Sabata {week_num}:.*?(?=\*Sabata {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Zambiri za *Sabata {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Pepani, sindinapeze zambiri za sabata {week_num}. Yesani sabata lina kuchokera pa 1 mpaka 40.", sender, phone_id)
            
            elif lang == "shona":
                pattern = rf"\*Vhiki {week_num}:.*?(?=\*Vhiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Ruzivo rwe *Vhiki {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndine urombo, handina kuwana ruzivo rwevhiki {week_num}. Edza imwe vhiki kubva pa 1 kusvika pa 40.", sender, phone_id)

            elif lang == "lozi":
                pattern = rf"\*Sunda {week_num}:.*?(?=\*Sunda {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Liziba la *Sunda {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ni maswabi, ha ni a fumana liziba la vhiki {week_num}. Linge sunda ye n'wi ku zwana 1 ku ya ku 40.", sender, phone_id)

            elif lang == "bemba":
                pattern = rf"\*Umulungu {week_num}:.*?(?=\*Umulungu {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Icibeela ca *Mulungu {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Natapa, nshasangile icibeela ca mulungu {week_num}. Esheni mulungu ubi ku 1 ukufika ku 40.", sender, phone_id)

            elif lang == "tonga":
                pattern = rf"\*Nhwiiiki {week_num}:.*?(?=\*Nhwiiiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Cibeela ca *Nhwiiiki {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndatola, tana kuwana cibeela ca nhwiiiki {week_num}. Lingenya vhiki linzwi kuzwa 1 kusika 40.", sender, phone_id)
            
            else:
                pattern = rf"\*Week {week_num}:.*?(?=\*Week {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Here's information for *Week {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Sorry, I couldn't find information for week {week_num}. Please try another week from 1 to 40.", sender, phone_id)
        
        else:
            range_map = {
                "shona": "Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.",
                "ndebele": "Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.",
                "bemba": "Napapita, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.",
                "chinyanja": "Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.",
                "tonga": "Ndatola, ingila vhiki kuzwa 1 kusika 40 pe.",
                "lozi": "Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.",
            }
            send(range_map.get(lang, "Please enter a week between 1 and 40 only."), sender, phone_id)
            
    except ValueError:
        range_map = {
            "shona": "Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.",
            "ndebele": "Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.",
            "bemba": "Napapita, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.",
            "chinyanja": "Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.",
            "tonga": "Ndatola, ingila vhiki kuzwa 1 kusika 40 pe.",
            "lozi": "Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.",
        }
        send(range_map.get(lang, "Please enter a week between 1 and 40 only."), sender, phone_id)
            

@app.route("/", methods=["GET"])
def home():
    return render_template("connected.html")

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == "BOT":
            return challenge, 200
        else:
            return "Failed", 403

    elif request.method == "POST":
        try:
            data = request.get_json()
            logging.info(f"Received webhook data: {data}")
            
            if "entry" in data:
                for entry in data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if "value" in change:
                                value = change["value"]
                                
                                if "messages" in value:
                                    for message in value["messages"]:
                                        sender = message["from"]
                                        phone_id = value["metadata"]["phone_number_id"]
                                        
                                        if "text" in message:
                                            prompt = message["text"]["body"]
                                            logging.info(f"Processing message from {sender}: {prompt}")
                                            
                                            # Ensure state exists
                                            is_new = ensure_user_state(sender)
                                            if not is_new:
                                                user_states[sender]["first_message"] = False

                                            # FIX: Re-detect language on every message
                                            maybe_update_language(sender, prompt)
                                            
                                            save_user_conversation(sender, "user", prompt)
                                            handle_conversation_state(sender, prompt, phone_id)
                                            
                                        else:
                                            logging.info(f"Non-text message received from {sender}")
                                            ensure_user_state(sender)
                                            state = user_states.get(sender, {})
                                            lang = state.get("language", "english")
                                            non_text_map = {
                                                "shona": "Ndine urombo, handigoni kugamuchira mameseji asiri mavara chete. Ndapota tumira meseji yemavara.",
                                                "ndebele": "Uxolo, angikwazi ukwamukela imilayezo engeyona imibhalo kuphela. Sicela uthumele umlayezo wombhalo.",
                                                "bemba": "Natapa, nshakwanishe ukupokeela amameseji yambi ukucila pa menso. Napapita, tuma ubutumwa bwamenso.",
                                                "chinyanja": "Pepani, sindingathe kulandira mameseji enama osati a zilembo. Chonde tumirani meseji ya zilembo.",
                                                "tonga": "Ndazwa kwiinda, tani konzy kujana mameseji aambi kusikwa aa mabbala. Ndatola, tuma meseji ya mabbala.",
                                                "lozi": "Ni maswabi, ha na kona kuzwela miiala yeng'wi kufita feela ya mangolo. Ndapota, lumeza molaala wa mangolo.",
                                            }
                                            send(non_text_map.get(lang, "Sorry, I can only process text messages. Please send a text message."), sender, phone_id)
                                
                                elif "statuses" in value:
                                    logging.info("Message status update received, ignoring.")
                                
                                else:
                                    logging.info("Webhook received non-message event, ignoring.")

        except Exception as e:
            logging.error(f"Error in webhook: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500
        
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    load_user_states()
    app.run(host="0.0.0.0", port=5000, debug=True)
