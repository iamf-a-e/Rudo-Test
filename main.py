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

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[])


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

# Keep save_user_states as a convenience wrapper (saves ALL currently cached users)
def save_user_states():
    """Save all in-memory user states to Redis (one key per user)."""
    for sender in list(user_states.keys()):
        save_single_user_state(sender)

# load_user_states is only used at startup now to warm the local cache (optional)
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
            return json.loads(history) if history else []
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return []
    return []

def save_user_conversation(sender, role, message):
    """Save user conversation to Upstash Redis"""
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            conversation.append({
                "role": role,
                "message": message,
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
        
    language_keywords = {
        "shona": [
            "mhoro", "mhoroi", "makadini", "hesi", "ndinonzi", "zvakanaka", "ndatenda",
            "pamuviri", "zvigadzirwa", "chirwere", "gomarara", "chibereko",
            "zviratidzo", "chiremba", "kubuda", "ropa", "kusvotwa", "kurwadziwa",
            "ndapota", "handina", "ndinoda", "unei", "ndiri", "uri", "tiri", "vari",
            "zviri", "zvichiri", "ndicha", "ucha", "ticha", "vacha", "zvaka", "zvanga",
            "zvichava", "zvichange", "zvichada", "zvichaita", "mumwe", "vamwe", "zvimwe",
            "zvakare", "zvakadaro", "saka", "asi", "nekuti", "kana", "kuti", "uye",
            "kune", "kwete", "hapana", "ndizvo", "zvakadii", "zvakafanana", "zvakadaro"
        ],
        "ndebele": [
            "sawubona", "salibonani", "unjani", "yebo", "ngiyabonga", "ngicela",
            "isisu", "umntwana", "imikhiqizo", "isifo", "umhlaza", "isibeletho",
            "izimpawu", "udokotela", "ukuphuma", "igazi", "ukuhlanza", "ubuhlungu",
            "kuhle", "angikwazi", "ngifuna", "unani", "ngingani", "usuyini", "sisani",
            "basani", "kukhona", "kuzoba", "kuzobe", "kuzokwazi", "kuzokwenza", "iviki",
            "amaviki", "ukukhulelwa", "indlunkulu", "ubuchwepheshe", "umuntu", "abantu",
            "izinto", "futhi", "kodwa", "ngoba", "uma", "ukuthi", "noma", "kepha", "cha",
            "akukho", "impela", "kangaka", "kakhulu"
        ],
        "chinyanja": [
            "moni", "muli bwanji", "bwanji", "zikomo", "ndapota", "pepani", "pakati",
            "zogulitsa", "matenda", "kansa", "zizindikiro", "dokotala", "kutuluka",
            "magazi", "kutentha", "kuopseza", "zabwino", "sindikudziwa", "ndikufuna",
            "uli ndi chani", "ndili", "uli", "tili", "ali", "zili", "zichiri", "ndidza",
            "udza", "tidza", "adza", "zaka", "zanga", "zichava", "zichabe", "zichada",
            "zichita", "wina", "ena", "zina", "zabwino", "kotero", "koma", "chifukwa",
            "ngati", "kuti", "ndipo", "kupita", "ayi", "palibe", "ndithu", "kodi",
            "kusuta", "kumawononga", "amene", "ndikumuyembekezera", "sabata", "zambiri",
            "funso", "masabata", "thanzo", "zinthu", "mavitamini", "zoyezera", "liti",
            "lotani", "monga", "mwanj", "pamene", "panopa", "pang'ono", "pamwamba"
        ],
        "lozi": [
            "mwa bona", "mwa amukelwa", "uli bwanji", "ee", "ndalumba", "ndapota",
            "mbele", "ngwana", "zintu", "mulimo", "lwalelo", "mubili", "zibonelelo",
            "ngaka", "kuhula", "mali", "kushisa", "kuzwisa buhlungu", "zande",
            "ha ndi zibi", "ni na", "ndina", "sina", "bana", "bali", "zili",
            "ku na", "ku ka ba", "ku ka konwa", "ku ka etwa", "viki", "maviki",
            "ku imelela mwana", "mutango", "maano", "muntu", "bantu",
            "zintu", "hape", "kamba", "ka mulandu wa", "haiba", "kuti",
            "kapa", "fela", "haa", "hakuna", "handi", "ngana", "kakhulu",
            "lini", "kai", "mutokolo", "mikopano", "lipuzo", "milimo",
            "matanga", "zivita", "mavita", "ku leka", "ku nwa", "ku ja",
            "mufuta", "mupilo", "mubonelelo", "mubulelo", "mufuta wa mubili",
            "ka nako", "sika", "sina nako", "mwahala", "cwalo", "muzuhile cwani", "kimusihali", "kimanzibwana", "mucwani", "mulumele", "lumela", "cwalo cwalo"
        ],
        "bemba": [
            "mwaiseni", "muli shani", "shani", "ulishani", "nalikutemwa", "natotela",
            "twatotela", "ee", "awe", "limbi", "nshishibe", "napapata", "mukwai",
            "icimbusu", "ngafweniko", "njafweniko", "bushe", "landa panono", "ifyo",
            "cilikwisa", "umulungu", "mailo", "lelo", "pali cimo", "pali cibili",
            "ulucelo", "icungulo", "ubushiku", "umuntu", "abantu", "umwana",
            "abaice", "ifyakulya", "amanina", "inshita", "umwaice", "umukashana",
            "umulumendo", "ukutemwa", "ukwenda", "ukwisa", "ukuya", "ukumona",
            "ukulanda", "ukumfwa", "ukubomba", "ukuteya", "bwino", "fye", "sana",
            "ukucilapo", "ukucepako", "icisungu", "icibemba", "shaleenipo",
            "twalamonana", "mwashibukeni", "mwabombeni", "sendamenipo", "kabiyeni"
        ],
        "tonga": [
            "mwalumela", "mwabuka buti", "mwalandwa buti", "ndatotela",
            "twatotela", "yebo", "ehe", "iyayi", "kapati", "ndapota", "pepani",
            "komboni", "muku", "mwana", "mwana musankwa", "mwana mwanakazi", "cisamu",
            "kulya", "kumwa", "kucita", "kuya", "kwiza", "kumona", "kuzyiba", "kuvwwa",
            "kubomba", "mbomba buti", "njise", "njaku", "njitu", "njibotu", "njibiyabi",
            "lili", "lindi", "lino", "majana", "mazuba", "mabbali", "kuzwa", "mpi",
            "kuti", "nchito", "ng'anda", "cisima", "bambo", "mama", "bamakwe", "mayo",
            "sekulu", "nkuku", "kwendela", "ibvu", "musamu", "muti", "luwombo",
            "ndisimutwe", "ndatola"
        ],
        "english": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                   "how are you", "what's up", "hey there", "hi there", "help", "please",
                   "thank you", "thanks", "yes", "no", "ok", "okay", "sorry"]
    }
    
    exact_matches = {
        "shona": ["mhoro", "mhoroi", "makadini", "hesi", "hapana", "ndizvo", "zvakanaka", "wadini", "taura", "ehe", "kwete"],
        "ndebele": ["sawubona", "salibonani", "unjani", "yebo", "ngiyabonga", "ngicela", "cha", "impela", "kunjani", "hatshi", "kambe"],
        "bemba": ["mwaiseni", "muli shani", "shani", "ulishani", "nalikutemwa", "natotela", "twatotela", "ee", "awe", "limbi", "bwino", "mukwai"],
        "chinyanja": ["moni", "muli bwanji", "bwanji", "zikomo", "ndapota", "pepani", "ayi", "ndithu", "kodi", "inde", "chonde", "eyaa"],
        "tonga": ["mwabuka buti", "mwalandwa buti", "mulibwanji", "ndatotela", "twatotela", "kapati", "ndapota", "ehe", "iyayi", "yebo", "mbwa"],
        "lozi": ["mwa bona", "uli bwanji", "ee", "ndalumba", "ndapota", "pepani", "haa", "handi", "luna", "inde", "kuli", "kacenu"],
        "english": ["hi", "hello", "hey", "hie", "yes", "no", "ok", "thanks", "thank you", "good", "great", "please"]
    }
    
    for lang, words in exact_matches.items():
        if message_lower in words:
            logging.info(f"Exact match detected: {message_lower} -> {lang}")
            return lang
    
    language_phrases = {
        "chinyanja": [
            "muli bwanji", "uli ndi chani", "kodi", "ndikufuna", "sindikudziwa",
            "ndikumuyembekezera", "pakati panga", "sabata la", "zambiri za", 
            "ndapota", "zikomo kwambiri", "muli bwino", "ndili bwino"
        ],
        "shona": [
            "makadini", "unei", "ndinoda", "handina", "ndiri", "uri", "tiri",
            "zviri", "zvichiri", "ndicha", "zvakanaka sei", "ndapota", "taura",
            "ehe", "kwete", "ndatenda", "ndiriku", "variku"
        ],
        "ndebele": [
            "unjani", "ungalokhu", "ungathanda", "angikwazi", "ngifuna", "usuyini",
            "ngiyabonga", "sicela", "kunjani", "hatshi", "kambe", "lapha", "khona"
        ],
        "lozi": [
            "uli bwanji", "una ni", "kai", "ni bata", "ha ndi zibi", "ni mu embeleza",
            "mukati ka mina", "viki ya", "zintu zeñi", "ndapota", "ndalumba",
            "kuli", "kacenu", "handi zibi", "luna", "inde"
        ],
        "bemba": [
            "muli shani", "ulishani", "nalikutemwa", "natotela", "twatotela", 
            "napapita", "nshishibe", "ngafweniko", "bushe", "cilikwisa", 
            "ukufuma", "ukufika", "pali cimo", "pali cibili", "mulungu",
            "ifintu", "ifyakulya", "ukumona", "ukulanda", "bwino"
        ],
        "tonga": [
            "mulibwanji", "mwabuka buti", "mwalandwa buti", "ndatotela", 
            "twatotela", "kapati", "ndapota", "ndazwa kwiinda", "ndisimutwe",
            "mbomba buti", "njise", "njaku", "njitu", "lili", "lino", "kuti",
            "ng'anda", "cisima", "bambo", "mama", "kuzwa", "kusika"
        ],
        "english": [
            "how are you", "what's up", "i want", "i don't know", "i'm waiting",
            "between", "week of", "more about", "please", "thank you", "sorry",
            "where is", "when", "today", "tomorrow", "yesterday", "good morning",
            "good afternoon", "good evening", "help me", "i need"
        ]
    }
    
    phrase_scores = {"shona": 0, "ndebele": 0, "chinyanja": 0, "lozi": 0, "english": 0, "bemba": 0, "tonga": 0}
    for lang, phrases in language_phrases.items():
        for phrase in phrases:
            if phrase in message_lower:
                phrase_scores[lang] += 5
    
    scores = {}
    for lang, keywords in language_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in message_lower:
                if f" {keyword} " in f" {message_lower} ":
                    score += 3
                else:
                    score += 1
        scores[lang] = score + phrase_scores.get(lang, 0)
    
    max_score = max(scores.values())
    if max_score > 0:
        detected_lang = max(scores, key=scores.get)
        logging.info(f"Language detected: {detected_lang} with score {max_score}")
        
        current_lang = user_states.get(sender, {}).get("language", "english")
        if detected_lang != current_lang and max_score < 3:
            logging.info(f"Low confidence switch ({max_score}), keeping current language: {current_lang}")
            return current_lang
            
        return detected_lang
    
    logging.info("No specific language detected, defaulting to English")
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
    detected_lang = detect_language(prompt)
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
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi"]

    if any(response in prompt_lower for response in no_responses):

        topic = state.get("topic")

        if topic == "maternal":
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri?", sender, phone_id)
            elif lang == "ndebele":
                send("Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu?", sender, phone_id)
            elif lang == "chinyanja":
                send("Zikomo! Kodi mukufuna kugula zinthu za Thanzi la Amayi?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase maternal health products?", sender, phone_id)

        elif topic == "cervical":
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zve cervical cancer?", sender, phone_id)
            elif lang == "ndebele":
                send("Ngiyabonga! Ungathanda ukuthengwa izinto zokuvikela isilonda somlomo wesibeletho?", sender, phone_id)
            elif lang == "chinyanja":
                send("Zikomo! Kodi mukufuna kugula zinthu za cervical cancer?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase cervical cancer products?", sender, phone_id)

        else:
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano?", sender, phone_id)
            elif lang == "ndebele":
                send("Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela impilo?", sender, phone_id)
            elif lang == "chinyanja":
                send("Zikomo! Kodi mukufuna kugula zinthu za thanzo?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase health products?", sender, phone_id)

        state["step"] = "product_inquiry"
        save_single_user_state(sender)
        return

    else:
        if lang == "shona":
            send("Ndiri kufunga...", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyacabangisisa...", sender, phone_id)
        elif lang == "chinyanja":
            send("Ndikuganiza...", sender, phone_id)
        else:
            send("Let me think...", sender, phone_id)

        reply = ask_gemini_general(prompt, lang)
        send(reply, sender, phone_id)

        if lang == "shona":
            send("Pane chimwe chamunoda kubvunza here?", sender, phone_id)
        elif lang == "ndebele":
            send("Uneminye imibuzo yini?", sender, phone_id)
        elif lang == "chinyanja":
            send("Kodi muli ndi mafunso ena?", sender, phone_id)
        elif lang == "tonga":
            send("Uli ne mabvuzo yanga yonse?", sender, phone_id)
        elif lang == "bemba":
            send("Uli ne fimo fyandi ifyakulya?", sender, phone_id)
        elif lang == "lozi":
            send("O na mabvuzo a mangi?", sender, phone_id)
        else:
            send("Do you have any more questions?", sender, phone_id)

        state["step"] = "general_followup"
        save_single_user_state(sender)


def is_exact_match(text, responses):
    words = re.findall(r"\b\w+\b", text)
    return any(word in responses for word in words)

def handle_general_followup(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde"]
    no_responses = ["no", "nah", "aiwa", "kwete", "hapana", "nope"]

    if any(r in prompt_lower for r in yes_responses):
        if lang == "shona":
            send("Bvunzai mubvunzo wenyu.", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyacela ubuze umbuzo wakho.", sender, phone_id)
        elif lang == "tonga":
            send("Nkumbira ubvunze mubvuzo wako.", sender, phone_id)
        elif lang == "chinyanja":
            send("Chonde funsani funso lanu.", sender, phone_id)
        elif lang == "bemba":
            send("Nomba, lwishibe fimo lyobe.", sender, phone_id)
        elif lang == "lozi":
            send("Nkumbira ubuze mubvuzo wako.", sender, phone_id)
        else:
            send("Please ask your question.", sender, phone_id)

        state["step"] = "general_question"
        save_single_user_state(sender)
        return

    if any(r in prompt_lower for r in no_responses):
        if lang == "shona":
            send("Ndatenda! Iva nezuva rakanaka.", sender, phone_id)
        else:
            send("Thank you! Have a good day.", sender, phone_id)

        reset_conversation(sender)
        return

    if lang == "shona":
        send("Ndiri kufunga...", sender, phone_id)
    elif lang == "ndebele":
        send("Ngiyacabangisisa...", sender, phone_id)
    elif lang == "chinyanja":
        send("Ndikuganiza...", sender, phone_id)
    else:
        send("Let me think...", sender, phone_id)

    reply = ask_gemini_general(prompt, lang)
    send(reply, sender, phone_id)

    if lang == "shona":
        send("Pane chimwe chamunoda kubvunza here?", sender, phone_id)
    elif lang == "ndebele":
        send("Uneminye imibuzo yini?", sender, phone_id)
    elif lang == "tonga":
        send("Uli ne mabvuzo yanga yonse?", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi muli ndi mafunso ena?", sender, phone_id)
    elif lang == "bemba":
        send("Uli ne fimo fyandi ifyakulya?", sender, phone_id)
    elif lang == "lozi":
        send("O na mabvuzo a mangi?", sender, phone_id)
    else:
        send("Do you have any more questions?", sender, phone_id)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def ask_follow_up_question(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Pane chimwe chandingakubatsira nacho here?", sender, phone_id)
    elif lang == "ndebele":
        send("Ingabe kukhona okunye engingakusiza ngakho?", sender, phone_id)
    elif lang == "tonga":
        send("Kuli chinco nchingakusebelesya nacho", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi pali zina zomwe ndingakuthandizireni?", sender, phone_id)
    elif lang == "bemba":
        send("Kuli fintu fyalumo nshingafye", sender, phone_id)
    elif lang == "lozi":
        send("Ki sina sika ni ka thusa ka sona", sender, phone_id)
    else:
        send("Is there anything else I can help you with?", sender, phone_id)
    
    state["step"] = "follow_up"
    save_single_user_state(sender)


def switch_language_and_respond(sender, prompt, phone_id, current_lang, detected_lang):
    state = user_states[sender]
    state["language"] = detected_lang
    
    current_step = state.get("step", "main_menu")
    logging.info(f"Language switch detected: {current_lang} -> {detected_lang} at step: {current_step}")
    
    if current_step == "main_menu":
        if detected_lang == "shona":
            send("Mhoroi! Ndingakubatsirei nhasi?", sender, phone_id)
        elif detected_lang == "ndebele":
            send("Sawubona! Ngingakusiza ngani namuhla?", sender, phone_id)
        elif detected_lang == "chinyanja":
            send("Moni! Ndingakuthandizireni lero?", sender, phone_id)
        elif detected_lang == "lozi":
            send("Mwa bona! Nka ku thusa ka mini sunu?", sender, phone_id)
        elif detected_lang == "tonga":
            send("Moni! Ndingamwafwa shani ilelo?", sender, phone_id)
        elif detected_lang == "bemba":
            send("Muli shani! Bushe kuti namwafwa shani lelo?", sender, phone_id)    
        else:
            send("Hello! How can I help you today?", sender, phone_id)
    
    elif current_step == "registration":
        if state.get("phone_digits") is None:
            if detected_lang == "shona":
                send("Mhoro! Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.", sender, phone_id)
            elif detected_lang == "ndebele":
                send("Sawubona! Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.", sender, phone_id)
            elif detected_lang == "chinyanja":
                send("Moni! Tiyambireni ndi kulembetsa. Chonde ndipatseni manambala anayi omaliza a nambala yanu yafoni.", sender, phone_id)
            elif detected_lang == "lozi":
                send("Mwa bona! A re simule ka ku itambula. Dinomolo za mafelele a lina la wena ki zini za mafelele a mane?", sender, phone_id)    
            else:
                send("Hello! Let's start with registration. What is the last 4 digits of your number?", sender, phone_id)
    
    elif current_step == "ask_week":
        if detected_lang == "shona":
            send("Ndapota isa vhiki re pamuviri ", sender, phone_id)
        elif detected_lang == "ndebele":
            send("Sicela ufake iviki lokukhulelwa ", sender, phone_id)
        elif detected_lang == "chinyanja":
            send("Chonde lowetsani sabata la pakati ", sender, phone_id)
        elif detected_lang == "lozi":
            send("Ndapota faka linomolo la viki ya ku imelela mwana ", sender, phone_id)    
        else:
            send("Please enter your pregnancy week number ", sender, phone_id)
    
    save_single_user_state(sender)


def handle_main_menu(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    logging.info(f"User {sender} said: '{prompt}' (lowercase: '{prompt_lower}')")
    logging.info(f"Current state: step={state.get('step')}, topic={state.get('topic')}, language={lang}")

    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset", "help"]
    greeting_words = ["hi", "hello", "hey", "hie", "mhoro", "mhoroi", "sawubona", "salibonani", "hey", "hi there", "good morning", "good afternoon", "good evening", "moni", "muli bwanji"]
    
    is_reset = any(keyword in prompt_lower for keyword in reset_keywords)
    is_greeting = (
        any(prompt_lower.strip() == word for word in greeting_words) or
        any(re.search(rf"\b{re.escape(word)}\b", prompt_lower) for word in greeting_words)
    )
    
    if is_greeting or is_reset:
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        if lang == "shona":
            send("Mhoroi! Ndingakubatsirei nhasi?", sender, phone_id)
        elif lang == "ndebele":
            send("Sawubona! Ngingakusiza ngani namuhla?", sender, phone_id)
        elif lang == "chinyanja":
            send("Moni! Ndingakuthandizireni lero?", sender, phone_id)
        elif lang == "lozi":
            send("Mwa bona! Nka ku thusa ka mini sunu?", sender, phone_id) 
        elif lang == "tonga":
            send("Moni! Ndingamwafwa shani ilelo?", sender, phone_id)
        elif lang == "bemba":
            send("Muli shani! Bushe kuti namwafwa shani lelo?", sender, phone_id)    
        else:
            send("Hello! How can I help you today?", sender, phone_id)
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
                if lang == "shona":
                    send("Ndapota isa vhiki re pamuviri ", sender, phone_id)
                elif lang == "ndebele":
                    send("Sicela ufake iviki lokukhulelwa ", sender, phone_id)
                elif lang == "chinyanja":
                    send("Chonde lowetsani sabata la pakati ", sender, phone_id)
                elif lang == "lozi":
                    send("Ndapota faka linomolo la viki ya ku imelela mwana ", sender, phone_id)    
                else:
                    send("Please enter your pregnancy week number:", sender, phone_id)
            elif state.get("topic") == "cervical":
                cervical_data = get_cervical_data(lang)
                if cervical_data and len(cervical_data) > 0:
                    send(str(cervical_data[0]), sender, phone_id)
                else:
                    if lang == "shona":
                        send("Ndine urombo, handina kuwana ruzivo rwe cervical cancer parizvino.", sender, phone_id)
                    elif lang == "ndebele":
                        send("Uxolo, anginayo imininingwane ye-cervical cancer okwamanje.", sender, phone_id)
                    elif lang == "chinyanja":
                        send("Pepani, sindinapeze zambiri za cervical cancer panopa.", sender, phone_id)
                    elif lang == "lozi":
                        send("Ndine u luvile, sina kungafumula zintu za kankere ya sibete sunu.", sender, phone_id)    
                    else:
                        send("Sorry, I couldn't find cervical cancer information at the moment.", sender, phone_id)
                
                ask_cervical_more_info(sender, phone_id)
            save_single_user_state(sender)
            return

        elif prompt_lower in ["2", "specific", "question", "questions", "mubvunzo", "umbuzo", "funso"]:
            if state.get("topic") == "maternal":
                state["step"] = "maternal_question_choice"
                if lang == "shona":
                    send(
                        "Sarudza mubvunzo:\n"
                        "1. Ndezvikaita zviratidzo zvepamuviri?\n"
                        "2. Ndeapi marairiro ezvokudya?\n"
                        "3. Ndingafanire kuona chiremba riini?",
                        sender, phone_id
                    )
                elif lang == "ndebele":
                    send(
                        "Khetha umbuzo:\n"
                        "1. Ngabe yiziphi izimpawu zesisu?\n"
                        "2. Ngabe yimaphi amathiphu okudla?\n"
                        "3. Ngabe kufanele ngibone udokotela nini?",
                        sender, phone_id
                    )
                elif lang == "chinyanja":
                    send(
                        "Sankhani funso:\n"
                        "1. Ndi zizindikiro zotani za pakati?\n"
                        "2. Ndi malangizo otani okudya?\n"
                        "3. Ndingafunire kuona dokotala liti?",
                        sender, phone_id
                    )
                elif lang == "lozi":
                    send(
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Zibonelelo ze ku imelela mwana zezi ntini?\n"
                        "2. Ni maano a ku nwa zintu za bupilo a ka landelwa?\n"
                        "3. Nini nka ya kwa dokotela?",
                        sender, phone_id
                    )  
                else:
                    send(
                        "You can choose a question or ask any of your own.\n"
                        "1. What are common pregnancy symptoms?\n"
                        "2. What nutrition tips should I follow?\n"
                        "3. When should I see a doctor?",
                        sender, phone_id
                    )
            elif state.get("topic") == "cervical":
                state["step"] = "cervical_question_choice"
                if lang == "shona":
                    send(
                        "Sarudza mubvunzo:\n"
                        "1. Chii chinonzi cervical cancer?\n"
                        "2. Ndezvipi zviratidzo zvekutanga zvecervical cancer?\n"
                        "3. Chii chinokonzera cervical cancer?",
                        sender, phone_id
                    )
                elif lang == "ndebele":
                    send(
                        "Khetha umbuzo:\n"
                        "1. Yini i-cervical cancer?\n"
                        "2. Ngabe yiziphi izimpawu zokuqala ze-cervical cancer?\n"
                        "3. Yini ebangela i-cervical cancer?",
                        sender, phone_id
                    )
                elif lang == "chinyanja":
                    send(
                        "Sankhani funso:\n"
                        "1. Ndi chiyani cervical cancer?\n"
                        "2. Ndi zizindikiro zotani zoyamba za cervical cancer?\n"
                        "3. Ndi chiyani chimayambitsa cervical cancer?",
                        sender, phone_id
                    )
                elif lang == "lozi":
                    send(
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Kankere ya sibete sa bomme ki yini?\n"
                        "2. Zibonelelo za kutanga za kankere ya sibete zezi ntini?\n"
                        "3. Zini zi bakela kankere ya sibete?",
                        sender, phone_id
                    )    
                else:
                    send(
                        "You can choose a question or ask any of your own.\n"
                        "1. What is cervical cancer?\n"
                        "2. What are the early symptoms of cervical cancer?\n"
                        "3. What causes cervical cancer?",
                        sender, phone_id
                    )
            save_single_user_state(sender)
            return

        else:
            if lang == "shona":
                send("Pindura ne '1' kuti uwane ruzivo kana '2' kuti ubvunze mibvunzo.", sender, phone_id)
            elif lang == "ndebele":
                send("Phendula ngo-'1' ukuze uthole ulwazi noma '2' ukuze ubuze imibuzo.", sender, phone_id)
            elif lang == "chinyanja":
                send("Yankhani ndi '1' kuti mupeze zambiri kapena '2' kuti mufunse mafunso.", sender, phone_id)
            elif lang == "lozi":
                send("", sender, phone_id)    
            else:
                send("Ndapota pindula na '1' ku lwisisa zintu ka bonya noma '2' ku mubuzo wa nene", sender, phone_id)
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
                else:
                    pattern = rf"\*Week {week}:.*?(?=\*Week {week+1}:|\Z)"
                    
                match = re.search(pattern, info_text, re.S)
                if match:
                    if lang == "shona":
                        send(f"Ruzivo rwe *Vhiki {week}:*\n\n{match.group(0)}", sender, phone_id)
                    elif lang == "ndebele":
                        send(f"Ulwazi lwe *Iviki {week}:*\n\n{match.group(0)}", sender, phone_id)
                    elif lang == "chinyanja":
                        send(f"Zambiri za *Sabata {week}:*\n\n{match.group(0)}", sender, phone_id)
                    elif lang == "lozi":
                        send(f"Yezi zintu za lwisisa ka bonya ku *Sunda {week}:*\n\n{match.group(0)}", sender, phone_id)    
                    else:
                        send(f"Here's information for *Week {week}:*\n\n{match.group(0)}", sender, phone_id)
                    
                    ask_another_week(sender, phone_id)
                else:
                    if lang == "shona":
                        send("Hapana ruzivo rwevhiki iyi.", sender, phone_id)
                    elif lang == "ndebele":
                        send("Alukho ulwazi lwaleviki.", sender, phone_id)
                    elif lang == "chinyanja":
                        send("Palibe zambiri za sabata ili.", sender, phone_id)
                    elif lang == "lozi":
                        send("Sina zintu za ku fumwa ka viki ye.", sender, phone_id)    
                    else:
                        send("No data available for that week.", sender, phone_id)
                    ask_another_week(sender, phone_id)
        except ValueError:
            if lang == "shona":
                send("Ndapota pinda nhamba chaiyo yevhiki kubva pa 1 kusvika pa 40.", sender, phone_id)
            elif lang == "ndebele":
                send("Sicela ufake inombolo yeviki evumelekile ephakathi kuka-1 no-40.", sender, phone_id)
            elif lang == "chinyanja":
                send("Chonde lowetsani nambala yoyenera ya sabata kuchokera pa 1 mpaka 40.", sender, phone_id)
            elif lang == "lozi":
                send("Ndapota faka linomolo la viki le li le ka 1 ku ya ka 40.", sender, phone_id)    
            else:
                send("Please enter a valid week number between 1 and 40.", sender, phone_id)
            ask_another_week(sender, phone_id)
        return  

    if state.get("step") == "maternal_question_choice":
        if prompt_lower in ["1", "symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
            if lang == "shona":
                send("Zviratidzo zvepamuviri zvinosanganisira kusvotwa, kuneta, kuvava mazamu, uye kuchinja mweya.", sender, phone_id)
            elif lang == "ndebele":
                send("Izimpawu zesisu zihlanganisa isicanucanu, ukukhathala, ubuhlungu bezebelé, nokushintsha kwemizwa.", sender, phone_id)
            elif lang == "chinyanja":
                send("Zizindikiro za pakati zimaphatikizapo kusanza, kulemba, kubvutika mabele, ndi kusintha kwa maganizo.", sender, phone_id)
            elif lang == "lozi":
                send("Limpande ze twayelehileng za buimana li akaretsa ho nyekeloa ke pelo, kukhathala, kubaba kwa matete ni kupotoloka kwa maikuto.", sender, phone_id)    
            else:
                send("Common pregnancy symptoms include nausea, fatigue, breast tenderness, and mood swings.", sender, phone_id)
    
        elif prompt_lower in ["2", "nutrition", "zvokudya", "ukudla", "kudya"]:
            if lang == "shona":
                send("Marairiro ezvokudya: Idya chikafu chakaringana, wedzera folic acid uye iron, uye nwa mvura yakawanda.", sender, phone_id)
            elif lang == "ndebele":
                send("Amathiphu okudla: Yidla ukudla okunempilo, khulisa i-folic acid ne-iron, futhi uhlale unamandla.", sender, phone_id)
            elif lang == "chinyanja":
                send("Malangizo okudya: Idyani chakudya chabwino, onjezerani folic acid ndi iron, ndipo muzikhala ndi madzi.", sender, phone_id)
            elif lang == "lozi":
                send("Litaba za swakudya: Ja swakudya se se lekalekanang, engetsa kufumana folic acid ni iron, mi u nne u nwa mezi a mangi.", sender, phone_id)    
            else:
                send("Nutrition tips: Eat balanced meals, increase folic acid and iron intake, and stay hydrated.", sender, phone_id)
    
        elif prompt_lower in ["3", "doctor", "chiremba", "udokotela", "dokotala"]:
            if lang == "shona":
                send("Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.", sender, phone_id)
            elif lang == "ndebele":
                send("Iya kudokotela uma unobuhlungu obukhulu, ukuphuma kwegazi okukhulu, noma imfiva ephezulu.", sender, phone_id)
            elif lang == "chinyanja":
                send("Pitani kudokotala ngati muli ndi kupweteka kwakukulu, kutuluka magazi ambiri, kapena malungo apamwamba.", sender, phone_id)
            elif lang == "lozi":
                send("Bona ngaka kapili ha u ka ba ni buhlungu bo boholo, kuelwa mali a mangi, kamba mufufutso o mutuna.", sender, phone_id)    
            else:
                send("See a doctor immediately if you experience severe pain, heavy bleeding, or high fever.", sender, phone_id)
    
        else:
            logging.info("DEBUG: Processing free-text maternal question")
            if lang == "shona":
                send("Kufunga...", sender, phone_id)
            elif lang == "ndebele":
                send("Ucabanga...", sender, phone_id)
            elif lang == "chinyanja":
                send("Kuganiza...", sender, phone_id)
            elif lang == "lozi":
                send("Kucabanga…", sender, phone_id)    
            else:
                send("Thinking...", sender, phone_id)
    
            gemini_response = ask_gemini(prompt, lang)  
            send(gemini_response, sender, phone_id)
        
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    if state.get("step") == "cervical_question_choice":
        if prompt_lower in ["1", "what is it", "what is cervical cancer", "chii", "yini", "chiyani"]:
            if lang == "shona":
                send("Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.", sender, phone_id)
            elif lang == "ndebele":
                send("I-cervical cancer yisifo se-cervix, ingxenye engezansi yesibeletho ehlobene nesibeletho. Yisifo somhlaza sesibili esivame kakhulu emhlabeni wonke futhi yisifo esivame kakhulu kwabesifazane eZambia. Isifo esingavinjwa futhi singelapheka, ikakhulukazi uma sitholakala ngokushesha.", sender, phone_id)
            elif lang == "chinyanja":
                send("Cervical cancer ndi matenda a cervix, gawo lotsika la chibereko lomwe limagwirizana ndi chibereko. Ndimatenda a kansa wachiwiri omwe amapezeka kwambiri padziko lapansi ndipo ndi omwe amachitika kwambiri kwa amayi ku Zambia. Matenda omwe angapweke ndi opatsirika, makamaka akadziwika msanga.", sender, phone_id)
            elif lang == "lozi":
                send("Kansa ya mulomo wa popelo ki malwale a mulomo wa popelo, sipande sa fafasi sa popelo se si kopanya kwa mukutu wa botsadi. Ki kansa ya bobeli e atile hahulu kwa basali mwa lifasi kaufela, mi ki yona e atile hahulu kwa basali mwa Zambia. Ki malwale a ka thibelwa ni ku alafiwa, haholoholo ha a lemohuoa kapili.", sender, phone_id)
            else:
                send("Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. It is the second most common female malignancy worldwide and the most common in females in Zambia. It is a preventable and treatable disease, especially when detected early.", sender, phone_id)
        elif prompt_lower in ["2", "symptoms", "early symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
            if lang == "shona":
                send("Mumatanho ekutanga, cervical cancer kazhinji haina zviratidzo zvinooneka. Ndokusaka kuongororwa nguva nenguva kwakakosha. Sezvo cancer ichikura, zviratidzo zvinogona kusanganisira kubuda ropa kusingawanzo (pakati penguva, mushure mekuita bonde, kana mushure mekuenda kumwedzi), kubuda kwezvipembenene zvinonhuwa, kana kurwadziwa panguva yekuita bonde.", sender, phone_id)
            elif lang == "ndebele":
                send("Ezitebhisini zokuqala, i-cervical cancer ivamise ukungabi nezimpawu ezibonakalayo. Yingakho ukuhlolwa ngesikhathi esithile kubalulekile. Njengoba umhlaza ukhula, izimpawu zingahlanganisa ukuphuma kwegazi okungajwayelekile (phakathi kwezikhathi, ngemva kokwenza ucansi, noma ngemva kokungena esikhathini sokugodla), ukuphuma kokomkhando olunephunga elibi, noma ubuhlungu ngesikhathi sokwenza ucansi.", sender, phone_id)
            elif lang == "chinyanja":
                send("M'magawo oyamba, cervical cancer imayambira mosazindikika. Ndi chifukwa chake kuyezetsa nthawi ndi nthawi ndi kofunikira. Pomwe kansa ikukula, zizindikiro zingakhale kutuluka magazi osayembekezereka (pakati pa nthawi, pambuyo pa kugonana, kapena pambuyo pa menopause), kutuluka kwa chinyezi choipa, kapena kupweteka panthawi ya kugonana.", sender, phone_id)
            else:
                send("In its early stages, cervical cancer often has no noticeable symptoms. This is why regular screening is so important. As the cancer progresses, symptoms may include unusual vaginal bleeding (between periods, after sex, or after menopause), foul-smelling vaginal discharge, or pain during sexual intercourse.", sender, phone_id)
        elif prompt_lower in ["3", "causes", "what causes it", "chikonzero", "izimbangela", "zoyambitsa"]:
            if lang == "shona":
                send("Kazhinji, cervical cancer inokonzerwa nehutachiona husingaperi hweHuman Papilloma Virus (HPV). HPV ihutachiona hwakajairika, hunotapuriranwa nekusangana pabonde. Kunyange immune system yemuviri ichibvisa hutachiona muvanhu vazhinji, hutachiona husingaperi hunogona kukonzera shanduko yamasero inogona kuzopedzisira yaita cancer.", sender, phone_id)
            elif lang == "ndebele":
                send("Ezimeni zonke, i-cervical cancer ibangelwa ukutheleleka okungapheli kwe-Human Papilloma Virus (HPV). I-HUV igciwane elivamile, elidluliselwa ngocansi. Ngenkathi amasosha omzimba emuncela igciwane kubantu abaningi, ukutheleleka okungapheli kungaholela ekushintsheni kwamaseli okungajwayelekile okungase igcine kube umhlaza.", sender, phone_id)
            elif lang == "chinyanja":
                send("M'magawo onse, cervical cancer imayambitsidwa ndi matenda osatha a Human Papilloma Virus (HPV). HPV ndi matenda amene amapezeka kwambiri, omwe amatengedwa pogonana. Pomwe immune system ya thupi imatulutsa matenda mwa anthu ambiri, matenda osatha angayambitse kusintha kwa maselo komwe kungatheka kukhala kansa.", sender, phone_id)
            else:
                send("In almost all cases, cervical cancer is caused by persistent infection with the Human Papilloma Virus (HPV). HPV is a very common, sexually transmitted virus. While the body's immune system clears the virus in most people, a persistent infection can lead to abnormal cell changes that may eventually develop into cancer.", sender, phone_id)
        else:
            if lang == "shona":
                send("Kufunga...", sender, phone_id)
            elif lang == "ndebele":
                send("Ucabanga...", sender, phone_id)
            elif lang == "chinyanja":
                send("Kuganiza...", sender, phone_id)
            else:
                send("Thinking...", sender, phone_id)
    
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
        if lang == "chinyanja":
            send("Kuganiza...", sender, phone_id)
        elif lang == "shona":
            send("Kufunga...", sender, phone_id)
        elif lang == "ndebele":
            send("Ucabanga...", sender, phone_id)
        else:
            send("Thinking...", sender, phone_id)
        
        gemini_response = ask_gemini(prompt, lang)
        send(gemini_response, sender, phone_id)
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    lang = state["language"]

    gemini_reply = ask_gemini_general(prompt, lang)
    send(gemini_reply, sender, phone_id)
    
    if lang == "shona":
        send("Muchiri nemumwe mubvunzo here?", sender, phone_id)
    elif lang == "ndebele":
        send("Ulomunye umbuzo yini?", sender, phone_id)
    elif lang == "tonga":
        send("Uli ne mabvuzo yanji yonse?", sender, phone_id)
    elif lang == "chinyanja":
        send("Muli ndi mafunso ena?", sender, phone_id)
    elif lang == "bemba":
        send("Uli ne fimo fyandi ifyakulya?", sender, phone_id)
    elif lang == "lozi":
        send("O na mabvuzo a mangi?", sender, phone_id)
    else:
        send("Do you have any more questions?", sender, phone_id)
    
    state["step"] = "general_followup"
    save_single_user_state(sender)
    return
    

def handle_purchase_response(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    
    if any(response in prompt_lower for response in no_responses):
        if lang == "shona":
            send("Ndatenda! Iva nezuva rakanaka. Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyabonga! Ube nosuku oluhle. Uma uneminye imibuzo, qala ingxoxo entsha ngo-'unjani'.", sender, phone_id)
        elif lang == "chinyanja":
            send("Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.", sender, phone_id)
        else:
            send("Thank you! Have a nice day. If you have more questions, start over by saying 'hi'.", sender, phone_id)
        
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
                if lang == "shona":
                    send("Ndine urombo, hapana zvigadzirwa zvehutano hwepamuviri zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.", sender, phone_id)
                elif lang == "ndebele":
                    send("Uxolo, azikho izinto zokunakekela isisu ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.", sender, phone_id)
                else:
                    send("Sorry, no maternal health products are currently available. We recommend visiting our clinic for more information.", sender, phone_id)
                
        elif topic == "cervical":
            cervical_products = extract_products_by_category("Cervical Cancer")
            if cervical_products:
                products_text = format_products_for_display(cervical_products, lang)
                send(products_text, sender, phone_id)
            else:
                if lang == "shona":
                    send("Ndine urombo, hapana zvigadzirwa zvecervical cancer zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.", sender, phone_id)
                elif lang == "ndebele":
                    send("Uxolo, azikho izinto zokuvikela isilonda somlomo wesibeletho ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.", sender, phone_id)
                else:
                    send("Sorry, no cervical cancer products are currently available. We recommend visiting our clinic for more information.", sender, phone_id)
        else:
            general_products = extract_products_by_category("General")
            if general_products:
                products_text = format_products_for_display(general_products, lang)
                send(products_text, sender, phone_id)
            else:
                if lang == "shona":
                    send("Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo.", sender, phone_id)
                elif lang == "ndebele":
                    send("Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane.", sender, phone_id)
                else:
                    send("Thank you! We'll contact you shortly for more details.", sender, phone_id)
        
        if lang == "shona":
            send("Ungada here kuenderera mberi nekutenga chimwe chezvigadzirwa izvi? ", sender, phone_id)
        elif lang == "ndebele":
            send("Ungathanda ukuqhubeka nokuthenga noma yini yale mikhiqizo? ", sender, phone_id)
        else:
            send("Would you like to proceed with purchasing any of these products? ", sender, phone_id)
        
        state["step"] = "confirm_purchase"
        save_single_user_state(sender)
        
    else:
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kutenga zvigadzirwa? ", sender, phone_id)
        elif lang == "ndebele":
            send("Angikuzwisisi. Phendula ngicela: Ungathanda ukuthenga imikhiqizo? ", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to purchase products?  ", sender, phone_id)


def handle_purchase_confirmation(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    
    if any(response in prompt_lower for response in no_responses):
        if lang == "shona":
            send("Zvakanaka. Tinokutendai! Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.", sender, phone_id)
        elif lang == "ndebele":
            send("Kulungile. Ngiyabonga! Uma uneminye imibuzo, qala kabusha ngokuthi 'unjani'.", sender, phone_id)
        elif lang == "chinyanja":
            send("Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.", sender, phone_id)
        else:
            send("Alright. Thank you! If you have more questions, start over by saying 'hi'.", sender, phone_id)
        reset_conversation(sender)
        
    elif any(response in prompt_lower for response in yes_responses):
        if lang == "shona":
            send("Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo nezvekutenga.", sender, phone_id)
        elif lang == "ndebele":
            send("Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane ngokuthenga.", sender, phone_id)
        else:
            send("Thank you! We'll contact you shortly for more details about your purchase.", sender, phone_id)
        reset_conversation(sender)
        
    else:
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuenderera mberi nekutenga? ", sender, phone_id)
        elif lang == "ndebele":
            send("Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokuthenga? ", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to proceed with purchasing?  ", sender, phone_id)


def extract_products_by_category(category_name):
    try:
        return products_by_category.get(category_name, [])
    except Exception as e:
        logging.error(f"Error extracting products for category {category_name}: {e}")
        return []

def format_products_for_display(products_list, lang):
    if not products_list:
        if lang == "shona":
            return "Hapana zvigadzirwa zvazvino onekwa."
        elif lang == "ndebele":
            return "Azikho imikhiqizo etholakalayo okwamanje."
        else:
            return "No products currently available."
    
    if lang == "shona":
        header = "🏥 Zvigadzirwa Zvehutano:\n\n"
    elif lang == "ndebele":
        header = "🏥 Imikhiqizo Yezempilo:\n\n"
    else:
        header = "🏥 Health Products:\n\n"
    
    products_text = header
    for i, product in enumerate(products_list, 1):
        name = product.get('name', 'Unknown Product')
        price = product.get('price', 'Price not available')
        availability = product.get('availability', 'Availability not specified')
        
        if lang == "shona":
            products_text += f"{i}. {name}\n"
            products_text += f"   💰 Mutengo: {price}\n"
            products_text += f"   📦 Kuwanikwa: {availability}\n\n"
        elif lang == "ndebele":
            products_text += f"{i}. {name}\n"
            products_text += f"   💰 Inani: {price}\n"
            products_text += f"   📦 Ukutholakala: {availability}\n\n"
        else:
            products_text += f"{i}. {name}\n"
            products_text += f"   💰 Price: {price}\n"
            products_text += f"   📦 Availability: {availability}\n\n"
    
    if lang == "shona":
        products_text += "Sarudza chirongwa nekuudza nhamba yacho."
    elif lang == "ndebele":
        products_text += "Khetha umkhiqizo ngokutshela inombolo yayo."
    else:
        products_text += "Select a product by telling us the number."
    
    return products_text

def handle_conversation_state(sender, prompt, phone_id):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()
    
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "reset", "help"]
    if any(keyword in prompt_lower for keyword in reset_keywords):
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        if lang == "shona":
            send("Ndingakubatsirei nhasi?", sender, phone_id)
        else:
            send("How can I help you today?", sender, phone_id)
        return

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
    elif current_step == "general_followup":
        handle_general_followup(sender, prompt, phone_id)
        return
    elif current_step == "general_question":
        lang = state["language"]
        reply = ask_gemini_general(prompt, lang)
        send(reply, sender, phone_id)
    
        if lang == "shona":
            send("Pane chimwe chamunoda kubvunza here?", sender, phone_id)
        elif lang == "ndebele":
            send("Uneminye imibuzo yini?", sender, phone_id)
        elif lang == "tonga":
            send("Uli ne mabvuzo yanga yonse?", sender, phone_id)
        elif lang == "chinyanja":
            send("Kodi muli ndi mafunso ena?", sender, phone_id)
        elif lang == "bemba":
            send("Uli ne fimo fyandi ifyakulya?", sender, phone_id)
        elif lang == "lozi":
            send("O na mabvuzo a mangi?", sender, phone_id)
        else:
            send("Do you have any more questions?", sender, phone_id)
    
        state["step"] = "general_followup"
        save_single_user_state(sender)
        return
    else:
        handle_main_menu(sender, prompt, phone_id)
        

def ask_cervical_more_info(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kuwana rumwe ruzivo rwe cervical cancer? ", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukuthola eminye imininingwane nge-cervical cancer? ", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kupeza zambiri za cervical cancer?", sender, phone_id)
    else:
        send("Would you like to get more information about cervical cancer?  ", sender, phone_id)
    
    state["step"] = "cervical_more_info"
    save_single_user_state(sender)

def ask_cervical_question_number(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Pinda nhamba yemubvunzo kubva pa 1 kusvika pa 100:", sender, phone_id)
    elif lang == "ndebele":
        send("Faka inombolo yombuzo kusuka ku-1 kuya ku-100:", sender, phone_id)
    elif lang == "chinyanja":
        send("Lowetsani nambala ya funso kuchokera pa 1 mpaka 100:", sender, phone_id)
    else:
        send("Enter a question number from 1 to 100:", sender, phone_id)
    
    state["step"] = "cervical_question_number"
    save_single_user_state(sender)

def ask_keep_learning(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kuramba uchidzidza zvimwe zvinhu zve cervical cancer? ", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukuqhubeka nokufunda ezinye izindaba ze-cervical cancer? ", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kupitiriza kuphunzira zina zambiri za cervical cancer?", sender, phone_id)
    else:
        send("Would you like to keep learning more about cervical cancer?  ", sender, phone_id)
    
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
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuwana rumwe ruzivo? ", sender, phone_id)
        elif lang == "ndebele":
            send("Angikuzwisisi. Phendula ngicela: Ungathanda ukuthola eminye imininingwane? ", sender, phone_id)
        elif lang == "chinyanja":
            send("Sindinamve. Yankhani chonde: Kodi mukufuna kupeza zambiri?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to get more information?  ", sender, phone_id)

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
                if lang == "shona":
                    send(f"Ndine urombo, handina kuwana mubvunzo wenhamba {question_num}. Edza imwe nhamba kubva pa 1 kusvika pa 100.", sender, phone_id)
                elif lang == "ndebele":
                    send(f"Uxolo, angikutholanga umbuzo wenombolo {question_num}. Zama enye inombolo kusuka ku-1 kuya ku-100.", sender, phone_id)
                elif lang == "chinyanja":
                    send(f"Pepani, sindinapeze funso la nambala {question_num}. Yesani nambala ina kuchokera pa 1 mpaka 100.", sender, phone_id)
                else:
                    send(f"Sorry, I couldn't find question number {question_num}. Please try another number from 1 to 100.", sender, phone_id)
                ask_cervical_question_number(sender, phone_id)
        else:
            if lang == "shona":
                send("Ndapota pinda nhamba kubva pa 1 kusvika pa 100 chete.", sender, phone_id)
            elif lang == "ndebele":
                send("Sicela ufake inombolo ephakathi kuka-1 no-100 kuphela.", sender, phone_id)
            elif lang == "chinyanja":
                send("Chonde lowetsani nambala kuchokera pa 1 mpaka 100 basi.", sender, phone_id)
            else:
                send("Please enter a number between 1 and 100 only.", sender, phone_id)
            ask_cervical_question_number(sender, phone_id)
            
    except ValueError:
        if lang == "shona":
            send("Ndapota pinda nhamba chaiyo kubva pa 1 kusvika pa 100.", sender, phone_id)
        elif lang == "ndebele":
            send("Sicela ufake inombolo evumelekile ephakathi kuka-1 no-100.", sender, phone_id)
        elif lang == "chinyanja":
            send("Chonde lowetsani nambala yoyenera kuchokera pa 1 mpaka 100.", sender, phone_id)
        else:
            send("Please enter a valid number between 1 and 100.", sender, phone_id)
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
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuramba uchidzidza? ", sender, phone_id)
        elif lang == "ndebele":
            send("Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokufunda? ", sender, phone_id)
        elif lang == "chinyanja":
            send("Sindinamve. Yankhani chonde: Kodi mukufuna kupitiriza kuphunzira?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to keep learning?  ", sender, phone_id)

def ask_another_week(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kudzidza nezve mamwe mavhiki epamuviri? ", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukufunda ngamanye amaviki okukhulelwa? ", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kudziwa za masabata ena a pakati?", sender, phone_id)
    else:
        send("Would you like to learn about other pregnancy weeks?  ", sender, phone_id)
    
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
        if lang == "shona":
            send("Ndapota isa vhiki re pamuviri ", sender, phone_id)
        elif lang == "ndebele":
            send("Sicela ufake iviki lokukhulelwa ", sender, phone_id)
        elif lang == "chinyanja":
            send("Chonde lowetsani sabata la pakati ", sender, phone_id)
        else:
            send("Please enter your pregnancy week number ", sender, phone_id)
        save_single_user_state(sender)
        
    elif any(response in prompt_lower for response in no_responses):
        state["step"] = "product_inquiry"
        state["topic"] = "maternal"
        
        if lang == "shona":
            send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri? Tinopa:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu? Sinakho:\n- Ama-Prenatal Vitamins\n- Izinto zokuhlola isisu\n- Amakhithi okunakekela isisu", sender, phone_id)
        elif lang == "chinyanja":
            send("Zikomo! Kodi mukufuna kugula zinthu za Thanzi la Amayi? Tili ndi:\n- Mavitamini a Prenatal\n- Zoyezera pakati\n- Makiti a Thanzi la Amayi", sender, phone_id)
        else:
            send("Thank you! Would you like to purchase maternal health products? We offer:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits", sender, phone_id)
        save_single_user_state(sender)
        
    else:
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kudzidza nezve mamwe mavhiki? ", sender, phone_id)
        elif lang == "ndebele":
            send("Angikuzwisisi. Phendula ngicela: Ungathanda ukufunda ngamanye amaviki? ", sender, phone_id)
        elif lang == "chinyanja":
            send("Sindinamve. Yankhani chonde: Kodi mukufuna kudziwa za masabata ena?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to learn about other weeks?  ", sender, phone_id)


def ask_gemini(question: str, lang: str = "english") -> str:
    try:
        if lang == "shona":
            instruction = (
                "Iwe uri mubatsiri wezvehutano hwepamuviri. "
                "Pindura mubvunzo uyu muShona yakajeka, yakapfava, uye ine ruzivo rwezvehutano:\n\n"
            )
        elif lang == "ndebele":
            instruction = (
                "Ungumsizi wezempilo yesisu. "
                "Phendula lo mbuzo ngesiNdebele esicacile, esilula, futhi enolwazi lwezempilo:\n\n"
            )
        elif lang == "chinyanja":
            instruction = (
                "Ndine mphungu wa Thanzi la Amayi. "
                "Yankhani funso ili m'Chinyanja moyenera, mosavuta, komanso moli ndi umanyambazi wa Thanzi la Amayi:\n\n"
            )
        else:
            instruction = (
                "You are a maternal health assistant. "
                "Answer the following question clearly, simply, and with accurate health information:\n\n"
            )

        response = convo.send_message(instruction + question)

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return (
                "Ndine urombo, handina kuwana mhinduro." if lang == "shona"
                else "Uxolo, angikutholanga impendulo." if lang == "ndebele"
                else "Pepani, sindinapeze yankho." if lang == "chinyanja"
                else "Sorry, I couldn't find an answer."
            )

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Kunenkinga ekuphenduleni umbuzo wakho." if lang == "ndebele"
            else "Pali vuto popanga yankho la funso lanu." if lang == "chinyanja"
            else "Sorry, there was a problem getting an answer."
        )


def ask_gemini_cancer(question: str, lang: str = "english") -> str:
    try:
        if lang == "shona":
            instruction = (
                "Iwe uri mubatsiri wezvehutano hwegomarara rechibereko. "
                "Pindura mubvunzo uyu muShona yakajeka uye yakapfava:\n\n"
            )
        elif lang == "ndebele":
            instruction = (
                "Ungumsizi wezempilo yomhlaza wesibeletho. "
                "Phendula lo mbuzo ngesiNdebele esicacile futhi esilula:\n\n"
            )
        elif lang == "chinyanja":
            instruction = (
                "Ndine mphungu wa thanzi la kansa ya chibereko. "
                "Yankhani funso ili mu Chinyanja momveka bwino komanso mwaulemu:\n\n"
            )
        else:
            instruction = (
                "You are a cervical cancer health assistant. "
                "Answer the following question clearly and simply in English:\n\n"
            )

        response = convo.send_message(instruction + question)

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return (
                "Ndine hurombo, handina kuwana mhinduro." if lang == "shona"
                else "Uxolo, angikutholanga impendulo." if lang == "ndebele"
                else "Pepani, sindinapeze yankho." if lang == "chinyanja"
                else "Sorry, I couldn't find an answer."
            )

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Kunenkinga ekuphenduleni umbuzo wakho." if lang == "ndebele"
            else "Pali vuto popanga yankho la funso lanu." if lang == "chinyanja"
            else "Sorry, there was a problem getting an answer."
        )


def ask_gemini_general(question: str, lang: str) -> str:
    try:
        company_address = "No. 50 Lunsemfwa Rd, Kalundu, Lusaka, Zambia"
        company_email = "hello@dawa-health.com"
        company_website = "https://dawa-health.com/"
        company_phone = "+260 977 985 063"

        if lang == "shona":
            instruction = (
                "Uri mubatsiri wezvehutano ane hunyanzvi hwakakosha muhutano hwevakadzi vane pamuviri uye gomarara remuromo wechibereko. "
                "Pindura mubvunzo wemushandisi uchishandisa ruzivo rwechokwadi uye rwakavakirwa pauchapupu rwezvehutano. "
                "ZVINOKOSHA: Mhinduro inofanira kuva yakadzama, ine chokwadi, uye yehunyanzvi. "
                "USATANGE nemitsara yakaita sekuti 'Zvakanaka', 'Hongu', 'Hezvino', kana kuti 'Rega nditsanangure'. "
                "USASANGANISA mazwi ekuzadza hurukuro asingakoshi. "
                "Tanga zvakananga nemhinduro. "
                "Pedzisa nekuyambira kupfupi kunoti ruzivo urwu harutsivi kuongororwa nachiremba. "
                "Pindura muChirungu chiri pachena uye chiri nyore:\n\n"
            )
        elif lang == "ndebele":
            instruction = (
                "Ungumsizi wezempilo ochwepheshile ogxile kwezempilo yabomama abakhulelweyo kanye lomdlavuza womlomo wesibeletho. "
                "Phendula umbuzo womsebenzisi usebenzisa ulwazi lwezempilo oluqondileyo futhi olusekelwe ebufakazini. "
                "OKUBALULEKILE: Impendulo kumele ibe eningiliziwe, ibe leqiniso, futhi ibe ngeyobungcweti. "
                "UNGAKALI ngemisho efana lokuthi 'Kulungile', 'Yebo', 'Nakhu', kumbe 'Ake ngichasise'. "
                "UNGAFaki amazwi okugcwalisa ingxoxo angabalulekanga. "
                "Qalisa masinyane ngempendulo uqobo. "
                "Qedisa ngesexwayiso esifitshane esithi ulwazi lolu aluthathi indawo yokuhlolwa ngudokotela. "
                "Phendula ngesiNgisi esicacileyo futhi esilula:\n\n"
            )
        elif lang == "tonga":
            instruction = (
                "Muli mweenzinyina wa buumi uuli mwiinda lyoonse mu buumi bwa banakazi abali mu buumi bwa kubusya mwana alimwi ne ndenda ya mulomo wa cibeleko. "
                "Mupandule mwaambo wa musisi nikukonzya kwa buumi kululeme alimwi kwakavumbululwa kuli buci bwakazibidwe. "
                "CINTU CAKUKOSHA: Mpendulo yenu yeelede kuba ndeepe, yakweene, alimwi ya buumi bwa bucita bwabukombi. "
                "MUSATANGI kwa mazwi nga 'Kulungile', 'Inzya', 'Nciici', naa 'Lekani ndichazye'. "
                "MUSAFUGI mazwi a kujazya kwa ng'anda atali a bulemu. "
                "Tangi mpoonya mpoonyo ku mpendulo. "
                "Malizya a kusinsimuna kufwaafwi kuti ulwazi ulu talusanduki ku lwandano lwa dokotela. "
                "Mupandule mu Chikuwa chakweelela alimwi chiswiipe:\n\n"
            )
        elif lang == "chinyanja":
            instruction = (
                "Ndinu mthandizi wa zaumoyo wa akatswiri okhazikika pa zaumoyo wa amayi apakati komanso khansa ya khomo la chiberekero. "
                "Yankhani funso la wogwiritsa ntchito pogwiritsa ntchito chidziwitso cholondola komanso chochokera ku umboni wa sayansi ya zaumoyo. "
                "CHOFUNIKA: Yankho liyenera kukhala latsatanetsatane, lolondola, komanso laukatswiri. "
                "MUSAYAMBE ndi mawu ngati 'Chabwino', 'Inde', 'Nazi', kapena 'Ndisiyeni ndifotokoze'. "
                "MUSAPHATIKIZE mawu odzaza zokambirana osafunika. "
                "Yambani mwachindunji ndi yankho. "
                "Malizitsani ndi chenjezo chachidule chonena kuti chidziwitsochi sichimalowa m'malo mwa kuyezetsa kwa dokotala. "
                "Yankhani mu Chingerezi chomveka bwino komanso chosavuta:\n\n"
            )
        elif lang == "bemba":
            instruction = (
                "Uli kapyunga wa buumi uwakwata ubuchindami uwashintilila pa buumi bwa banakashi abali ne fumo pamo ne kansa ya ku mulomo wa cibeleshi. "
                "Yasuka ilipusho lya muntu uulefwaya amashiwi ukoresha ubusuma bwa buumi ubwalungama kabili ubwashintilila pa bucapo. "
                "ICAKOSA: Ilyasuko lifwile ukuba ilyapwililika, ilyalungama, kabili ilya bucapo. "
                "WILATAMBA ne mashiwi nga 'Cisuma', 'Ee', 'Ici cili', nangu 'Lekeni nsoshe'. "
                "WILAFWIKAKO amashiwi ayakufwailisha ayashili ayafunika. "
                "Tambilila ku lyasuko mwachindunji. "
                "Pwishisheni ne cilembelo cipepa icilelanda ati ubu busuma tabusendapo icifulo ca kuyeshiwa kuli dokota. "
                "Yasukeni mu Chingeleshi icamoneka bwino kabili icapepuka:\n\n"
            )
        elif lang == "lozi":
            instruction = (
                "Mu muthusi wa za mapilo wa bucwani ya iketile hahulu ku mapilo a basali baimana ni kankere ya mulomo wa sibeleko. "
                "Alaba lipuzo la musebelisi mu kusebelisa ziboho za mapilo ze nepahezi ni ze zishimbilwe ku bupaki bwa sayansi. "
                "SA BUTOKWA: Karabo i fanele kuba ya bunyinyani, ya niti, ni ya bucwani. "
                "MU SA QALISE ka mafoko a swana ni 'Ho lukile', 'Eeni', 'Se', kamba 'Ni ka talusa'. "
                "MU SA KENYE mafoko a ku tlatsa puiso a sa tokwi. "
                "Qalisa hanghang ka karabo. "
                "Felelisa ka temoso ye nyinyani ye e re ziboho ze ha zi nkeleli sibaka sa ku lekolwa ki dokota. "
                "Arabela ka Sizingelesi se si bonahala hande ni se si nolofetse:\n\n"
            )
        else:
            instruction = (                
                "You are a professional health assistant specializing in maternal health and cervical cancer for Dawa Health. "
                "Answer the user's question using correct and evidence-based health information. "
                "IMPORTANT: The response must be detailed, factual, and professional. "
                "DO NOT start with phrases like 'Okay', 'Sure', 'Here's', or 'Let me explain'. "
                "DO NOT include any conversational fillers. "
                "Start directly with the answer. "
                "Include a brief disclaimer at the end stating that this information does not replace a doctor's evaluation."
                "If a user asks questions like do you do home visits they are referring to Dawa Health and yes, Dawa Health clinicians do home visits."
                "If a user asks for a service or say that they want one you check for pricing in {products} "
                "Instructions in {instructions} are to be strictly followed."
                f"When a user asks about contact details, use company_email={company_email}, company_phone={company_phone}, company_address={company_address} and company_website={company_website}"
                "Respond in clear, simple English:\n\n"
            )

        model = genai.GenerativeModel("gemini-2.5-flash") 
        response = model.generate_content(instruction + question)

        if response and response.text:
            return response.text.strip()
        else:
            return "Ndine urombo, handina kuwana mhinduro." if lang == "shona" else "Sorry, I couldn't find an answer."

    except Exception as e:
        print(f"[Gemini General Error] {e}")
        logging.error(f"[Gemini General Error] {type(e).__name__}: {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Sorry, there was an error answering your question."
        )


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
                            section = info_text[start_idx:end_idx]
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
                    week_info = match.group(0).strip()
                    send(f"Zambiri za *Sabata {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Pepani, sindinapeze zambiri za sabata {week_num}. Yesani sabata lina kuchokera pa 1 mpaka 40.", sender, phone_id)
            
            elif lang == "shona":
                pattern = rf"\*Vhiki {week_num}:.*?(?=\*Vhiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    week_info = match.group(0).strip()
                    send(f"Ruzivo rwe *Vhiki {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndine urombo, handina kuwana ruzivo rwevhiki {week_num}. Edza imwe vhiki kubva pa 1 kusvika pa 40.", sender, phone_id)

            elif lang == "lozi":
                pattern = rf"\*Sunda {week_num}:.*?(?=\*Sunda {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    week_info = match.group(0).strip()
                    send(f"Liziba la *Sunda {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ni maswabi, ha ni a fumana liziba la vhiki {week_num}. Linge sunda ye n'wi ku zwana 1 ku ya ku 40.", sender, phone_id)

            elif lang == "bemba":
                pattern = rf"\*Umulungu {week_num}:.*?(?=\*Umulungu {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    week_info = match.group(0).strip()
                    send(f"Icibeela ca *Mulungu {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Natapa, nshasangile icibeela ca mulungu {week_num}. Esheni mulungu ubi ku 1 ukufika ku 40.", sender, phone_id)

            elif lang == "tonga":
                pattern = rf"\*Nhwiiiki {week_num}:.*?(?=\*Nhwiiiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    week_info = match.group(0).strip()
                    send(f"Cibeela ca *Nhwiiiki {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndatola, tana kuwana cibeela ca nhwiiiki {week_num}. Lingenya vhiki linzwi kuzwa 1 kusika 40.", sender, phone_id)
            
            else:
                pattern = rf"\*Week {week_num}:.*?(?=\*Week {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    week_info = match.group(0).strip()
                    send(f"Here's information for *Week {week_num}:*\n\n{week_info}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Sorry, I couldn't find information for week {week_num}. Please try another week from 1 to 40.", sender, phone_id)
        
        else:
            if lang == "shona":
                send("Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.", sender, phone_id)
            elif lang == "ndebele":
                send("Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.", sender, phone_id)
            elif lang == "bemba":
                send("Napapita, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.", sender, phone_id)
            elif lang == "chinyanja":
                send("Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.", sender, phone_id)
            elif lang == "tonga":
                send("Ndatola, ingila vhiki kuzwa 1 kusika 40 pe.", sender, phone_id)
            elif lang == "lozi":
                send("Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.", sender, phone_id)
            else:
                send("Please enter a week between 1 and 40 only.", sender, phone_id)
            
    except ValueError:
        if lang == "shona":
            send("Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.", sender, phone_id)
        elif lang == "ndebele":
            send("Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.", sender, phone_id)
        elif lang == "bemba":
            send("Napapita, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.", sender, phone_id)
        elif lang == "chinyanja":
            send("Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.", sender, phone_id)
        elif lang == "tonga":
            send("Ndatola, ingila vhiki kuzwa 1 kusika 40 pe.", sender, phone_id)
        elif lang == "lozi":
            send("Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.", sender, phone_id)
        else:
            send("Please enter a week between 1 and 40 only.", sender, phone_id)
            

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
                                            
                                            # ─── KEY FIX: per-user state loading ───
                                            is_new = ensure_user_state(sender)
                                            if not is_new:
                                                user_states[sender]["first_message"] = False
                                            # ────────────────────────────────────────
                                            
                                            save_user_conversation(sender, "user", prompt)
                                            handle_conversation_state(sender, prompt, phone_id)
                                            
                                        else:
                                            logging.info(f"Non-text message received from {sender}")
                                            # Ensure state exists for non-text messages too
                                            ensure_user_state(sender)
                                            state = user_states.get(sender, {})
                                            lang = state.get("language", "english")
                                            if lang == "shona":
                                                send("Ndine urombo, handigoni kugamuchira mameseji asiri mavara chete. Ndapota tumira meseji yemavara.", sender, phone_id)
                                            elif lang == "ndebele":
                                                send("Uxolo, angikwazi ukwamukela imilayezo engeyona imibhalo kuphela. Sicela uthumele umlayezo wombhalo.", sender, phone_id)
                                            elif lang == "bemba":
                                                send("Natapa, nshakwanishe ukupokeela amameseji yambi ukucila pa menso. Napapita, tuma ubutumwa bwamenso.", sender, phone_id)
                                            elif lang == "chinyanja":
                                                send("Pepani, sindingathe kulandira mameseji enama osati a zilembo. Chonde tumirani meseji ya zilembo.", sender, phone_id)
                                            elif lang == "tonga":
                                                send("Ndazwa kwiinda, tani konzy kujana mameseji aambi kusikwa aa mabbala. Ndatola, tuma meseji ya mabbala.", sender, phone_id)
                                            elif lang == "lozi":
                                                send("Ni maswabi, ha na kona kuzwela miiala yeng'wi kufita feela ya mangolo. Ndapota, lumeza molaala wa mangolo.", sender, phone_id)
                                            else:
                                                send("Sorry, I can only process text messages. Please send a text message.", sender, phone_id)
                                
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
