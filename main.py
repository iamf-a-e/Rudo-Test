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
from training import products, instructions, pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba, pregnancy_data_lozi

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

