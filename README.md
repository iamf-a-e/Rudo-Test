# Rudo

Rudo is a Flask WhatsApp chatbot for Dawa Health. It supports maternal health, cervical cancer information, product inquiries, registration, and multilingual responses.

## Requirements

- Python 3.11 or newer
- WhatsApp Cloud API credentials
- Gemini API key
- Optional: Upstash Redis for persistent per-user state

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set environment variables:

```powershell
$env:WA_TOKEN="your-whatsapp-token"
$env:PHONE_ID="your-phone-number-id"
$env:GEN_API="your-gemini-api-key"
$env:UPSTASH_REDIS_URL="your-upstash-url"
$env:UPSTASH_REDIS_TOKEN="your-upstash-token"
```

`UPSTASH_REDIS_URL` and `UPSTASH_REDIS_TOKEN` are optional. Without them, state is kept only in memory for the current process.

## Run Locally

```powershell
python main.py
```

The health/status page is available at:

```text
http://127.0.0.1:5000/
```

Test WhatsApp webhook verification:

```powershell
Invoke-WebRequest "http://127.0.0.1:5000/webhook?hub.mode=subscribe&hub.verify_token=BOT&hub.challenge=12345"
```

Expected response:

```text
12345
```

## Testing Before Deploy

Run syntax and diff checks:

```powershell
python -m py_compile main.py
git diff --check
```

For live WhatsApp webhook testing, expose the local Flask server using a tunnel such as ngrok, then configure the tunnel URL in the WhatsApp Cloud API webhook settings.

## Important Notes

- Tonga cervical cancer static data is not currently available. The app falls back to English cervical cancer data for Tonga users instead of crashing.
- Product data is canonical in `products_data.py`. The `training/products.py` and `training/products_data.py` modules import from that source for backward compatibility.
- Medical translations, especially Tonga cervical cancer content, should be reviewed by a native Chitonga speaker with health context before production use.

## Deploy

The project includes `vercel.json` for Vercel deployment. Configure the same environment variables in Vercel before deploying.
