import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fyers_apiv3 import fyersModel
import logging
from config import GEMINI_API_KEY, GEMINI_MODEL, CLIENT_ID

logger = logging.getLogger(__name__)

# Initialize Gemini AI
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        logger.info(f"Gemini AI initialized with model: {GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini AI: {e}")
        gemini_model = None
else:
    logger.warning("GEMINI_API_KEY not found. AI features will be disabled.")

# Initialize FyersModel
fyers_instance = None
