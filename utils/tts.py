import os
import re
import logging
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

def convert_pause_markers_to_ssml(text: str) -> str:
    """Convert [PAUSE:X.Xs] markers to SSML break tags."""
    def replace_pause(match):
        duration = match.group(1)
        return f'<break time="{duration}s"/>'
    
    return re.sub(r'\[PAUSE:(\d+\.?\d*)s\]', replace_pause, text)

def render_tts(text: str, out_path: str) -> None:
    """Generate TTS audio using ElevenLabs API."""
    logger.info(f"Generating TTS for text: {text[:40]}...")
    try:
        # Convert pause markers to SSML
        ssml_text = convert_pause_markers_to_ssml(text)
        
        gen = client.text_to_speech.convert(
            text=ssml_text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={"speed": 1.0, "stability": 0.35, "similarity_boost": 0.75}
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in gen:
                f.write(chunk)
        logger.info(f"TTS audio saved to {out_path}")
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        raise 