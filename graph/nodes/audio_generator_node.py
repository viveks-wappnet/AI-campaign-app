import os, subprocess, json
from typing import Any, Dict, List
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

def _get_duration(path: str) -> float:
    """ffprobe → duration in seconds."""
    cmd = [
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","json", path
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(cp.stdout)["format"]["duration"])

def _render_tts(text: str, out_path: str) -> None:
    """Stream ElevenLabs TTS into a file."""
    gen = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings={"speed": 0.9, "stability":0.35, "similarity_boost":0.75}
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in gen:
            f.write(chunk)

def generate_audio_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scenes: List[Dict[str, Any]] = state["script"]["scenes"]
    output_scenes = []
    for s in scenes:
        sid = s["scene_id"]
        path = f"audio/scene_{sid}.mp3"
        _render_tts(s["dialogue"], path)
        dur = _get_duration(path)
        output_scenes.append({
            **s,  # carry forward scene_id, video_url, etc.
            "audio_path": path,
            "audio_duration_s": dur,
        })
    return {"scenes": output_scenes}