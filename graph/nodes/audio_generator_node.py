import os
import subprocess
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

def _get_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(cp.stdout)["format"]["duration"])

def _render_tts(text: str, out_path: str) -> None:
    """Stream ElevenLabs TTS into a file."""
    # gen = client.text_to_speech.convert(
    #     text=text,
    #     voice_id=VOICE_ID,
    #     model_id="eleven_multilingual_v2",
    #     output_format="mp3_44100_128",
    #     voice_settings={"speed": 0.9, "stability": 0.35, "similarity_boost": 0.75}
    # )
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # with open(out_path, "wb") as f:
    #     for chunk in gen:
    #         f.write(chunk)
    pass

def _mux_audio_video(video_in: str, audio_in: str, video_out: str):
    """
    Merge video_in and audio_in into video_out. 
    Audio is re-encoded to AAC, video is stream-copied.
    """
    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-i", audio_in,
        "-c:v", "copy",
        "-c:a", "aac",
        # Map streams explicitly (0:v = video, 1:a = audio)
        "-map", "0:v",
        "-map", "1:a",
        # *No* -shortest: lets output match the video duration
        video_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_audio_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scenes: List[Dict[str, Any]] = state["script"]["scenes"]
    output_scenes = []

    for s in scenes:
        sid = s["scene_id"]
        # Paths
        audio_path = f"audio/scene_{sid}.mp3"
        final_video = f"scenes/scene_{sid}_av.mp4"
        raw_video   = s.get("scene_video_path")
        
        # 1) Render TTS
        # _render_tts(s["dialogue"], audio_path)
        
        # 2) Measure audio length
        audio_dur = _get_duration(audio_path)
        
        # 3) Mux into video
        if raw_video and os.path.isfile(raw_video):
            _mux_audio_video(raw_video, audio_path, final_video)
        else:
            final_video = None
        
        # 4) Collect
        output = {
            **s,
            "audio_path": audio_path,
            "audio_duration_s": audio_dur,
            "final_video_path": final_video
        }
        output_scenes.append(output)

    return {"scenes": output_scenes}
