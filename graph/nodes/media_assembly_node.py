import os
import logging
from typing import Any, Dict, List
from tempfile import TemporaryDirectory
from tqdm import tqdm

from utils.tts import render_tts
from utils.download import download_file
from utils.media import trim_and_mux, concatenate_videos 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_audio_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process scenes to generate audio and combine with video.
    Returns updated state with video paths.
    """
    scenes: List[Dict[str, Any]] = state["script"]["scenes"]
    logger.info(f"Processing {len(scenes)} scenes")
    
    os.makedirs("scenes", exist_ok=True)

    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_id = scene["scene_id"]
        sub_paths = []
        logger.info(f"Processing scene {scene_id}")

        with TemporaryDirectory() as tmp:
            for sub in tqdm(scene["sub_scenes"], desc=f"Scene {scene_id} sub-scenes", leave=False):
                sid = sub["sub_id"]
                logger.info(f"Processing sub-scene {sid}")

                audio_path = os.path.join(tmp, f"scene{scene_id}_sub{sid}.mp3")
                raw_vid = os.path.join(tmp, f"scene{scene_id}_sub{sid}.mp4")
                final_sub = os.path.join("scenes", f"scene{scene_id}_sub{sid}_av.mp4")

                # Generate audio and combine with video using utility functions
                render_tts(sub["dialogue"], audio_path)
                download_file(sub["video_url"], raw_vid)
                trim_and_mux(raw_vid, audio_path, final_sub)

                sub_paths.append(final_sub)

        # Create scene video using concatenate_videos utility
        scene_out = os.path.join("scenes", f"scene_{scene_id}.mp4")
        logger.info(f"Creating scene video: {scene_out}")
        scene_out = concatenate_videos(sub_paths, scene_out)
        scene["scene_video_path"] = scene_out

    # Create final video from all scenes
    logger.info("Creating final video from all scenes")
    scene_paths = [scene["scene_video_path"] for scene in scenes]
    final_video = concatenate_videos(scene_paths, "scenes/final_video.mp4")

    logger.info("Video generation complete")
    return {
        "script": {"scenes": scenes},
        "final_video_path": final_video
    }