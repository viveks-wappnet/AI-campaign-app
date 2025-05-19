import ipdb
import os
import subprocess
import requests
from typing import Dict, Any
from tempfile import TemporaryDirectory

def download_video(url: str, dest: str) -> None:
    """Download a video from a URL to the given path."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

def stitch_scene_subclips(scene: Dict[str, Any]) -> str:
    """
    Fast FFmpeg-based stitching: trims each sub-scene via -ss/-t (copy codec),
    scales all videos to 1080p with consistent parameters for future audio addition.
    Returns the path to the final scene file.
    """
    scene_id = scene["scene_id"]
    with TemporaryDirectory() as tmpdir:
        # First, download and trim all clips
        trimmed_clips = []
        for sub in scene["sub_scenes"]:
            sub_id = sub["sub_id"]
            duration = int(sub["duration"].rstrip("s"))
            inp_path = os.path.join(tmpdir, f"in_{sub_id}.mp4")
            out_path = os.path.join(tmpdir, f"trim_{sub_id}.mp4")
            scaled_path = os.path.join(tmpdir, f"scaled_{sub_id}.mp4")

            # 1) download
            download_video(sub["video_url"], inp_path)

            # 2) trim video only (no audio)
            cmd_trim = [
                "ffmpeg", "-y",
                "-ss", "0",
                "-i", inp_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-an",  # Remove audio
                "-preset", "ultrafast",
                out_path
            ]
            subprocess.run(cmd_trim, check=True)

            # 3) scale and normalize to consistent parameters
            cmd_scale = [
                "ffmpeg", "-y",
                "-i", out_path,
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-r", "30",  # Force 30fps
                "-b:v", "5M",  # Consistent bitrate
                "-maxrate", "5M",
                "-bufsize", "10M",
                "-pix_fmt", "yuv420p",  # Standard pixel format
                "-g", "30",  # Keyframe every second (at 30fps)
                "-keyint_min", "30",
                "-sc_threshold", "0",
                "-profile:v", "high",  # High profile for better quality
                "-level", "4.0",  # Compatibility level
                scaled_path
            ]
            subprocess.run(cmd_scale, check=True)

            # Verify the trimmed duration
            cmd_probe = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                scaled_path
            ]
            actual_duration = float(subprocess.check_output(cmd_probe).decode().strip())
            print(f"Sub-scene {sub_id}: Expected {duration}s, Got {actual_duration}s")
            if abs(actual_duration - duration) > 0.5:
                print(f"Warning: Sub-scene {sub_id} has duration {actual_duration}s instead of expected {duration}s")
            
            trimmed_clips.append(scaled_path)

        final_path = f"scenes/scene_{scene_id}.mp4"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        # Build filter_complex string for video-only concatenation
        filter_complex = []
        inputs = []
        for i, clip in enumerate(trimmed_clips):
            inputs.extend(["-i", clip])
            filter_complex.append(f"[{i}:v]")
        
        # Simple video-only concatenation
        filter_str = "".join(filter_complex) + f"concat=n={len(trimmed_clips)}:v=1[outv]"

        # Concatenate using filter_complex with consistent output parameters
        cmd_concat = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-b:v", "5M",  # Consistent bitrate
            "-maxrate", "5M",
            "-bufsize", "10M",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-g", "30",
            "-keyint_min", "30",
            "-sc_threshold", "0",
            "-profile:v", "high",
            "-level", "4.0",
            "-movflags", "+faststart",
            final_path
        ]
        
        print(f"Running concatenation command for scene {scene_id}")
        try:
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error details for scene {scene_id}:")
            print(f"Command: {' '.join(cmd_concat)}")
            print(f"Error output: {e.stderr}")
            raise

        # Verify final duration and parameters
        cmd_probe_final = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            final_path
        ]
        final_duration = float(subprocess.check_output(cmd_probe_final).decode().strip())
        expected_duration = sum(int(sub["duration"].rstrip("s")) for sub in scene["sub_scenes"])
        print(f"Scene {scene_id} final duration: Expected {expected_duration}s, Got {final_duration}s")
        if abs(final_duration - expected_duration) > 0.5:
            print(f"Warning: Final scene {scene_id} has duration {final_duration}s instead of expected {expected_duration}s")

    return final_path

def patch_video_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: stitches sub-scenes into full scenes using fast FFmpeg.
    After creating the video, removes the sub_scenes data as it's no longer needed.
    """
    # ipdb.set_trace()
    for scene in state["script"]["scenes"]:
        try:
            scene["scene_video_path"] = stitch_scene_subclips(scene)
            # Remove sub_scenes after video is created
            if "sub_scenes" in scene:
                del scene["sub_scenes"]
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error for scene {scene['scene_id']}: {e}")
            scene["scene_video_path"] = None
    return {"script": state["script"]}
