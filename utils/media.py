import os
import json
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_duration(path: str) -> float:
    """Get the duration of a media file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(cp.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error getting duration for {path}: {str(e)}")
        raise

def trim_and_mux(video_in: str, audio_in: str, out_path: str) -> None:
    """Trim video to match audio duration and mux with audio."""
    try:
        aud_dur = get_duration(audio_in)
        logger.info(f"Audio duration: {aud_dur:.2f}s")

        trimmed = video_in.replace(".mp4", "_trim.mp4")
        cmd_trim = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-ss", "0",
            "-t", str(aud_dur),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            trimmed
        ]
        subprocess.run(cmd_trim, check=True, capture_output=True, text=True)
        logger.info(f"Video trimmed to {aud_dur:.2f}s")

        cmd_mux = [
            "ffmpeg", "-y",
            "-i", trimmed,
            "-i", audio_in,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v",
            "-map", "1:a",
            "-shortest",
            out_path
        ]
        subprocess.run(cmd_mux, check=True, capture_output=True, text=True)
        logger.info(f"Video and audio muxed to {out_path}")

        try:
            os.unlink(trimmed)
            logger.debug(f"Cleaned up temporary file: {trimmed}")
        except OSError as e:
            logger.warning(f"Could not delete temporary file {trimmed}: {str(e)}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error in trim_and_mux: {str(e)}")
        raise

def concatenate_videos(video_paths: List[str], output_path: str) -> str:
    """
    Concatenate videos using filter_complex for better synchronization and quality.
    Normalizes all videos to consistent parameters before concatenation.
    Handles both video and audio streams.
    """
    logger.info("Starting video concatenation process")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        # Normalize videos
        normalized_clips = []
        for i, clip in enumerate(video_paths):
            normalized_path = os.path.join(tmpdir, f"normalized_{i}.mp4")
            logger.info(f"Normalizing clip {i+1}/{len(video_paths)}: {os.path.basename(clip)}")
            
            cmd_normalize = [
                "ffmpeg", "-y",
                "-i", clip,
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-preset", "ultrafast",
                "-r", "30",
                "-b:v", "5M",
                "-maxrate", "5M",
                "-bufsize", "10M",
                "-pix_fmt", "yuv420p",
                "-g", "30",
                "-keyint_min", "30",
                "-sc_threshold", "0",
                "-profile:v", "high",
                "-level", "4.0",
                normalized_path
            ]
            try:
                subprocess.run(cmd_normalize, check=True, capture_output=True, text=True)
                normalized_clips.append(normalized_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error normalizing {clip}: {e.stderr}")
                raise

        # Build concatenation filter
        filter_complex = []
        inputs = []
        for i, clip in enumerate(normalized_clips):
            inputs.extend(["-i", clip])
            filter_complex.append(f"[{i}:v][{i}:a]")
        
        filter_str = "".join(filter_complex) + f"concat=n={len(normalized_clips)}:v=1:a=1[outv][outa]"

        # Concatenate videos
        logger.info(f"Concatenating {len(normalized_clips)} clips into final video")
        cmd_concat = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-preset", "ultrafast",
            "-b:v", "5M",
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
            output_path
        ]
        
        try:
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Concatenation error: {e.stderr}")
            raise

        # Verify output
        cmd_probe = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json",
            output_path
        ]
        probe_output = json.loads(subprocess.check_output(cmd_probe).decode())
        final_duration = float(probe_output["format"]["duration"])
        stream_types = [stream["codec_type"] for stream in probe_output["streams"]]
        
        logger.info(f"Final video created: {output_path}")
        logger.info(f"Duration: {final_duration:.2f}s")
        logger.info(f"Streams: {', '.join(stream_types)}")
        
        return output_path 