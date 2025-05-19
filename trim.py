import os
import ffmpeg
import requests
from module.script import generate_ad_script
from module.video_finder import VideoFinderAgent

class VideoAssembler:
    def __init__(self, output_dir: str = "outputs/final", temp_dir: str = "outputs/temp"):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.script = []
        self.clips = {}

    def load_script_and_clips(self, script, clips):
        self.script = script
        self.clips = clips

    def _download_video_if_needed(self, url: str, scene_id: int) -> str:
        filename = os.path.join(self.temp_dir, f"scene_{scene_id}_raw.mp4")
        if not os.path.exists(filename):
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename

    def trim_clips(self):
        for scene in self.script:
            scene_id = scene['scene']
            duration_s = int(scene['duration'].replace('s', ''))
            clip_info = self.clips.get(scene_id)
            if not clip_info:
                print(f"⚠️ No clip found for scene {scene_id}")
                continue

            input_path = self._download_video_if_needed(clip_info['video_file_url'], scene_id)
            output_path = os.path.join(self.temp_dir, f"scene_{scene_id}_trimmed.mp4")

            try:
                (
                    ffmpeg
                    .input(input_path, ss=0, t=duration_s)
                    .output(output_path, codec='copy')
                    .run(quiet=True, overwrite_output=True)
                )
                print(f"✅ Trimmed scene {scene_id} to {duration_s}s → {output_path}")
            except ffmpeg.Error as e:
                print(f"❌ Error trimming scene {scene_id}: {e}")

if __name__ == '__main__':
    assembler = VideoAssembler()
    agent = VideoFinderAgent(api_key=os.getenv('PIXABAY_API_KEY'), per_page=50)
    prompt = "Promote a short Europe travel spot highlighting no-quarantine rules."
    script = generate_ad_script(prompt)
    print(f"The generated script: {script}")
    clips = agent.process_script(script)
    assembler.load_script_and_clips(script, clips)

    assembler.trim_clips()