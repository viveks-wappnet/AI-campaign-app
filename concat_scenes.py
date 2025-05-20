from graph.nodes.audio_generator_node import concatenate_scenes
import glob
import os

def main():
    # Get all sub-scene videos in order
    scene_files = sorted(glob.glob("scenes/scene*_sub*_av.mp4"))
    
    if not scene_files:
        print("No sub-scene videos found in scenes directory!")
        return
    
    print(f"Found {len(scene_files)} sub-scene videos:")
    for f in scene_files:
        print(f"  - {os.path.basename(f)}")
    
    # Concatenate them into final video
    final_path = concatenate_scenes(scene_files)
    print(f"\nFinal video created at: {final_path}")

if __name__ == "__main__":
    main() 