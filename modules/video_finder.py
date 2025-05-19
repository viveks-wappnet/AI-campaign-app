import os
import requests
import re
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv
from groq import Groq
from utils.prompt import get_video_finder_prompts

class PixabayVideoFinder:
    """
    An LLM-driven agent for finding the most appropriate video clips from Pixabay
    based on natural language scene descriptions.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PixabayVideoFinder with your API key.

        Args:
            api_key: Your Pixabay API key. If None, will try to load from environment variables.
        """
        # Load API key from environment if not provided
        load_dotenv()
        self.api_key = api_key or os.environ.get("PIXABAY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "No Pixabay API key provided. Either pass it directly or set the PIXABAY_API_KEY environment variable."
            )

        self.base_url = "https://pixabay.com/api/videos/"

    def _generate_search_terms(self, scene_description: str, llm_client: Groq) -> List[str]:
        """
        Generate relevant search terms from a scene description using an LLM.

        Args:
            scene_description: Natural language description of the video needed
            llm_client: Groq LLM client

        Returns:
            List of search terms to try
        """
        # Get the prompt from the utils module
        prompts = get_video_finder_prompts(scene_description)
        prompt = prompts["search_terms_prompt"]

        try:
            # Use Groq's LLaMA-3 model for efficient keyword generation
            response = llm_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # Lower temperature for more focused results
                max_tokens=100
            )
            search_terms = response.choices[0].message.content.strip().split('\n')

            return [term.strip() for term in search_terms if term.strip()]
        except Exception as e:
            print(f"Error generating search terms: {e}")
            # Fallback to basic keyword extraction
            return [scene_description]

    def search_videos(self, query: str, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """
        Search for videos on Pixabay with the given query.

        Args:
            query: Search term
            page: Page number for pagination
            per_page: Number of results per page

        Returns:
            Dictionary containing search results
        """
        try:
            response = requests.get(
                self.base_url,
                params={
                    "key": self.api_key,
                    "q": query,
                    "page": page,
                    "per_page": per_page
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error searching Pixabay: {e}")
            return {"total": 0, "totalHits": 0, "hits": []}

    def _rank_videos(self, videos: List[Dict[str, Any]], scene_description: str, llm_client: Groq) -> List[Dict[str, Any]]:
        """
        Find the single best video match for the scene description using an LLM.

        Args:
            videos: List of video results from Pixabay
            scene_description: Original scene description
            llm_client: Groq LLM client

        Returns:
            List containing only the best matching video
        """
        if not videos:
            return []

        # Prepare video information for ranking
        video_info = []
        for i, video in enumerate(videos[:10]):  # Limit to top 10 to avoid token limits
            tags = video.get("tags", "")
            duration = video.get("duration", 0)
            resolution = f"{video.get('videos', {}).get('large', {}).get('width', 0)}x{video.get('videos', {}).get('large', {}).get('height', 0)}"
            video_info.append(f"Video {i+1}: Tags: {tags}, Duration: {duration}s, Resolution: {resolution}, Views: {video.get('views', 0)}")

        video_info_text = "\n".join(video_info)

        # Get the prompt from the utils module
        prompts = get_video_finder_prompts(scene_description, video_info_text)
        prompt = prompts["rank_videos_prompt"]

        try:
            # Use Mixtral for better reasoning capabilities when selecting the best video
            response = llm_client.chat.completions.create(
                model="mistral-saba-24b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for more deterministic selection
                max_tokens=10
            )
            selection_text = response.choices[0].message.content.strip()

            # Parse selection
            try:
                # Extract just the digits from the response
                digits = re.findall(r'\d+', selection_text)
                if digits:
                    selected_idx = int(digits[0]) - 1
                    if 0 <= selected_idx < len(videos):
                        return [videos[selected_idx]]

                # If parsing fails or index is invalid, return the first video
                return [videos[0]]
            except Exception as e:
                print(f"Error parsing selection: {e}")
                return [videos[0]]
        except Exception as e:
            print(f"Error selecting best video: {e}")
            return [videos[0]]

    def find_videos(self, scene_description: str, llm_client: Groq) -> List[Dict[str, Any]]:
        """
        Find the single most appropriate video for a scene using LLM-generated search terms.

        Args:
            scene_description: Natural language description of the scene
            llm_client: Groq LLM client

        Returns:
            List containing only the most relevant video
        """
        search_terms = self._generate_search_terms(scene_description, llm_client)
        print(f"Generated search terms: {search_terms}")

        all_videos = []
        for term in search_terms:
            results = self.search_videos(term)
            if results.get("hits"):
                all_videos.extend(results["hits"])

        if not all_videos:
            print("No videos found for the given search terms.")
            return []

        # Remove duplicates by video ID
        unique_videos = {}
        for video in all_videos:
            if video["id"] not in unique_videos:
                unique_videos[video["id"]] = video

        # Find the best video using LLM
        best_video = self._rank_videos(list(unique_videos.values()), scene_description, llm_client)

        # Return only the best match
        return best_video


def main():
    """Command line interface for the PixabayVideoFinder."""
    parser = argparse.ArgumentParser(description='Find the perfect video on Pixabay for a scene description')
    parser.add_argument('scene', help='Natural language description of the scene')

    args = parser.parse_args()

    # Load API keys from environment
    load_dotenv()
    pixabay_api_key = os.environ.get("PIXABAY_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not pixabay_api_key:
        print("Error: Pixabay API key is required. Set the PIXABAY_API_KEY environment variable.")
        return

    if not groq_api_key:
        print("Error: Groq API key is required. Set the GROQ_API_KEY environment variable.")
        return

    # Initialize Groq client
    llm_client = Groq(api_key=groq_api_key)

    try:
        finder = PixabayVideoFinder()
        videos = finder.find_videos(args.scene, llm_client)

        if not videos:
            print(f"\nNo videos found for scene: '{args.scene}'\n")
            return

        video = videos[0]  # We only have one video now

        print(f"\nFound the perfect video for scene: '{args.scene}'\n")
        print("Video details:")
        print(f"  ID: {video.get('id')}")
        print(f"  Tags: {video.get('tags')}")
        print(f"  Duration: {video.get('duration')} seconds")

        # Get the highest quality video URL
        video_urls = video.get("videos", {})
        best_url = None
        for quality in ["large", "medium", "small", "tiny"]:
            if quality in video_urls and video_urls[quality].get("url"):
                best_url = video_urls[quality]["url"]
                quality_label = quality
                break

        if best_url:
            print(f"  Video URL ({quality_label}): {best_url}")

        print(f"  Preview page: {video.get('pageURL')}")
        print()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()