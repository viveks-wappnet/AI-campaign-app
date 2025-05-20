import logging
import requests

logger = logging.getLogger(__name__)

def download_file(url: str, dest: str) -> None:
    """Download a file from URL to destination path."""
    logger.info(f"Downloading from: {url}")
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        logger.info(f"Download completed: {dest}")
    except requests.RequestException as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        raise 