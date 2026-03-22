import importlib
import requests
from bs4 import BeautifulSoup

__all__ = ['WebsiteTextExtractor']

class WebsiteTextExtractor:
    dependencies = ["requests", "beautifulsoup4"]

    inputSchema = {
        "name": "WebsiteTextExtractor",
        "description": "Extracts text content from a website.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to extract content from.",
                }
            },
            "required": ["url"],
        }
    }

    def run(self, **kwargs):
        url = kwargs.get("url")

        if not url:
            return {
                "status": "error",
                "message": "Missing required parameters: 'url'",
                "output": None
            }

        try:
            requests = importlib.import_module("requests")
            BeautifulSoup = importlib.import_module("bs4").BeautifulSoup

            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)

            return {
                "status": "success",
                "message": "Text extracted successfully",
                "output": text
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "output": None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None
            }
