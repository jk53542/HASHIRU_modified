import importlib
import re

__all__ = ['WordCountExtractor']

class WordCountExtractor:
    dependencies = ["requests", "beautifulsoup4"]

    inputSchema = {
        "name": "WordCountExtractor",
        "description": "Extracts the word count from a website.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to extract the word count from.",
                },
            },
            "required": ["url"],
        },
    }

    def run(self, **kwargs):
        url = kwargs.get("url")

        if not url:
            return {
                "status": "error",
                "message": "Missing required parameter: 'url'",
                "output": None,
            }

        try:
            requests = importlib.import_module("requests")
            bs4 = importlib.import_module("beautifulsoup4")
            BeautifulSoup = bs4.BeautifulSoup

            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()

            # Remove punctuation and split into words
            words = re.findall(r"\b\w+\b", text.lower())
            word_count = len(words)

            return {
                "status": "success",
                "message": "Word count extracted successfully.",
                "output": word_count,
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "output": None,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None,
            }
