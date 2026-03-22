import importlib

__all__ = ['WebsiteContentExtractor']

class WebsiteContentExtractor:
    dependencies = ["requests", "beautifulsoup4"]

    inputSchema = {
        "name": "WebsiteContentExtractor",
        "description": "Extracts text content from a website, optionally using a CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website.",
                },
                "css_selector": {
                    "type": "string",
                    "description": "A CSS selector to target specific content. Defaults to None.",
                }
            },
            "required": ["url"],
        }
    }

    def run(self, **kwargs):
        url = kwargs.get("url")
        css_selector = kwargs.get("css_selector")

        if not url:
            return {
                "status": "error",
                "message": "Missing required parameters: 'url'",
                "output": None
            }

        try:
            requests = importlib.import_module("requests")
            BeautifulSoup = importlib.import_module("bs4").BeautifulSoup

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            if css_selector:
                elements = soup.select(css_selector)
                text_content = "\n".join(element.get_text(separator="\n") for element in elements)
            else:
                text_content = soup.get_text(separator="\n")

            return {
                "status": "success",
                "message": "Content extracted successfully",
                "output": text_content
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Error fetching URL: {e}",
                "output": None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing website: {e}",
                "output": None
            }
