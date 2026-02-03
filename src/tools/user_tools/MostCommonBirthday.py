import requests
from bs4 import BeautifulSoup

__all__ = ['MostCommonBirthday']

class MostCommonBirthday():
    dependencies = ["requests", "beautifulsoup4"]

    inputSchema = {
        "name": "MostCommonBirthday",
        "description": "Finds the most common birthday from a census.gov page.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    }

    def run(self, **kwargs):
        try:
            response = requests.get("https://www.census.gov/newsroom/blogs/random-samplings/2015/09/how-common-is-your-birthday.html")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                most_common_row = rows[1]
                columns = most_common_row.find_all('td')
                most_common_date = columns[0].text.strip()
                return {"status": "success", "message": "Most common birthday found", "output": most_common_date}
            else:
                return {"status": "error", "message": "Table not found", "output": "Could not retrieve data. Table not found on page."}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Request error: {e}", "output": "Could not retrieve data. Request error."} # Updated message
        except Exception as e:
            return {"status": "error", "message": f"An error occurred: {e}", "output": "Could not retrieve data. An error occurred."} # Updated message
