import math

__all__ = ['WordsPerMinuteCalculator']

class WordsPerMinuteCalculator:
    dependencies = []
    inputSchema = {
        "name": "WordsPerMinuteCalculator",
        "description": "Calculates words per minute.",
        "parameters": {
            "type": "object",
            "properties": {
                "word_count": {
                    "type": "integer",
                    "description": "The number of words.",
                },
                "minutes": {
                    "type": "integer",
                    "description": "The number of minutes.",
                }
            },
            "required": ["word_count", "minutes"],
        }
    }

    def run(self, **kwargs):
        word_count = kwargs.get("word_count")
        minutes = kwargs.get("minutes")

        if minutes <= 0:
            return {
                "status": "error",
                "message": "Minutes must be greater than zero.",
                "output": None
            }
        try:
            wpm = float(word_count) / float(minutes)
            return {
                "status": "success",
                "message": "WPM calculated successfully.",
                "output": wpm,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Calculation failed: {str(e)}",
                "output": None,
            }
