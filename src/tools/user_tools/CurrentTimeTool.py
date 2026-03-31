import datetime

class CurrentTimeTool:
    __all__ = ['CurrentTimeTool']
    inputSchema = {}

    def run(self):
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}"
