import requests
import json

class LogService:
    def __init__(self, host) -> None:
        self.host = host
        self.headers = {'Content-type': 'application/json'}

    def log(self, payload):
        payloadToSend = json.dumps(payload)
        requests.post(self.host, data=payloadToSend, headers=self.headers)
