import os, requests
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

class IBMScoringClient:
    def __init__(self):
        self.api_key = os.getenv("IBM_API_KEY")
        self.url = os.getenv("IBM_SCORING_URL")
        if not self.api_key or not self.url:
            raise ValueError("IBM_API_KEY and IBM_SCORING_URL are required.")

    def predict(self, fields, values):
        token = IAMAuthenticator(self.api_key).token_manager.get_token()
        payload = {"input_data": [{"fields": fields, "values": values}]}
        r = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        return r.json()
