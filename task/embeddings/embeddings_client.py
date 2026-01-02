# import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment: str, api_key: str):
        if not api_key or api_key.strip() == '':
            raise ValueError("API key cannot be null or empty")

        self._deployment=deployment,
        self._api_key=api_key
        self._url = DIAL_EMBEDDINGS.format(model=deployment)

    def get_embeddings(self, inputs: str | list[str], dimensions) -> dict[int, list[float]]:
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "input": inputs,
            "dimensions": dimensions
        }

        try:
            res = requests.post(self._url, json=payload, headers=headers)
            res.raise_for_status()
            parsed = res.json()
            data = parsed.get("data", [])
            return self._format_embeddings(data)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            raise e

    def _format_embeddings(self, data: list[dict]) -> dict[int, list[float]]:
        return {emb["index"]:emb["embedding"] for emb in data}

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
