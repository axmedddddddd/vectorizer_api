import requests

BASE_URL = "https://footprint.auditory.ru/api/vec/"


def vectorize_token(token, model_name):
    response = requests.get(
        f"{BASE_URL}/vectorize_token", params={"query": token, "model_name": model_name}
    )
    data = response.json()
    return data['vector']


def vectorize_text(text, model_name):
    response = requests.get(
        f"{BASE_URL}/vectorize_raw_text", params={"query": text, "model_name": model_name}
    )

    data = response.json()
    return data['vector']


def vectorize_array(tokens, model_name):
    response = requests.get(
        f"{BASE_URL}/vectorize_multiple_tokens",
        params={"query": tokens, "model_name": model_name},
    )

    data = response.json()
    return data['vector']

token = "математика"
text = "программист математик"
tokens_array = ["программист", "математик"]

token_vector = vectorize_token(token, model_name="bert")
text_vector = vectorize_text(text, model_name="bert")
tokens_array_vector = vectorize_array(tokens_array, model_name="bert")

print(type(token_vector), type(token_vector[0]))  # <class 'list'> <class 'float'>
print(type(text_vector), type(text_vector[0]))  # <class 'list'> <class 'float'>
print(type(tokens_array_vector), type(tokens_array_vector[0]))  # <class 'list'> <class 'float'>

# Преобразование ответа в np.ndarray

import numpy as np

token_vector = np.array(token_vector)
text_vector = np.array(text_vector)

print(f"{token_vector.shape = }")  # token_vector.shape = (384,)
print(f"{text_vector.shape = }")  # text_vector.shape = (384,)
