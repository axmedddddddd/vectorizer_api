# API векторизации текста

Данный API разработан с целью облегчить развертывание новых приложений, требующих в своей работе векторизацию текста.

В настоящий момент, доступ к API осуществляется по адресу `https://footprint.auditory.ru/api/vec/`

## Локальное использование

Установка

```bash
pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
```

Запуск сервера

```bash
python -m uvicorn vectorServer.app:app --port 5000 --reload
```

## Документация

Документация в формате _OpenAPI_:
- в случае удаленного использования: `https://footprint.auditory.ru/api/vec/docs`
- в случае локального использования: `localhost:5000/docs`

### Пример использования

Ниже продемонстрирован минимальный пример отправки запроса к API и обработки ответа на языке *Python*.

```python
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
```