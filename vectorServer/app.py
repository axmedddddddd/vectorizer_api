import logging
from typing import List
import os

from fastapi import FastAPI
from fastapi.params import Query

from vectorServer.models import ModelSelection, model_mapping
from vectorServer.schemas import QueryResponse

import aioredis
import json
from Crypto.Hash import BLAKE2b
import zlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            filename="log.log", 
            mode="a",
            maxBytes=5*1024*1024,
            backupCount=2,
            encoding=None,
            delay=0
        ),
        logging.StreamHandler()
    ]
)

app = FastAPI()
redis = None

redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', '6379')
redis_password = os.environ.get('REDIS_PASSWORD', None)

@app.on_event('startup')
async def startup_event():
    global redis
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = os.environ.get('REDIS_PORT', '6379')
    redis_password = os.environ.get('REDIS_PASSWORD', None)
    redis = await aioredis.create_redis_pool(
        f"redis://{redis_host}:{redis_port}",
        password=redis_password if redis_password else None
    )


@app.on_event('shutdown')
async def shutdown_event():
    redis.close()
    await redis.wait_closed()


def get_prefixed_hashed_key(key: str) -> str:
    prefix = "400:vec:"
    hash_length = 16
    hashed_key = BLAKE2b.new(digest_bits=8*hash_length, data=key.encode()).hexdigest()
    return f"{prefix}{hashed_key}"
    

def compress_response(response: QueryResponse) -> bytes:
    compressed_response = zlib.compress(response.json().encode())
    return compressed_response

def decompress_response(compressed_cache: bytes) -> QueryResponse:
    decompressed_cache = zlib.decompress(compressed_cache)
    response = json.loads(decompressed_cache)
    return response

@app.get("/vectorize_token", response_model=QueryResponse)
async def vectorize_token(
    token: str = Query("python", max_length=250),
    model_name: ModelSelection = ModelSelection.BERT,
):
    """Векторизация запроса, состоящего из *одного* слова

    Args:
        token (str): токен для векторизации
        model_name (ModelSelection, optional): Модель, используемая для векторизации.
            Defaults to ModelSelection.BERT.
    \f
    Returns:
        List[float]: векторное представление токена
    """
    
    # получаем хэшированное имя ключа с префиксом
    key = get_prefixed_hashed_key(token)
    
    # получаем кэш из памяти, используя хэшированное имя ключа
    compressed_cache = await redis.get(key)

    # если кэш не пустой, то возвращаем его
    if compressed_cache is not None:
        decompress_response(compressed_cache)
    
    # иначе выполняем векторизацию токена
    vector = model_mapping[model_name.value].vectorize_token(token)
    response = QueryResponse(query=token, vector=vector)

    # сжимаем и сохраняем кэш в памяти, используя хэшированное имя ключа
    compressed_response = compress_response(response)
    await redis.set(key, compressed_response)
    await redis.execute('EXPIRE', key, '3600')  # Удаление записи через час
    
    return response


@app.get("/vectorize_raw_text", response_model=QueryResponse)
async def vectorize_raw_text(
    text: str = Query("мама мыла раму"),
    model_name: ModelSelection = ModelSelection.BERT,
):
    """Векторизация запроса, состоящего из нескольких слов

    Args:
        text (str): текст для векторизации
        model_name (ModelSelection, optional): Модель, используемая для векторизации.
            Defaults to ModelSelection.BERT.
    \f
    Returns:
        List[float]: векторное представление текста
    """
    
    # получаем хэшированное имя ключа с префиксом
    key = get_prefixed_hashed_key(text)
    
    # получаем кэш из памяти, используя хэшированное имя ключа
    compressed_cache = await redis.get(key)

    # если кэш не пустой, то возвращаем его
    if compressed_cache is not None:
        decompress_response(compressed_cache)
    
    vector = model_mapping[model_name.value].vectorize_text(text)
    response = QueryResponse(query=text, vector=vector)
    
    # сжимаем и сохраняем кэш в памяти, используя хэшированное имя ключа
    compressed_response = compress_response(response)
    await redis.set(key, compressed_response)
    await redis.execute('EXPIRE', key, '3600')  # Удаление записи через час
    
    return response


@app.get("/vectorize_multiple_tokens", response_model=QueryResponse)
async def vectorize_multiple_tokens(
    tokens: List[str] = Query(["мама", "мыла", "раму"]),
    model_name: ModelSelection = ModelSelection.BERT,
):
    """Векторизация запроса, состоящего из нескольких слов

    Args:
        tokens (str): текст для векторизации
        model_name (ModelSelection, optional): Модель, используемая для векторизации.
            Defaults to ModelSelection.BERT.
    \f
    Returns:
        List[float]: векторное представление текста
    """
    string_of_tokens = str(tokens)
    
    # получаем хэшированное имя ключа с префиксом
    key = get_prefixed_hashed_key(string_of_tokens)
    
    # получаем кэш из памяти, используя хэшированное имя ключа
    compressed_cache = await redis.get(key)

    # если кэш не пустой, то возвращаем его
    if compressed_cache is not None:
        decompress_response(compressed_cache)
    
    vector = model_mapping[model_name.value].vectorize_array(tokens)
    response = QueryResponse(query=" ".join(tokens), vector=vector)
    
    # сжимаем и сохраняем кэш в памяти, используя хэшированное имя ключа
    compressed_response = compress_response(response)
    await redis.set(key, compressed_response)
    await redis.execute('EXPIRE', key, '3600')  # Удаление записи через час
    
    return response