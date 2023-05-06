import os
import pathlib
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import List
import shutil
import logging

import numpy as np
from gensim.models import KeyedVectors
from navec import Navec
from razdel import tokenize
from sentence_transformers import SentenceTransformer

from vectorServer.utils.downloader import download_file


ROOT_PATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
MODELS_PATH = ROOT_PATH / "models_storage/"

if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)


class ModelSelection(Enum):
    BERT = "bert"
    NAVEC = "navec"
    FASTTEXT = "fasttext"


class NLPModel(ABC):
    """Абстрактный класс для модели векторизации, способной:
    1. Векторизаовтаь токен
    2. Векторизовать текст
    3. Векторизовать список токенов

    Также, предварительно создает папку модели `MODEL_PATH/model_name`

    Attributes:
        path (str): Локальный путь до модели
        model (Any): Модель векторизации
    """

    def __init__(self, model_name: str, model_math: str) -> None:
        super().__init__()

        self.model = None
        self.model_dir = MODELS_PATH / model_name

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.model_path = self.model_dir / model_math
        self._output_size = None
        self.download_model()
        self.read_model()

    @abstractproperty
    def output_size(self):
        return self._output_size

    @abstractmethod
    def download_model(self):
        pass

    @abstractmethod
    def read_model(self):
        """Чтение модели из `self.path`"""
        pass

    @abstractmethod
    def vectorize_token(self, token: str) -> List[float]:
        """Векторизация токена (одного слова)

        Args:
            token (str): токен для векторизации

        Returns:
            List[float]: вектор
        """
        pass

    def vectorize_array(self, tokens: List[str]) -> List[float]:
        tokens_vectors = [self.vectorize_token(token) for token in tokens]
        vector = np.mean(tokens_vectors, axis=0).tolist()
        return vector

    def vectorize_text(self, text: str) -> List[float]:
        """Векторизация текста (нескольких токенов)

        По умолчанию реализовано через метод вектоирзации токенов:
        1. Текст токенизируется
        2. Каждый токен векторизуется через `self.vectorize_token`

        Args:
            text (str): текст для векторизации

        Returns:
            list[float]: вектор
        """
        tokens = [t.text for t in tokenize(text)]
        tokens_vectors = [self.vectorize_token(token) for token in tokens]
        vector = np.mean(tokens_vectors, axis=0).tolist()
        return vector


class NavecModel(NLPModel):
    def __init__(self) -> None:
        super().__init__("navec", "navec_hudlit_v1_12B_500K_300d_100q.tar")
        self._output_size = 300

    @property
    def output_size(self):
        return super().output_size

    def download_model(self):
        logging.info("Setting environment (navec)")
        if not os.path.exists(self.model_path):
            download_file(
                "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar",
                target_filename=str(self.model_path.absolute()),
                description="Navec model download",
            )

    def read_model(self):
        self.model = Navec.load(self.model_path)

    def vectorize_token(self, token: str) -> List[float]:
        if token in self.model:
            return self.model[token].tolist()
        else:
            return self.model["<unk>"].tolist()


class BERTModel(NLPModel):
    def __init__(self) -> None:
        super().__init__("bert", "sentence-transformers_multi-qa-MiniLM-L6-cos-v1")
        self._output_size = 384

    def download_model(self):
        # no need to download manually
        # but it is possible
        # https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.save
        logging.info("Setting environment (bert)")

    def read_model(self):
        self.model = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            cache_folder=str(self.model_path.absolute()),
        )

    @property
    def output_size(self):
        return super().output_size

    def vectorize_token(self, token: str) -> List[float]:
        return self.model.encode(token, convert_to_numpy=True).tolist()

    def vectorize_text(self, text: str) -> List[float]:
        """Векторизация текста.

        Так как `sentence_transformers` позволяет векторизовать текст
        в целом, пользуемся этой возможностью

        Args:
            text (str): текст для векторизации

        Return:
            List[float]: вектор
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()


class FASTTextModel(NLPModel):
    def __init__(self) -> None:
        super().__init__("fasttext", "araneum_none_fasttextcbow_300_5_2018.model")
        self._output_size = 300

    def download_model(self):
        # araneum_none_fasttextcbow_300_5_2018
        logging.info("Setting environment (fasttext)")
        if not os.path.exists(self.model_path):
            archive_path = str(
                self.model_dir / "araneum_none_fasttextcbow_300_5_2018.tgz"
            )
            download_file(
                "https://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz",
                target_filename=archive_path,
                description="FastText download",
            )
            shutil.unpack_archive(archive_path, self.model_dir)
            os.remove(archive_path)

    def read_model(self):
        self.model = KeyedVectors.load(str(self.model_path.absolute()), mmap='r')

    @property
    def output_size(self):
        return super().output_size

    def vectorize_token(self, token: str) -> List[float]:
        return self.model[token].tolist()

# таблица соответствия имени модели экземпляру класса
model_mapping = {
    "navec": NavecModel(),
    "bert": BERTModel(),
    "fasttext": FASTTextModel(),
}
