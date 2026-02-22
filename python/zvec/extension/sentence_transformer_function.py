# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Literal, Optional

from ..tool import require_module


class SentenceTransformerFunctionBase:
    """Clase base para funciones de Sentence Transformer (densas y dispersas).

    Esta clase base proporciona funcionalidades comunes para cargar y gestionar
    modelos de sentence-transformers desde Hugging Face o ModelScope. Soporta
    tanto modelos densos (p.ej., all-MiniLM-L6-v2) como dispersos (p.ej., SPLADE).

    Esta clase no está destinada a usarse directamente. Usa las implementaciones concretas:
    - ``SentenceTransformerEmbeddingFunction`` para embeddings densos
    - ``SentenceTransformerSparseEmbeddingFunction`` para embeddings dispersos
    - ``DefaultDenseEmbedding`` para embeddings densos por defecto
    - ``DefaultSparseEmbedding`` para embeddings dispersos por defecto

    Args:
        model_name (str): Identificador del modelo o ruta local.
        model_source (Literal["huggingface", "modelscope"]): Fuente del modelo.
        device (Optional[str]): Dispositivo en el que ejecutar el modelo.
        trust_remote_code (bool): Si se permite la ejecución de código personalizado
            del modelo desde el repositorio. Por defecto ``False``.

    Note:
        - Esta es una clase base interna para reutilización de código
        - Las subclases deben heredar del Protocol adecuado (Denso/Disperso)
        - Proporciona funcionalidad de carga y gestión de modelos
        - ``trust_remote_code=True`` permite que código Python arbitrario de un
          repositorio de modelos descargado se ejecute; actívalo solo para modelos de confianza
    """

    def __init__(
        self,
        model_name: str,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """Inicializa la funcionalidad base de Sentence Transformer.

        Args:
            model_name (str): Identificador del modelo o ruta local.
            model_source (Literal["huggingface", "modelscope"]): Fuente del modelo.
            device (Optional[str]): Dispositivo en el que ejecutar el modelo.
            trust_remote_code (bool): Si se permite la ejecución de código
                personalizado del modelo desde el repositorio. Por defecto ``False``.

                .. warning::
                    Establecer esto en ``True`` permite que código Python arbitrario de un
                    repositorio de modelos descargado se ejecute en tu máquina. Solo
                    actívalo para modelos en los que confíes explícitamente.

        Raises:
            ValueError: Si model_source no es válido.
        """
        # Validar model_source
        if model_source not in ("huggingface", "modelscope"):
            raise ValueError(
                f"Invalid model_source: '{model_source}'. "
                "Must be 'huggingface' or 'modelscope'."
            )

        self._model_name = model_name
        self._model_source = model_source
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model = None

    @property
    def model_name(self) -> str:
        """str: El nombre del modelo Sentence Transformer actualmente en uso."""
        return self._model_name

    @property
    def model_source(self) -> str:
        """str: La fuente del modelo en uso ("huggingface" o "modelscope")."""
        return self._model_source

    @property
    def device(self) -> str:
        """str: El dispositivo en el que se ejecuta el modelo."""
        model = self._get_model()
        if model is not None:
            return str(model.device)
        return self._device or "cpu"

    def _get_model(self):
        """Carga o recupera el modelo Sentence Transformer.

        Returns:
            SentenceTransformer o SparseEncoder: La instancia del modelo cargada.

        Raises:
            ImportError: Si los paquetes requeridos no están instalados.
            ValueError: Si el modelo no se puede cargar.
        """
        # Retornar el modelo en caché si existe
        if self._model is not None:
            return self._model

        # Cargar el modelo
        try:
            sentence_transformers = require_module("sentence_transformers")

            if self._model_source == "modelscope":
                # Cargar desde ModelScope
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                # Descargar el modelo a caché
                model_dir = snapshot_download(self._model_name)

                # Cargar desde ruta local
                self._model = sentence_transformers.SentenceTransformer(
                    model_dir, device=self._device, trust_remote_code=self._trust_remote_code
                )
            else:
                # Cargar desde Hugging Face (predeterminado)
                self._model = sentence_transformers.SentenceTransformer(
                    self._model_name, device=self._device, trust_remote_code=self._trust_remote_code
                )

            return self._model

        except ImportError as e:
            if "modelscope" in str(e) and self._model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load Sentence Transformer model '{self._model_name}' "
                f"from {self._model_source}: {e!s}"
            ) from e

    def _is_sparse_model(self) -> bool:
        """Verifica si el modelo cargado es un codificador disperso (p.ej., SPLADE).

        Returns:
            bool: True si el modelo soporta codificación dispersa.
        """
        model = self._get_model()
        # Verificar si el modelo tiene métodos de codificación dispersa
        return hasattr(model, "encode_query") or hasattr(model, "encode_document")
