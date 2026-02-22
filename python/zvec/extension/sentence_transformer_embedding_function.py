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

from typing import ClassVar, Literal, Optional

import numpy as np

from ..common.constants import TEXT, DenseVectorType, SparseVectorType
from .embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalDenseEmbedding(
    SentenceTransformerFunctionBase, DenseEmbeddingFunction[TEXT]
):
    """Embedding denso local por defecto usando el modelo all-MiniLM-L6-v2.

    Esta es la implementación por defecto para embedding de texto denso que utiliza
    el modelo ``all-MiniLM-L6-v2`` de Hugging Face por defecto. Este modelo proporciona
    un buen equilibrio entre velocidad y calidad para el embedding de texto de propósito general.

    La clase proporciona capacidades de embedding denso de texto a vector usando la
    librería sentence-transformers. Soporta modelos de Hugging Face Hub y ModelScope,
    se ejecuta localmente sin llamadas a API y soporta aceleración CPU/GPU.

    El modelo produce embeddings de 384 dimensiones y está optimizado para tareas de
    similitud semántica. Se ejecuta localmente sin requerir claves de API.

    Args:
        model_source (Literal["huggingface", "modelscope"], optional): Fuente del modelo.
            - ``"huggingface"``: Usar Hugging Face Hub (predeterminado, para usuarios internacionales)
            - ``"modelscope"``: Usar ModelScope (recomendado para usuarios en China)
            Por defecto ``"huggingface"``.
        device (Optional[str], optional): Dispositivo en el que ejecutar el modelo.
            Opciones: ``"cpu"``, ``"cuda"``, ``"mps"`` (para Apple Silicon) o ``None``
            para detección automática. Por defecto ``None``.
        normalize_embeddings (bool, optional): Si se normalizan los embeddings a
            longitud unitaria (normalización L2). Útil para similitud coseno.
            Por defecto ``True``.
        batch_size (int, optional): Tamaño de lote para la codificación. Por defecto ``32``.
        **kwargs: Parámetros adicionales para extensión futura.

    Attributes:
        dimension (int): Siempre 384 para ambos modelos.
        model_name (str): "all-MiniLM-L6-v2" (HF) o "iic/nlp_gte_sentence-embedding_chinese-small" (MS).
        model_source (str): La fuente del modelo en uso.
        device (str): El dispositivo en el que se ejecuta el modelo.

    Raises:
        ValueError: Si el modelo no se puede cargar o la entrada no es válida.
        TypeError: Si la entrada de ``embed()`` no es una cadena.
        RuntimeError: Si la inferencia del modelo falla.

    Note:
        - Requiere Python 3.10, 3.11 o 3.12
        - Requiere el paquete ``sentence-transformers``:
          ``pip install sentence-transformers``
        - Para ModelScope, también requiere: ``pip install modelscope``
        - La primera ejecución descarga el modelo (~50-80MB) desde la fuente elegida
        - Caché de Hugging Face: ``~/.cache/torch/sentence_transformers/``
        - Caché de ModelScope: ``~/.cache/modelscope/hub/``
        - No se requieren claves de API ni red tras la descarga inicial
        - Velocidad de inferencia: ~1000 frases/seg en CPU, ~10000 en GPU

        **Para usuarios en China:**

        Si encuentras problemas de acceso a Hugging Face, usa ModelScope:

        .. code-block:: python

            # Recomendado para usuarios en China
            emb = DefaultLocalDenseEmbedding(model_source="modelscope")

        Alternativamente, usa el espejo de Hugging Face:

        .. code-block:: bash

            export HF_ENDPOINT=https://hf-mirror.com
            # Luego usa el modo predeterminado de Hugging Face

    Examples:
        >>> # Uso básico con Hugging Face (predeterminado)
        >>> from zvec.extension import DefaultLocalDenseEmbedding
        >>>
        >>> emb_func = DefaultLocalDenseEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        384
        >>> isinstance(vector, list)
        True

        >>> # Recomendado para usuarios en China (usa ModelScope)
        >>> emb_func = DefaultLocalDenseEmbedding(model_source="modelscope")
        >>> vector = emb_func.embed("你好，世界！")  # Funciona bien con texto en chino
        >>> len(vector)
        384

        >>> # Alternativa para usuarios en China: usar espejo de Hugging Face
        >>> import os
        >>> os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        >>> emb_func = DefaultLocalDenseEmbedding()  # Usa el espejo de HF
        >>> vector = emb_func.embed("Hello, world!")

        >>> # Usar GPU para inferencia más rápida
        >>> emb_func = DefaultLocalDenseEmbedding(device="cuda")
        >>> vector = emb_func("Machine learning is fascinating")
        >>> # El vector normalizado tiene longitud unitaria
        >>> import numpy as np
        >>> np.linalg.norm(vector)
        1.0

        >>> # Procesamiento por lotes
        >>> texts = ["First text", "Second text", "Third text"]
        >>> vectors = [emb_func.embed(text) for text in texts]
        >>> len(vectors)
        3
        >>> all(len(v) == 384 for v in vectors)
        True

        >>> # Similitud semántica
        >>> v1 = emb_func.embed("The cat sits on the mat")
        >>> v2 = emb_func.embed("A feline rests on a rug")
        >>> v3 = emb_func.embed("Python programming")
        >>> similarity_high = np.dot(v1, v2)  # Frases similares
        >>> similarity_low = np.dot(v1, v3)   # Temas diferentes
        >>> similarity_high > similarity_low
        True

        >>> # Manejo de errores
        >>> try:
        ...     emb_func.embed("")  # Cadena vacía
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``DenseEmbeddingFunction``: Clase base para embeddings densos
        - ``DefaultLocalSparseEmbedding``: Embedding disperso con SPLADE
        - ``QwenDenseEmbedding``: Alternativa usando la API de Qwen
    """

    def __init__(
        self,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Inicializa con el modelo all-MiniLM-L6-v2.

        Args:
            model_source (Literal["huggingface", "modelscope"]): Fuente del modelo.
                Por defecto "huggingface".
            device (Optional[str]): Dispositivo destino ("cpu", "cuda", "mps" o None).
                Por defecto None (detección automática).
            normalize_embeddings (bool): Si se normalizan los vectores de salida (L2).
                Por defecto True.
            batch_size (int): Tamaño de lote para la codificación. Por defecto 32.
            trust_remote_code (bool): Si se permite la ejecución de código personalizado
                del modelo desde el repositorio. Por defecto ``False``.

                .. warning::
                    Establecer esto en ``True`` permite que código Python arbitrario de un
                    repositorio de modelos descargado se ejecute en tu máquina. Solo
                    actívalo para modelos en los que confíes explícitamente.
            **kwargs: Parámetros adicionales para extensión futura.

        Raises:
            ImportError: Si sentence-transformers o modelscope no están instalados.
            ValueError: Si el modelo no se puede cargar.
        """
        # Usar diferentes modelos según la fuente
        if model_source == "modelscope":
            # Usar modelo optimizado para chino en ModelScope (mejor para texto chino)
            model_name = "iic/nlp_gte_sentence-embedding_chinese-small"
        else:
            model_name = "all-MiniLM-L6-v2"

        # Inicializar la clase base para la carga del modelo
        SentenceTransformerFunctionBase.__init__(
            self,
            model_name=model_name,
            model_source=model_source,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size

        # Cargar el modelo y obtener la dimensión
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension()

        # Almacenar parámetros adicionales
        self._extra_params = kwargs

    @property
    def dimension(self) -> int:
        """int: La dimensionalidad esperada del vector de embedding."""
        return self._dimension

    @property
    def extra_params(self) -> dict:
        """dict: Parámetros adicionales para la personalización específica del modelo."""
        return self._extra_params

    def __call__(self, input: str) -> DenseVectorType:
        """Hace que la función de embedding sea invocable."""
        return self.embed(input)

    def embed(self, input: str) -> DenseVectorType:
        """Genera el vector de embedding denso para el texto de entrada.

        Este método usa el modelo Sentence Transformer para convertir el texto de
        entrada en una representación vectorial densa. El modelo se ejecuta localmente
        sin requerir llamadas a API.

        Args:
            input (str): Cadena de texto de entrada a embeber. Debe ser no vacía tras
                eliminar espacios. La longitud máxima depende del modelo usado
                (típicamente 128-512 tokens para la mayoría de modelos).

        Returns:
            DenseVectorType: Una lista de flotantes que representa el vector de embedding.
                La longitud es igual a ``self.dimension``. Si ``normalize_embeddings=True``,
                el vector tiene longitud unitaria. Ejemplo:
                ``[0.123, -0.456, 0.789, ...]``

        Raises:
            TypeError: Si ``input`` no es una cadena.
            ValueError: Si la entrada está vacía o contiene solo espacios en blanco.
            RuntimeError: Si la inferencia del modelo falla.

        Examples:
            >>> emb = DefaultLocalDenseEmbedding()
            >>> vector = emb.embed("Natural language processing")
            >>> len(vector)
            384
            >>> isinstance(vector[0], float)
            True

            >>> # Los vectores normalizados tienen longitud unitaria
            >>> import numpy as np
            >>> emb = DefaultLocalDenseEmbedding(normalize_embeddings=True)
            >>> vector = emb.embed("Test sentence")
            >>> np.linalg.norm(vector)
            1.0

            >>> # Error: entrada vacía
            >>> emb.embed("   ")
            ValueError: Input text cannot be empty or whitespace only

            >>> # Error: entrada que no es cadena
            >>> emb.embed(123)
            TypeError: Expected 'input' to be str, got int

            >>> # Ejemplo de similitud semántica
            >>> v1 = emb.embed("The cat sits on the mat")
            >>> v2 = emb.embed("A feline rests on a rug")
            >>> similarity = np.dot(v1, v2)  # Alta similitud por significado semántico
            >>> similarity > 0.7
            True

        Note:
            - La primera llamada puede ser más lenta por la carga del modelo
            - Las llamadas siguientes son mucho más rápidas ya que el modelo permanece en memoria
            - Para procesamiento por lotes, considera codificar varios textos juntos
              (aunque este método solo maneja textos individuales)
            - La aceleración GPU proporciona una mejora de velocidad de 5-10x sobre CPU
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            model = self._get_model()
            embedding = model.encode(
                input,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
                batch_size=self._batch_size,
            )

            # Convertir array numpy a lista
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)

            # Validar dimensión
            if len(embedding_list) != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding_list)}"
                )

            return embedding_list

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate embedding: {e!s}") from e


class DefaultLocalSparseEmbedding(
    SentenceTransformerFunctionBase, SparseEmbeddingFunction[TEXT]
):
    """Embedding disperso local por defecto usando el modelo SPLADE.

    Esta clase proporciona embedding de vectores dispersos usando el modelo SPLADE
    (SParse Lexical AnD Expansion). SPLADE genera representaciones dispersas e
    interpretables donde cada dimensión corresponde a un término del vocabulario con
    pesos de importancia aprendidos. Es ideal para búsqueda léxica, recuperación
    estilo BM25 y escenarios de búsqueda híbrida.

    El modelo por defecto es ``naver/splade-cocondenser-ensembledistil``, disponible
    públicamente sin autenticación. Produce vectores dispersos con miles de dimensiones
    pero solo cientos de valores distintos de cero, lo que los hace eficientes para
    almacenamiento y recuperación manteniendo una fuerte correspondencia léxica.

    **Caché de modelos:**

    Esta clase usa caché a nivel de clase para compartir el modelo SPLADE entre todas las
    instancias con la misma configuración (model_source, device). Esto reduce
    significativamente el uso de memoria al crear múltiples instancias para distintos
    tipos de codificación (consulta vs documento).

    **Gestión de caché:**

    La clase proporciona métodos para gestionar la caché del modelo:

    - ``clear_cache()``: Limpiar todos los modelos en caché para liberar memoria
    - ``get_cache_info()``: Obtener información sobre los modelos en caché
    - ``remove_from_cache(model_source, device)``: Eliminar un modelo específico de la caché

    .. note::
        **¿Por qué no usar splade-v3?**

        El modelo más reciente ``naver/splade-v3`` está bloqueado (requiere aprobación de acceso).
        Usamos ``naver/splade-cocondenser-ensembledistil`` en su lugar.

        **Para usar splade-v3 (si tienes acceso):**

        1. Solicita acceso en https://huggingface.co/naver/splade-v3
        2. Obtén tu token de Hugging Face en https://huggingface.co/settings/tokens
        3. Establece la variable de entorno:

           .. code-block:: bash

               export HF_TOKEN="your_huggingface_token"

        4. O inicia sesión programáticamente:

           .. code-block:: python

               from huggingface_hub import login
               login(token="your_huggingface_token")

        5. Para usar un modelo SPLADE personalizado, puedes crear una subclase y
           sobreescribir model_name en ``__init__``, o crear tu propia implementación
           heredando de ``SentenceTransformerFunctionBase`` y ``SparseEmbeddingFunction``.

    Args:
        model_source (Literal["huggingface", "modelscope"], optional): Fuente del modelo.
            Por defecto ``"huggingface"``. El soporte de ModelScope puede variar para modelos SPLADE.
        device (Optional[str], optional): Dispositivo en el que ejecutar el modelo.
            Opciones: ``"cpu"``, ``"cuda"``, ``"mps"`` (para Apple Silicon) o ``None``
            para detección automática. Por defecto ``None``.
        encoding_type (Literal["query", "document"], optional): Tipo de codificación.
            - ``"query"``: Optimizar para consultas de búsqueda (predeterminado)
            - ``"document"``: Optimizar para documentos indexados
        **kwargs: Parámetros adicionales (actualmente no usados, para extensión futura).

    Attributes:
        model_name (str): Identificador del modelo.
        model_source (str): La fuente del modelo en uso.
        device (str): El dispositivo en el que se ejecuta el modelo.

    Raises:
        ValueError: Si el modelo no se puede cargar o la entrada no es válida.
        TypeError: Si la entrada de ``embed()`` no es una cadena.
        RuntimeError: Si la inferencia del modelo falla.

    Note:
        - Requiere Python 3.10, 3.11 o 3.12
        - Requiere el paquete ``sentence-transformers``:
          ``pip install sentence-transformers``
        - La primera ejecución descarga el modelo (~100MB) desde Hugging Face
        - Ubicación de la caché: ``~/.cache/torch/sentence_transformers/``
        - No se requieren claves de API ni autenticación
        - Los vectores dispersos tienen ~30k dimensiones pero solo ~100-200 valores distintos de cero
        - Se combina mejor con embeddings densos para recuperación híbrida

        **SPLADE vs Embeddings Densos:**

        - **Denso**: Vectores semánticos continuos, buenos para similitud semántica
        - **Disperso**: Basado en palabras clave léxicas, interpretable, bueno para búsqueda exacta
        - **Híbrido**: Combina ambos para el mejor rendimiento de recuperación

    Examples:
        >>> # Eficiente en memoria: ambas instancias comparten el mismo modelo (~200MB)
        >>> from zvec.extension import DefaultLocalSparseEmbedding
        >>>
        >>> # Embedding de consulta
        >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
        >>> query_vec = query_emb.embed("machine learning algorithms")
        >>> type(query_vec)
        <class 'dict'>
        >>> len(query_vec)  # Solo dimensiones distintas de cero
        156

        >>> # Embedding de documento (comparte modelo con query_emb)
        >>> doc_emb = DefaultLocalSparseEmbedding(encoding_type="document")
        >>> doc_vec = doc_emb.embed("Machine learning is a subset of AI")
        >>> # Memoria total: ~200MB (no 400MB) gracias a la caché de modelos

        >>> # Ejemplo de recuperación asimétrica
        >>> query_vec = query_emb.embed("what causes aging fast")
        >>> doc_vec = doc_emb.embed(
        ...     "UV-A light causes tanning, skin aging, and cataracts..."
        ... )
        >>>
        >>> # Calcular similitud (producto escalar para vectores dispersos)
        >>> similarity = sum(
        ...     query_vec.get(k, 0) * doc_vec.get(k, 0)
        ...     for k in set(query_vec) | set(doc_vec)
        ... )

        >>> # Procesamiento por lotes
        >>> queries = ["query 1", "query 2", "query 3"]
        >>> query_vecs = [query_emb.embed(q) for q in queries]
        >>>
        >>> documents = ["doc 1", "doc 2", "doc 3"]
        >>> doc_vecs = [doc_emb.embed(d) for d in documents]

        >>> # Inspeccionar dimensiones dispersas (la salida está ordenada por índices)
        >>> query_vec = query_emb.embed("machine learning")
        >>> list(query_vec.items())[:5]  # Primeras 5 dimensiones (por índice)
        [(10, 0.45), (23, 0.87), (56, 0.32), (89, 1.12), (120, 0.65)]
        >>>
        >>> # Ordenar por peso para encontrar los términos más importantes
        >>> sorted_by_weight = sorted(query_vec.items(), key=lambda x: x[1], reverse=True)
        >>> top_5 = sorted_by_weight[:5]  # Top 5 términos más importantes
        >>> top_5
        [(1023, 1.45), (245, 1.23), (8901, 0.98), (5678, 0.87), (12034, 0.76)]

        >>> # Usar GPU para inferencia más rápida
        >>> sparse_emb = DefaultLocalSparseEmbedding(device="cuda")
        >>> vector = sparse_emb.embed("natural language processing")

        >>> # Ejemplo de recuperación híbrida (combinando denso + disperso)
        >>> from zvec.extension import DefaultDenseEmbedding
        >>> dense_emb = DefaultDenseEmbedding()
        >>> sparse_emb = DefaultLocalSparseEmbedding()
        >>>
        >>> query = "deep learning neural networks"
        >>> dense_vec = dense_emb.embed(query)   # [0.1, -0.3, 0.5, ...]
        >>> sparse_vec = sparse_emb.embed(query)  # {12: 0.8, 45: 1.2, ...}

        >>> # Manejo de errores
        >>> try:
        ...     sparse_emb.embed("")  # Cadena vacía
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

        >>> # Gestión de caché
        >>> # Verificar estado de la caché
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 1
        >>>
        >>> # Limpiar caché para liberar memoria
        >>> DefaultLocalSparseEmbedding.clear_cache()
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 0
        >>>
        >>> # Eliminar modelo específico de la caché
        >>> query_emb = DefaultLocalSparseEmbedding()  # Crea modelo CPU
        >>> cuda_emb = DefaultLocalSparseEmbedding(device="cuda")  # Crea modelo CUDA
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 2
        >>>
        >>> # Eliminar solo el modelo CPU
        >>> removed = DefaultLocalSparseEmbedding.remove_from_cache(device=None)
        >>> print(f"Removed: {removed}")
        True
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 1

    See Also:
        - ``SparseEmbeddingFunction``: Clase base para embeddings dispersos
        - ``DefaultDenseEmbedding``: Embedding denso con all-MiniLM-L6-v2
        - ``QwenDenseEmbedding``: Alternativa usando la API de Qwen

    References:
        - SPLADE Paper: https://arxiv.org/abs/2109.10086
        - Model: https://huggingface.co/naver/splade-cocondenser-ensembledistil
    """

    # Caché de modelos a nivel de clase: {(model_name, model_source, device): model}
    # Compartida entre todas las instancias de DefaultLocalSparseEmbedding para ahorrar memoria
    _model_cache: ClassVar[dict] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Limpia todos los modelos SPLADE de la caché de memoria.

        Esto es útil para:
        - Liberar memoria cuando los modelos ya no son necesarios
        - Forzar una recarga del modelo
        - Pruebas y depuración
                Examples:
            >>> # Limpiar caché para liberar memoria
            >>> DefaultLocalSparseEmbedding.clear_cache()

            >>> # O en pruebas para asegurar una carga fresca del modelo
            >>> def test_something():
            ...     DefaultLocalSparseEmbedding.clear_cache()
            ...     emb = DefaultLocalSparseEmbedding()
            ...     # Prueba con modelo recién cargado
        """
        cls._model_cache.clear()

    @classmethod
    def get_cache_info(cls) -> dict:
        """Obtiene información sobre los modelos actualmente en caché.

        Returns:
            dict: Diccionario con estadísticas de la caché:
                - cached_models (int): Número de instancias de modelos en caché
                - cache_keys (list): Lista de claves de caché (model_name, model_source, device)

        Examples:
            >>> info = DefaultLocalSparseEmbedding.get_cache_info()
            >>> print(f"Cached models: {info['cached_models']}")
            Cached models: 2
            >>> print(f"Cache keys: {info['cache_keys']}")
            Cache keys: [('naver/splade-cocondenser-ensembledistil', 'huggingface', None),
                        ('naver/splade-cocondenser-ensembledistil', 'huggingface', 'cuda')]
        """
        return {
            "cached_models": len(cls._model_cache),
            "cache_keys": list(cls._model_cache.keys()),
        }

    @classmethod
    def remove_from_cache(
        cls, model_source: str = "huggingface", device: Optional[str] = None
    ) -> bool:
        """Elimina un modelo específico de la caché.

        Args:
            model_source (str): Fuente del modelo ("huggingface" o "modelscope").
                Por defecto "huggingface".
            device (Optional[str]): Identificador del dispositivo. Por defecto None.

        Returns:
            bool: True si el modelo fue encontrado y eliminado, False en caso contrario.

        Examples:
            >>> # Eliminar modelo CPU de la caché
            >>> removed = DefaultLocalSparseEmbedding.remove_from_cache()
            >>> print(f"Removed: {removed}")
            True

            >>> # Eliminar modelo CUDA de la caché
            >>> removed = DefaultLocalSparseEmbedding.remove_from_cache(device="cuda")
            >>> print(f"Removed: {removed}")
            True
        """
        model_name = "naver/splade-cocondenser-ensembledistil"
        cache_key = (model_name, model_source, device)

        if cache_key in cls._model_cache:
            del cls._model_cache[cache_key]
            return True
        return False

    def __init__(
        self,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        encoding_type: Literal["query", "document"] = "query",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Inicializa con el modelo SPLADE.

        Args:
            model_source (Literal["huggingface", "modelscope"]): Fuente del modelo.
                Por defecto "huggingface".
            device (Optional[str]): Dispositivo destino ("cpu", "cuda", "mps" o None).
                Por defecto None (detección automática).
            encoding_type (Literal["query", "document"]): Tipo de codificación para embeddings.
                - "query": Optimizar para consultas de búsqueda (predeterminado)
                - "document": Optimizar para documentos indexados
                Esta distinción es importante para tareas de recuperación asimétrica.
            trust_remote_code (bool): Si se permite la ejecución de código personalizado
                del modelo desde el repositorio. Por defecto ``False``.

                .. warning::
                    Establecer esto en ``True`` permite que código Python arbitrario de un
                    repositorio de modelos descargado se ejecute en tu máquina. Solo
                    actívalo para modelos en los que confíes explícitamente.
            **kwargs: Parámetros adicionales (reservados para uso futuro).

        Raises:
            ImportError: Si sentence-transformers no está instalado.
            ValueError: Si el modelo no se puede cargar.

        Note:
            Las instancias con la misma configuración (model_source, device) compartirán
            el mismo modelo subyacente para ahorrar memoria. Las distintas instancias
            pueden usar diferentes configuraciones de encoding_type compartiendo el modelo.

            **Selección del modelo:**

            Usa ``naver/splade-cocondenser-ensembledistil`` en lugar del más reciente
            ``naver/splade-v3`` porque splade-v3 es un modelo bloqueado que requiere
            autenticación de Hugging Face. La variante cocondenser-ensembledistil:

            - No requiere autenticación ni tokens de API
            - Está disponible inmediatamente para todos los usuarios
            - Ofrece un rendimiento de recuperación comparable (~2% de diferencia)
            - Evita los errores "Access to model is restricted"

            Si necesitas splade-v3 y tienes acceso, puedes crear una subclase
            y sobreescribir el parámetro model_name.

        Examples:
            >>> # Ambas instancias comparten el mismo modelo (ahorra memoria)
            >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
            >>> doc_emb = DefaultLocalSparseEmbedding(encoding_type="document")
            >>> # Solo un modelo está cargado en memoria
        """
        # Usar el modelo SPLADE de acceso público (no requiere acceso bloqueado)
        # Nota: naver/splade-v3 requiere autenticación, por eso usamos la
        # variante cocondenser-ensembledistil, que es de acceso público
        model_name = "naver/splade-cocondenser-ensembledistil"

        # Inicializar la clase base para la carga del modelo
        SentenceTransformerFunctionBase.__init__(
            self,
            model_name=model_name,
            model_source=model_source,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        self._encoding_type = encoding_type
        self._extra_params = kwargs

        # Crear clave de caché para esta configuración de modelo
        self._cache_key = (model_name, model_source, device)

        # Cargar el modelo para asegurarse de que está disponible (usa caché si existe)
        self._get_model()

    @property
    def extra_params(self) -> dict:
        """dict: Parámetros adicionales para la personalización específica del modelo."""
        return self._extra_params

    def __call__(self, input: str) -> SparseVectorType:
        """Hace que la función de embedding sea invocable."""
        return self.embed(input)

    def embed(self, input: str) -> SparseVectorType:
        """Genera el vector de embedding disperso para el texto de entrada.

        Este método usa el modelo SPLADE para convertir el texto de entrada en una
        representación vectorial dispersa. El resultado es un diccionario donde las
        claves son índices de dimensión y los valores son pesos de importancia
        (solo se incluyen valores distintos de cero).

        El embedding se optimiza según el ``encoding_type`` especificado durante la
        inicialización: "query" para consultas de búsqueda o "document" para contenido indexado.

        Args:
            input (str): Cadena de texto de entrada a embeber. Debe ser no vacía tras
                eliminar espacios.

        Returns:
            SparseVectorType: Un diccionario que mapea índice de dimensión a peso.
                Solo se incluyen dimensiones distintas de cero. El diccionario está ordenado
                por índices (claves) de forma ascendente para una salida consistente.
                Ejemplo: ``{10: 0.5, 245: 0.8, 1023: 1.2, 5678: 0.5}``

        Raises:
            TypeError: Si ``input`` no es una cadena.
            ValueError: Si la entrada está vacía o contiene solo espacios en blanco.
            RuntimeError: Si la inferencia del modelo falla.

        Examples:
            >>> # Embedding de consulta
            >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
            >>> query_vec = query_emb.embed("machine learning")
            >>> isinstance(query_vec, dict)
            True

        Note:
            - La primera llamada puede ser más lenta por la carga del modelo
            - Las llamadas siguientes son mucho más rápidas ya que el modelo permanece en memoria
            - La aceleración GPU proporciona una mejora significativa de velocidad
            - Los vectores dispersos son eficientes en memoria (solo almacenan valores distintos de cero)
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            model = self._get_model()

            # Usar el método de codificación adecuado según el tipo
            if self._encoding_type == "document" and hasattr(model, "encode_document"):
                # Usar codificación de documento
                sparse_matrix = model.encode_document([input])
            elif hasattr(model, "encode_query"):
                # Usar codificación de consulta (predeterminado)
                sparse_matrix = model.encode_query([input])
            else:
                # Alternativa: implementación manual para sentence-transformers más antiguos
                return self._manual_sparse_encode(input)

            # Convertir la matriz dispersa a diccionario
            # SPLADE devuelve forma [1, vocab_size] para una sola entrada

            # Verificar si es una matriz dispersa (duck typing - tiene método toarray)
            if hasattr(sparse_matrix, "toarray"):
                # Matriz dispersa (CSR/CSC/etc.) - convertir a array denso
                sparse_array = sparse_matrix[0].toarray().flatten()
                sparse_dict = {
                    int(idx): float(val)
                    for idx, val in enumerate(sparse_array)
                    if val > 0
                }
            else:
                # Formato de array denso (array numpy o similar)
                if isinstance(sparse_matrix, np.ndarray):
                    sparse_array = sparse_matrix[0]
                else:
                    sparse_array = sparse_matrix

                sparse_dict = {
                    int(idx): float(val)
                    for idx, val in enumerate(sparse_array)
                    if val > 0
                }

            # Ordenar por índices (claves) para garantizar un orden consistente
            return dict(sorted(sparse_dict.items()))

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate sparse embedding: {e!s}") from e

    def _manual_sparse_encode(self, input: str) -> SparseVectorType:
        """Codificación SPLADE manual de reserva para sentence-transformers más antiguos.

        Args:
            input (str): Texto de entrada a codificar.

        Returns:
            SparseVectorType: Vector disperso como diccionario.
        """
        import torch

        model = self._get_model()

        # Tokenizar la entrada
        features = model.tokenize([input])

        # Mover al dispositivo correcto
        features = {k: v.to(model.device) for k, v in features.items()}

        # Paso hacia adelante sin gradiente
        with torch.no_grad():
            embeddings = model.forward(features)

            # Obtener logits de la salida del modelo
            # Los modelos SPLADE típicamente generan 'token_embeddings'
            if isinstance(embeddings, dict) and "token_embeddings" in embeddings:
                logits = embeddings["token_embeddings"][0]  # Primer elemento del lote
            elif hasattr(embeddings, "token_embeddings"):
                logits = embeddings.token_embeddings[0]
            # Alternativa: intentar obtener el primer valor
            elif isinstance(embeddings, dict):
                logits = next(iter(embeddings.values()))[0]
            else:
                logits = embeddings[0]

            # Aplicar activación SPLADE: log(1 + relu(x))
            relu_log = torch.log(1 + torch.relu(logits))

            # Pooling máximo sobre la dimensión del token (reducir a tamaño de vocabulario)
            if relu_log.dim() > 1:
                sparse_vec, _ = torch.max(relu_log, dim=0)
            else:
                sparse_vec = relu_log

            # Convertir a diccionario disperso (solo valores distintos de cero)
            sparse_vec_np = sparse_vec.cpu().numpy()
            sparse_dict = {
                int(idx): float(val) for idx, val in enumerate(sparse_vec_np) if val > 0
            }

            # Ordenar por índices (claves) para garantizar un orden consistente
            return dict(sorted(sparse_dict.items()))

    def _get_model(self):
        """Carga o recupera el modelo SPLADE desde la caché a nivel de clase.

        Returns:
            SentenceTransformer: La instancia del modelo SPLADE cargada.

        Raises:
            ImportError: Si los paquetes requeridos no están instalados.
            ValueError: Si el modelo no se puede cargar.

        Note:
            Los modelos se almacenan en caché a nivel de clase y se comparten entre todas
            las instancias con la misma configuración (model_name, model_source, device).
            Esto permite un uso eficiente de memoria al crear múltiples instancias
            con diferentes configuraciones de encoding_type.
        """
        # Verificar primero la caché a nivel de clase
        if self._cache_key in self._model_cache:
            return self._model_cache[self._cache_key]

        # Usar el método de la clase padre para cargar el modelo
        model = super()._get_model()

        # Almacenar el modelo en la caché a nivel de clase
        self._model_cache[self._cache_key] = model

        return model
