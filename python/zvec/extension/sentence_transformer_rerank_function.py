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

from ..model.doc import Doc
from ..tool import require_module
from .rerank_function import RerankFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalReRanker(SentenceTransformerFunctionBase, RerankFunction):
    """Re-clasificador usando modelos cross-encoder de Sentence Transformer para re-clasificación semántica.

    Este re-clasificador aprovecha modelos cross-encoder preentrenados para realizar
    una re-clasificación semántica profunda de los resultados de búsqueda. Se ejecuta
    localmente sin llamadas a API, soporta aceleración GPU y funciona con modelos de
    Hugging Face o ModelScope.

    Los modelos cross-encoder evalúan pares consulta-documento de forma conjunta,
    proporcionando puntuaciones de relevancia más precisas que la similitud basada en
    bi-encoder (embedding).

    Args:
        query (str): Texto de consulta para la re-clasificación semántica. **Requerido**.
        topn (int, optional): Número máximo de documentos a devolver tras la re-clasificación.
            Por defecto 10.
        rerank_field (Optional[str], optional): Nombre del campo del documento a usar como
            texto de entrada para la re-clasificación. **Requerido** (p.ej., "content", "title", "body").
        model_name (str, optional): Identificador del modelo cross-encoder o ruta local.
            Por defecto ``"cross-encoder/ms-marco-MiniLM-L6-v2"`` (MS MARCO MiniLM).
            Opciones comunes:
            - ``"cross-encoder/ms-marco-MiniLM-L6-v2"``: Ligero, rápido (~80MB, recomendado)
            - ``"cross-encoder/ms-marco-MiniLM-L12-v2"``: Mejor precisión (~120MB)
            - ``"BAAI/bge-reranker-base"``: BGE Reranker Base (~280MB)
            - ``"BAAI/bge-reranker-large"``: BGE Reranker Large (máxima calidad, ~560MB)
        model_source (Literal["huggingface", "modelscope"], optional): Fuente del modelo.
            Por defecto ``"huggingface"``.
            - ``"huggingface"``: Cargar desde Hugging Face Hub
            - ``"modelscope"``: Cargar desde ModelScope (recomendado para usuarios en China)
        device (Optional[str], optional): Dispositivo en el que ejecutar el modelo.
            Opciones: ``"cpu"``, ``"cuda"``, ``"mps"`` (para Apple Silicon) o ``None``
            para detección automática. Por defecto ``None``.
        batch_size (int, optional): Tamaño de lote para procesar pares consulta-documento.
            Valores más grandes aceleran el proceso pero usan más memoria. Por defecto ``32``.

    Attributes:
        query (str): El texto de consulta usado para la re-clasificación.
        topn (int): Número máximo de documentos a devolver.
        rerank_field (Optional[str]): Nombre del campo usado como entrada para re-clasificación.
        model_name (str): El modelo cross-encoder en uso.
        model_source (str): La fuente del modelo ("huggingface" o "modelscope").
        device (str): El dispositivo en el que se ejecuta el modelo.

    Raises:
        ValueError: Si ``query`` está vacío/es None, ``rerank_field`` es None
            o el modelo no se puede cargar.
        TypeError: Si los tipos de entrada no son válidos.
        RuntimeError: Si la inferencia del modelo falla.

    Note:
        - Requiere Python 3.10, 3.11 o 3.12
        - Requiere el paquete ``sentence-transformers``: ``pip install sentence-transformers``
        - Para soporte de ModelScope, también requiere: ``pip install modelscope``
        - La primera ejecución descarga el modelo (~80-560MB según el modelo) desde la fuente elegida
        - No se requieren claves de API ni red tras la descarga inicial
        - Los cross-encoders son más lentos que los bi-encoders pero más precisos
        - La aceleración GPU proporciona una mejora significativa de velocidad (5-10x)

        **Modelo MS MARCO MiniLM-L6-v2 (predeterminado):**

        El modelo por defecto ``cross-encoder/ms-marco-MiniLM-L6-v2`` es un cross-encoder
        ligero y eficiente entrenado en el dataset MS MARCO. Ofrece:

        - Velocidad de inferencia rápida (adecuada para aplicaciones en tiempo real)
        - Tamaño de modelo pequeño (~80MB, descarga rápida)
        - Buen equilibrio entre velocidad y precisión
        - Entrenado en más de 500K pares consulta-documento
        - Disponibilidad pública sin autenticación

        **Para usuarios en China:**

        Si encuentras problemas de acceso a Hugging Face, usa ModelScope:

        .. code-block:: python

            # Recomendado para usuarios en China
            reranker = SentenceTransformerReRanker(
                query="机器学习算法",
                rerank_field="content",
                model_source="modelscope"
            )

        Alternativamente, usa el espejo de Hugging Face:

        .. code-block:: bash

            export HF_ENDPOINT=https://hf-mirror.com

    Examples:
        >>> # Uso básico con el modelo MS MARCO MiniLM predeterminado
        >>> from zvec.extension import SentenceTransformerReRanker
        >>>
        >>> reranker = SentenceTransformerReRanker(
        ...     query="machine learning algorithms",
        ...     topn=5,
        ...     rerank_field="content"
        ... )
        >>>
        >>> # Usar en collection.query()
        >>> results = collection.query(
        ...     data={"vector_field": query_vector},
        ...     reranker=reranker,
        ...     topk=20
        ... )

        >>> # Usando ModelScope para usuarios en China
        >>> reranker = SentenceTransformerReRanker(
        ...     query="深度学习",
        ...     topn=10,
        ...     rerank_field="content",
        ...     model_source="modelscope"
        ... )

        >>> # Usando modelo más grande para mejor calidad
        >>> reranker = SentenceTransformerReRanker(
        ...     query="neural networks",
        ...     topn=5,
        ...     rerank_field="content",
        ...     model_name="BAAI/bge-reranker-large",
        ...     device="cuda",
        ...     batch_size=64
        ... )

        >>> # Llamada directa de rerank (para pruebas)
        >>> query_results = {
        ...     "vector1": [
        ...         Doc(id="1", score=0.9, fields={"content": "Machine learning is..."}),
        ...         Doc(id="2", score=0.8, fields={"content": "Deep learning is..."}),
        ...     ]
        ... }
        >>> reranked = reranker.rerank(query_results)
        >>> for doc in reranked:
        ...     print(f"ID: {doc.id}, Score: {doc.score:.4f}")
        ID: 2, Score: 0.9234
        ID: 1, Score: 0.8567

    See Also:
        - ``RerankFunction``: Clase base abstracta para re-clasificadores
        - ``QwenReRanker``: Re-clasificador usando la API de Qwen
        - ``RrfReRanker``: Re-clasificador multi-vector usando RRF
        - ``WeightedReRanker``: Re-clasificador multi-vector usando puntuaciones ponderadas

    References:
        - MS MARCO Cross-Encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
        - BGE Reranker: https://huggingface.co/BAAI/bge-reranker-base
        - Cross-Encoder vs Bi-Encoder: https://www.sbert.net/examples/applications/cross-encoder/README.html
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        batch_size: int = 32,
        trust_remote_code: bool = False,
    ):
        """Inicializa SentenceTransformerReRanker con consulta y configuración.

        Args:
            query (Optional[str]): Texto de consulta para la correspondencia semántica. Requerido.
            topn (int): Número de resultados principales a devolver.
            rerank_field (Optional[str]): Campo del documento para la entrada de re-clasificación.
            model_name (str): Identificador del modelo cross-encoder.
            model_source (Literal["huggingface", "modelscope"]): Fuente del modelo.
            device (Optional[str]): Dispositivo destino ("cpu", "cuda", "mps" o None).
            batch_size (int): Tamaño de lote para procesar pares consulta-documento.
            trust_remote_code (bool): Si se permite la ejecución de código personalizado
                del modelo desde el repositorio. Por defecto ``False``.

                .. warning::
                    Establecer esto en ``True`` permite que código Python arbitrario de un
                    repositorio de modelos descargado se ejecute en tu máquina. Solo
                    actívalo para modelos en los que confíes explícitamente.

        Raises:
            ValueError: Si la consulta está vacía o el modelo no se puede cargar.
        """
        # Inicializar la clase base para la carga del modelo
        SentenceTransformerFunctionBase.__init__(
            self,
            model_name=model_name,
            model_source=model_source,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        # Inicializar la función de re-clasificación
        RerankFunction.__init__(self, topn=topn, rerank_field=rerank_field)

        # Validar la consulta
        if not query:
            raise ValueError("Query is required for DefaultLocalReRanker")
        self._query = query
        self._batch_size = batch_size

        # Cargar y validar el modelo cross-encoder
        model = self._get_model()
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model '{model_name}' does not appear to be a cross-encoder model. "
                "Cross-encoder models should have a 'predict' method."
            )
        self._model = model

    def _get_model(self):
        """Carga o recupera el modelo CrossEncoder.

        Este método sobreescribe el de la clase base para cargar CrossEncoder en lugar
        de SentenceTransformer, ya que la re-clasificación requiere modelos cross-encoder.

        Returns:
            CrossEncoder: La instancia del modelo cross-encoder cargada.

        Raises:
            ImportError: Si los paquetes requeridos no están instalados.
            ValueError: Si el modelo no se puede cargar.
        """
        # Retornar el modelo en caché si existe
        if self._model is not None:
            return self._model

        # Cargar el modelo cross-encoder
        try:
            sentence_transformers = require_module("sentence_transformers")

            if self._model_source == "modelscope":
                # Cargar desde ModelScope
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                # Descargar el modelo a caché
                model_dir = snapshot_download(self._model_name)

                # Cargar CrossEncoder desde ruta local
                model = sentence_transformers.CrossEncoder(
                    model_dir, device=self._device
                )
            else:
                # Cargar CrossEncoder desde Hugging Face (predeterminado)
                model = sentence_transformers.CrossEncoder(
                    self._model_name,
                    device=self._device,
                    trust_remote_code=self._trust_remote_code,
                )

            return model

        except ImportError as e:
            if "modelscope" in str(e) and self._model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load CrossEncoder model '{self._model_name}' "
                f"from {self._model_source}: {e!s}"
            ) from e

    @property
    def query(self) -> str:
        """str: Texto de consulta usado para la re-clasificación semántica."""
        return self._query

    @property
    def batch_size(self) -> int:
        """int: Tamaño de lote para procesar pares consulta-documento."""
        return self._batch_size

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-clasifica documentos usando el modelo cross-encoder de Sentence Transformer.

        Evalúa cada par consulta-documento usando el modelo cross-encoder para calcular
        puntuaciones de relevancia. Los documentos se ordenan luego por estas puntuaciones
        y se devuelven los k mejores resultados.

        Args:
            query_results (dict[str, list[Doc]]): Mapeo de nombres de campos vectoriales
                a listas de documentos recuperados. Los documentos de todos los campos se
                deduplicaran y re-clasificarán juntos.

        Returns:
            list[Doc]: Documentos re-clasificados (hasta ``topn``) con campos ``score``
                actualizados que contienen las puntuaciones de relevancia del modelo cross-encoder.

        Raises:
            ValueError: Si no se encuentran documentos válidos o la inferencia del modelo falla.

        Note:
            - Los documentos duplicados (mismo ID) entre campos se procesan una vez
            - Los documentos con contenido vacío o ausente en ``rerank_field`` se omiten
            - Las puntuaciones devueltas son logits del modelo cross-encoder
            - Las puntuaciones más altas indican mayor relevancia
            - El tiempo de procesamiento es O(n) donde n es el número de documentos

        Examples:
            >>> reranker = SentenceTransformerReRanker(
            ...     query="machine learning",
            ...     topn=3,
            ...     rerank_field="content"
            ... )
            >>> query_results = {
            ...     "vector1": [
            ...         Doc(id="1", score=0.9, fields={"content": "ML basics"}),
            ...         Doc(id="2", score=0.8, fields={"content": "DL tutorial"}),
            ...     ]
            ... }
            >>> reranked = reranker.rerank(query_results)
            >>> len(reranked) <= 3
            True
        """
        if not query_results:
            return []

        # Recopilar y deduplicar documentos
        id_to_doc: dict[str, Doc] = {}
        doc_ids: list[str] = []
        contents: list[str] = []

        for _, query_result in query_results.items():
            for doc in query_result:
                doc_id = doc.id
                if doc_id in id_to_doc:
                    continue

                # Extraer contenido de texto del campo especificado
                field_value = doc.field(self.rerank_field)
                rank_content = str(field_value).strip() if field_value else ""
                if not rank_content:
                    continue

                id_to_doc[doc_id] = doc
                doc_ids.append(doc_id)
                contents.append(rank_content)

        if not contents:
            raise ValueError("No documents to rerank")

        try:
            # Usar el método predict estándar del cross-encoder
            pairs = [[self.query, content] for content in contents]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convertir a lista de flotantes si es necesario
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = [float(s) for s in scores]

        except Exception as e:
            raise RuntimeError(f"Failed to compute rerank scores: {e!s}") from e

        # Crear documentos con puntuación
        scored_docs = [
            (doc_ids[i], id_to_doc[doc_ids[i]], scores[i]) for i in range(len(doc_ids))
        ]

        # Ordenar por puntuación (descendente) y tomar los k mejores
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        top_scored_docs = scored_docs[: self.topn]

        # Construir lista de resultados con puntuaciones actualizadas
        results: list[Doc] = []
        for _, doc, score in top_scored_docs:
            new_doc = doc._replace(score=score)
            results.append(new_doc)

        return results
