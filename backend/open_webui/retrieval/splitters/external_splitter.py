import logging
from typing import List, Optional

import requests
from langchain_core.documents import Document
from open_webui.env import REQUESTS_VERIFY
from open_webui.utils.headers import include_user_info_headers, parse_custom_headers

log = logging.getLogger(__name__)


class ExternalTextSplitter:
    """Delegate chunking to an external HTTP service.

    Exposes the same ``split_documents`` surface as the LangChain splitters so the
    dispatch in ``save_docs_to_vector_db`` stays uniform. Like ``ExternalDocumentLoader``
    this is synchronous: callers already run it inside a worker thread.
    """

    def __init__(
        self,
        url: str,
        api_key: str = '',
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        timeout: Optional[int] = None,
        user=None,
        user_groups=None,
        headers=None,
        metadata=None,
    ) -> None:
        self.url = url
        self.api_key = api_key

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timeout = timeout

        self.user = user
        self.user_groups = user_groups
        self.headers = headers
        self.metadata = metadata

    def _request_headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        headers.update(parse_custom_headers(self.headers, self.user, self.metadata, user_groups=self.user_groups))

        if self.user is not None:
            headers = include_user_info_headers(headers, self.user)

        return headers

    def _split_document(self, document: Document, headers: dict) -> List[Document]:
        payload = {
            'documents': [
                {
                    'page_content': document.page_content,
                    'metadata': document.metadata,
                }
            ],
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }

        try:
            response = requests.post(
                self.url, json=payload, headers=headers, verify=REQUESTS_VERIFY, timeout=self.timeout
            )
        except Exception as e:
            log.error(f'Error connecting to endpoint: {e}')
            raise Exception(f'Error connecting to endpoint: {e}')

        if not response.ok:
            raise Exception(f'Error splitting document: {response.status_code} {response.text}')

        try:
            response_data = response.json()
        except Exception as e:
            raise Exception(f'Error splitting document: Unable to parse response: {e}')

        if isinstance(response_data, dict):
            chunks = response_data.get('chunks')
        elif isinstance(response_data, list):
            chunks = response_data
        else:
            chunks = None

        if not isinstance(chunks, list):
            raise Exception('Error splitting document: Unable to parse chunks')

        documents = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise Exception('Error splitting document: Unable to parse chunks')

            documents.append(
                Document(
                    page_content=chunk.get('page_content') or '',
                    # Fall back to the source metadata so chunks stay attributable to
                    # their file even if the service does not echo it back.
                    metadata={**document.metadata, **(chunk.get('metadata') or {})},
                )
            )

        if not documents:
            raise Exception('Error splitting document: No chunks returned')

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        headers = self._request_headers()

        # One request per document: `save_docs_to_vector_db` is also called with
        # documents from several files at once, and per-document requests are what
        # keep each returned chunk attributable to the file it came from.
        split_documents = []
        for document in documents:
            split_documents.extend(self._split_document(document, headers))

        return split_documents
