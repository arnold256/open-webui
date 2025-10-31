import requests
import logging
from typing import List, Optional
from langchain_core.documents import Document
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class ExternalTextSplitter:
    def __init__(
        self,
        url: str,
        api_key: str = "",
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using an external splitter service via HTTP POST.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
            
        Raises:
            Exception: If the external service request fails
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.url
        if url.endswith("/"):
            url = url[:-1]

        try:
            response = requests.post(
                url,
                json={
                    "documents": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in documents
                    ],
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Error connecting to external text splitter: {e}")
            raise Exception(f"Error connecting to external text splitter: {e}")

        response_data = response.json()
        if response_data:
            if isinstance(response_data, dict):
                # Handle dict with "chunks" key
                if "chunks" in response_data:
                    chunks = response_data.get("chunks", [])
                    documents = []
                    for chunk in chunks:
                        documents.append(
                            Document(
                                page_content=chunk.get("page_content"),
                                metadata=chunk.get("metadata"),
                            )
                        )
                    return documents
                else:
                    raise Exception("Error loading document: Unable to parse content")
            elif isinstance(response_data, list):
                # Handle direct list of chunks
                documents = []
                for chunk in response_data:
                    documents.append(
                        Document(
                            page_content=chunk.get("page_content"),
                            metadata=chunk.get("metadata"),
                        )
                    )
                return documents
            else:
                raise Exception("Error loading document: Unable to parse content")
        else:
            raise Exception("Error loading document: Empty response")
