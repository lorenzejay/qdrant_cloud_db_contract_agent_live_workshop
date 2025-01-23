import json
import os
from typing import Any, Optional, Type

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from qdrant_client.http.models import Distance, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = Any  # type placeholder
    Filter = Any
    FieldCondition = Any
    MatchValue = Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QdrantToolSchema(BaseModel):
    """Input for QdrantTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Qdrant database. Pass only the query, not the question.",
    )
    filter_by: Optional[str] = Field(
        default=None,
        description="Filter by properties. Pass only the properties, not the question.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Filter by value. Pass only the value, not the question.",
    )


class QdrantVectorSearchTool(BaseTool):
    """Tool to query, and if needed filter results from a Qdrant database"""

    model_config = {"arbitrary_types_allowed": True}
    client: QdrantClient = None
    name: str = "QdrantVectorSearchTool"
    description: str = "A tool to search the Qdrant database for relevant information on internal documents."
    args_schema: Type[BaseModel] = QdrantToolSchema
    query: Optional[str] = None
    filter_by: Optional[str] = None
    filter_value: Optional[str] = None
    collection_name: Optional[str] = None
    limit: Optional[int] = Field(default=3)
    score_threshold: float = Field(default=0.35)
    qdrant_url: str = Field(
        ...,
        description="The URL of the Qdrant server",
    )
    qdrant_api_key: str = Field(
        ...,
        description="The API key for the Qdrant server",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        self.client.set_model("BAAI/bge-small-en-v1.5")

    def _run(
        self,
        query: str,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                "Please install it with: pip install qdrant-client"
            )

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set")

        # Create filter if filter parameters are provided
        search_filter = None
        if filter_by and filter_value:
            search_filter = Filter(
                must=[
                    FieldCondition(key=filter_by, match=MatchValue(value=filter_value))
                ]
            )

        # Search in Qdrant using the built-in query method
        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=[query],
            query_filter=search_filter,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )

        # Format results similar to storage implementation
        results = []
        for point in search_results:
            result = {
                "id": point.id,
                "metadata": point.metadata,
                "context": point.document,
                "score": point.score,
            }
            results.append(result)

        return json.dumps(results, indent=2)


if __name__ == "__main__":
    tool = QdrantVectorSearchTool(
        qdrant_url=os.environ.get("QDRANT_URL"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        collection_name="netflix_data_system_2",
    )
    result = tool.run("Find me similar shows to How I Met Your Mother?")
    print("result", result)
