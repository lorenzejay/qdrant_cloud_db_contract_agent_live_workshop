import os
from qdrant_client import QdrantClient
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()
# Setup Qdrant client
COLLECTION_NAME = "contracts_business_2"
doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])  # Allow PDF format
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
client.set_model("sentence-transformers/all-MiniLM-L6-v2")
# client.set_sparse_model("Qdrant/bm25")
# Define the folder where PDFs are stored
pdf_folder = "knowledge/contracts/"
# Initialize documents and metadata lists
documents, metadatas = [], []
# Loop through all PDFs in the folder and process them
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processing {pdf_path}")

        result = doc_converter.convert(pdf_path)

        # Chunk the converted document
        for chunk in HybridChunker().chunk(result.document):
            print("chunk", chunk)
            documents.append(chunk.text)
            metadatas.append(chunk.meta.export_json_dict())
# Upload the documents to Qdrant
_ = client.add(
    collection_name=COLLECTION_NAME,
    documents=documents,
    metadata=metadatas,
    batch_size=64,
)
# Retrieve and print results from Qdrant
points = client.query(
    collection_name=COLLECTION_NAME,
    query_text="What is the grants to rights of digital cinema destinations corp?",
    limit=10,
)

for i, point in enumerate(points):
    print(f"=== {i} ===")
    print(point.document)
    print()
