import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")

def validate_env_variables():
    print("Validating environment variables...")
    for env_value in (PDF_PATH, OPENROUTER_EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PG_VECTOR_COLLECTION_NAME, DATABASE_URL):
        if not env_value:
            raise RuntimeError(f"Environment variable {env_value} is not set")

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF file does not exist: {PDF_PATH}")

def load_document() -> list[Document]:
    print("Loading document...")
    document = PyPDFLoader(str(PDF_PATH)).load()
    return document

def split_document(document: list[Document]) -> list[Document]:
    print("Splitting document...")
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(document)
    return splits

def enrich_splits(splits: list[Document]) -> list[Document]:
    print("Enriching splits...")
    enriched = []
    for split in splits:
        metadata = {}
        for metadata_key, metadata_value in split.metadata.items():
            if metadata_value not in ("", None):
                metadata[metadata_key] = metadata_value

        doc = Document(
            page_content=split.page_content,
            metadata=metadata,
        )
        enriched.append(doc)
    return enriched

def add_document_to_pgvector(enriched: list[Document]):
    print("Adding documents to PGVector...")
    print(f"Using embedding model: {OPENROUTER_EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        model=OPENROUTER_EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,        
        model_kwargs={"encoding_format": "float"},
        chunk_size=128,
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    ids = [f"doc-{i}" for i in range(len(enriched))]
    store.add_documents(documents=enriched, ids=ids)

def ingest_pdf():
    print("Starting PDF ingestion...")
    validate_env_variables()
    document = load_document()
    splits = split_document(document)
    enriched = enrich_splits(splits)
    add_document_to_pgvector(enriched)
    print("PDF ingestion completed successfully.")

if __name__ == "__main__":
    ingest_pdf()