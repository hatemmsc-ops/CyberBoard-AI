import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SEC_FILINGS_DIR = DATA_DIR / "sec_filings"
CHROMA_DIR = DATA_DIR / "chromadb"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
SEC_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "Hatem Isa applehtm777@gmail.com")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_DIM = 1536
HYBRID_ALPHA = 0.7  # weight for dense vs sparse (1.0 = all dense, 0.0 = all sparse)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "sec_filings"

COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "JNJ": "0000200406",
    "V": "0001403161",
    "WMT": "0000104169",
    "PG": "0000080424",
    "MA": "0001141391",
    "UNH": "0000731766",
    "KO": "0000021344",
    "PFE": "0000078003",
    "CVX": "0000093410",
    "CSCO": "0000858877",
    "INTC": "0000050863",
    "GS": "0000886982",
}

def gemini_available():
    return bool(GEMINI_API_KEY)
