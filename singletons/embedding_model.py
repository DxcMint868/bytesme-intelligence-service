from sentence_transformers import SentenceTransformer


_embedding_model = None

VIETNAMESE_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"


def get_embedding_model():
    """
    Get or create a cached SentenceTransformer model optimized for Vietnamese.
    This prevents reloading the model multiple times during execution.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            print(
                f"Load pretrained SentenceTransformer: {VIETNAMESE_MODEL}")
            _embedding_model = SentenceTransformer(
                VIETNAMESE_MODEL, device="cpu")
            # fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # _model = SentenceTransformer(fallback_model, device="cpu")
            print(f"Successfully loaded model: {VIETNAMESE_MODEL}")
        except Exception as e:
            fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(
                f"Failed to load {VIETNAMESE_MODEL}, falling back to {fallback_model}: {e}")
            _embedding_model = SentenceTransformer(
                fallback_model, device="cpu")
    return _embedding_model
