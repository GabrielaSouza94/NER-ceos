import os

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def _resolve_device(device=None):
    """Auto-detecta GPU (CUDA > MPS > CPU). Respeita CUDA_VISIBLE_DEVICES."""
    if device is not None:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _device_info(device):
    """Descreve o device incluindo a GPU física fixada por CUDA_VISIBLE_DEVICES."""
    if device != "cuda":
        return device
    visivel = os.environ.get("CUDA_VISIBLE_DEVICES", "(todas)")
    nome = "?"
    try:
        import torch
        nome = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return f"cuda | CUDA_VISIBLE_DEVICES={visivel} | {nome}"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name, device=None, normalize=True):
        device = _resolve_device(device)
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        run_tag = os.environ.get("RUN_TAG", "")
        prefixo = f"[{run_tag}] " if run_tag else ""
        print(f"  {prefixo}[Embeddings em: {_device_info(device)}]")

    def embed_documents(self, texts):
        return self._model.encode(
            texts, normalize_embeddings=self._normalize
        ).tolist()

    def embed_query(self, text):
        return self._model.encode(
            [text], normalize_embeddings=self._normalize
        ).tolist()[0]


def create_embeddings(chunks, model="bge-m3", base_url=None):
    if "/" in model:
        embeddings = LocalEmbeddings(model_name=model)
    else:
        kwargs = {"model": model}
        if base_url is not None:
            kwargs["base_url"] = base_url
        embeddings = OllamaEmbeddings(**kwargs)
    return Chroma.from_texts(chunks, embeddings)