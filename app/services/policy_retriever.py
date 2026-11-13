from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "docs" / "policies" / "academy_policy.md"
PolicyRetriever = Runnable[str, list[Document]]

# 조항 순서에 대응하는 가짜 검색용 키워드다.
POLICY_TERMS = {
    "1": ("환불", "수강료", "결제", "보상"),
    "2": ("결강", "보강", "휴강", "수업 미진행"),
    "3": ("반 변경", "수강 연기", "시간표", "반 이동"),
    "4": ("출결", "지각", "결석", "출석"),
    "5": ("수료", "재수강", "이수"),
    "6": ("강사 변경", "강사 태도"),
    "7": ("수업 자료", "녹화", "다시보기"),
    "8": ("실습 환경", "장비", "설치 오류"),
    "9": ("개인정보", "계정 삭제", "정보 유출"),
    "10": ("민원", "이의 제기", "쿠폰"),
}


def _policy_documents() -> list[Document]:
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "article")], strip_headers=True
    ).split_text(POLICY_PATH.read_text(encoding="utf-8"))
    articles = [section for section in sections if "article" in section.metadata]
    return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100).split_documents(articles)


def build_fake_policy_retriever() -> PolicyRetriever:
    documents = _policy_documents()

    async def retrieve(query: str) -> list[Document]:
        matches = [
            document
            for document in documents
            if any(
                term in query
                for term in POLICY_TERMS[document.metadata["article"].split(".", 1)[0]]
            )
        ]
        return matches[:3]

    return RunnableLambda(retrieve)


def build_chroma_policy_retriever(settings: Settings) -> PolicyRetriever:
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    documents = _policy_documents()
    document_ids = [
        sha256(f"{doc.metadata['article']}\n{doc.page_content}".encode()).hexdigest() for doc in documents
    ]
    index_version = sha256("\n".join([settings.openai_embedding_model, *document_ids]).encode()).hexdigest()[:16]
    store = Chroma(
        collection_name=f"academy-policy-{index_version}",
        embedding_function=OpenAIEmbeddings(
            model=settings.openai_embedding_model, api_key=settings.openai_api_key, timeout=30, max_retries=1
        ),
        persist_directory=str(PROJECT_ROOT / settings.chroma_persist_directory),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    existing = set(store.get(include=[])["ids"])
    missing = [(doc_id, doc) for doc_id, doc in zip(document_ids, documents, strict=True) if doc_id not in existing]
    if missing:
        store.add_documents([doc for _, doc in missing], ids=[doc_id for doc_id, _ in missing])
    return store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.35})
