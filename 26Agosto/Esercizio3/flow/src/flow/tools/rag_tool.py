from pydantic import Field
from crewai.tools import BaseTool
from typing import List, Optional
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

class RagTool(BaseTool):
    name: str = "RAG Tool"
    description: str = (
        "Uno strumento CrewAI che implementa Retrieval-Augmented Generation (RAG) "
        "con persistenza FAISS su disco. Carica documenti locali (.txt, .md), "
        "e risponde alle domande citando le fonti."
    )

    load_dotenv()

    # Campi input Pydantic
    documents_path: str = Field(default="docs")
    persist_dir: Path = Field(default=Path("faiss_index_docs"))

    embeddings: Optional[AzureOpenAIEmbeddings] = None
    llm: Optional[object] = None
    vector_store: Optional[FAISS] = None
    retriever: Optional[object] = None
    chain: Optional[object] = None

    def __init__(self, **data):
        super().__init__(**data)

        # 1️⃣ Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            model=os.getenv("EMBEDDING_MODEL")
        )
        # 2️⃣ LLM
        self.llm = init_chat_model(
            os.getenv("LLM_MODEL"),
            model_provider="azure_openai",
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )
        try:
        # Test embedding: genera embedding per una stringa di esempio
            test_emb = self.embeddings.embed_query("test")
            print("Connessione embedding OK")
        except Exception as e:
            print("Errore embedding:", e)

        try:
            # Test LLM: genera una risposta di esempio
            response = self.llm.invoke("Ciao!")
            print("Connessione LLM OK")
        except Exception as e:
            print("Errore LLM:", e)

        # 3️⃣ Carica o costruisce FAISS
        docs = self._load_documents()
        self.vector_store = self._load_or_build_vectorstore(docs)
        self.retriever = self._make_retriever(self.vector_store)

        # 4️⃣ Catena RAG
        self.chain = self._build_chain()


    def _load_documents(self) -> List[Document]:
        folder = Path(self.documents_path)
        documents: List[Document] = []
        for file_path in folder.glob("**/*"):
            if file_path.suffix.lower() not in [".txt", ".md"]:
                continue
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
        return documents

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        return splitter.split_documents(docs)

    def _build_vectorstore(self, chunks: List[Document]) -> FAISS:
        vs = FAISS.from_documents(chunks, self.embeddings)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self.persist_dir))
        return vs

    def _load_or_build_vectorstore(self, docs: List[Document]) -> FAISS:
        index_file = self.persist_dir / "index.faiss"
        meta_file = self.persist_dir / "index.pkl"
        if index_file.exists() and meta_file.exists():
            return FAISS.load_local(
                str(self.persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        chunks = self._split_documents(docs)
        return self._build_vectorstore(chunks)

    def _make_retriever(self, vector_store: FAISS):
        return vector_store.as_retriever(search_kwargs={"k": 5})


    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            [f"[source:{d.metadata.get('source','doc')}] {d.page_content}" for d in docs]
        )

    def _build_chain(self):
        system_prompt = (
            "Sei un assistente esperto. Rispondi in italiano. "
            "Usa solo le informazioni nei documenti forniti. "
            "Se non trovi la risposta, scrivi: 'Non è presente nel contesto fornito.'"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Domanda:\n{question}\n\nContesto:\n{context}")
        ])

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda q: self._format_docs(
                    self.retriever.get_relevant_documents(q) if self.retriever else []
                )
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _run(self, query: str) -> str:
        return self.chain.invoke(query)
