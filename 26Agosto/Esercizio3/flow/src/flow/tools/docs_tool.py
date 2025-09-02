from pydantic import Field
from crewai.tools import BaseTool
from typing import List, Optional
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from pathlib import Path
from typing import List
import os
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup

class RagToolSphinx(BaseTool):
    """
    Tool CrewAI per Retrieval-Augmented Generation (RAG) su documenti Sphinx.
    """
    
    name: str = "RAG Tool Sphinx"
    description: str = (
        "Tool CrewAI che implementa RAG con FAISS sui documenti HTML generati da Sphinx. "
        "Carica documenti HTML dalla directory sphinx_path e risponde alle domande citando le fonti."
    )

    load_dotenv()

    sphinx_path: str = Field(default="docs/build")
    persist_dir_sphinx: Path = Field(default=Path("faiss_index_sphinx"))

    embeddings: Optional[AzureOpenAIEmbeddings] = None
    llm: Optional[object] = None
    vector_store_sphinx: Optional[FAISS] = None
    retriever_sphinx: Optional[object] = None
    chain: Optional[object] = None

    def __init__(self, **data):
        super().__init__(**data)

        # Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            model=os.getenv("EMBEDDING_MODEL")
        )

        # LLM
        self.llm = init_chat_model(
            os.getenv("LLM_MODEL"),
            model_provider="azure_openai",
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )

        # Test connessioni
        try:
            test_emb = self.embeddings.embed_query("test")
            print("Connessione embedding OK")
        except Exception as e:
            print("Errore embedding:", e)

        try:
            response = self.llm.invoke("Ciao!")
            print("Connessione LLM OK")
        except Exception as e:
            print("Errore LLM:", e)

        # Carica o costruisci FAISS Sphinx
        docs = self._load_sphinx_documents_bs()
        self.vector_store_sphinx = self._load_or_build_sphinx_vectorstore(docs)
        self.retriever_sphinx = self._make_retriever(self.vector_store_sphinx)
        try:
            print(f"RagToolSphinx: indicizzati {len(docs)} documenti HTML")
        except Exception:
            pass

        # Catena RAG
        self.chain = self._build_chain()

    def _load_sphinx_documents_bs(self) -> List[Document]:
        documents: List[Document] = []
        folder = Path(self.sphinx_path)

        for file_path in folder.glob("**/*.html"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    html = f.read()
            except Exception:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Preferisci il contenuto nel body
            container = soup.body or soup

            current_title: Optional[str] = None
            current_content: List[str] = []
            default_title = file_path.stem

            def flush_section():
                nonlocal current_title, current_content
                if current_title and current_content:
                    documents.append(Document(
                        page_content="\n".join(current_content).strip(),
                        metadata={"title": current_title.strip(), "source": file_path.name}
                    ))
                current_content = []

            # Considera una gamma più ampia di tag testuali
            selectable_tags = [
                "h1", "h2", "h3", "h4", "h5", "h6",
                "p", "li", "dd", "dt", "pre", "code"
            ]

            # Aggiungi anche il testo delle celle delle tabelle
            for el in container.find_all(selectable_tags + ["td", "th"], recursive=True):
                name = el.name.lower()
                if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                    # Nuova sezione: emetti la precedente
                    flush_section()
                    current_title = el.get_text(" ", strip=True)
                else:
                    # Se non c'è ancora un titolo, usa il nome file come titolo di default
                    if not current_title:
                        current_title = default_title
                    text = el.get_text(" ", strip=True)
                    if text:
                        current_content.append(text)

            # Flush finale per l'ultima sezione
            flush_section()

            # Fallback: nessuna sezione estratta, usa tutto il testo della pagina
            if not any(d.metadata.get("source") == file_path.name for d in documents):
                full_text = soup.get_text(" ", strip=True)
                if full_text:
                    documents.append(Document(
                        page_content=full_text,
                        metadata={"title": default_title, "source": file_path.name}
                    ))

        return documents

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        return splitter.split_documents(docs)

    def _build_vectorstore(self, chunks: List[Document]) -> FAISS:
        vs = FAISS.from_documents(chunks, self.embeddings)
        self.persist_dir_sphinx.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self.persist_dir_sphinx))
        return vs

    def _load_or_build_sphinx_vectorstore(self, docs: List[Document]) -> FAISS:
        index_file = self.persist_dir_sphinx / "index.faiss"
        meta_file = self.persist_dir_sphinx / "index.pkl"
        if index_file.exists() and meta_file.exists():
            return FAISS.load_local(
                str(self.persist_dir_sphinx),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        chunks = self._split_documents(docs)
        return self._build_vectorstore(chunks)

    def _make_retriever(self, vector_store: FAISS):
        return vector_store.as_retriever(search_kwargs={"k": 12})

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
                    self.retriever_sphinx.get_relevant_documents(q) if self.retriever_sphinx else []
                )
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _run(self, query: str) -> str:
        return self.chain.invoke(query)
