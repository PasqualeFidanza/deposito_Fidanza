# """
# Modulo per il RagTool - Strumento CrewAI per Retrieval-Augmented Generation.

# Questo modulo implementa un tool CrewAI che utilizza tecniche di RAG per rispondere
# alle domande basandosi su documenti locali. Il tool carica documenti, crea embeddings,
# costruisce un indice FAISS e utilizza un LLM per generare risposte contestualizzate.
# """

# from pydantic import Field
# from crewai.tools import BaseTool
# from typing import List, Optional
# from pathlib import Path
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain.chat_models import init_chat_model
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import os
# from dotenv import load_dotenv

# class RagTool(BaseTool):
#     """
#     Tool CrewAI per Retrieval-Augmented Generation (RAG) con persistenza FAISS.
    
#     Questo tool implementa un sistema RAG completo che:
#     1. Carica documenti locali (.txt, .md) da una directory specificata
#     2. Crea embeddings utilizzando Azure OpenAI
#     3. Costruisce e persiste un indice FAISS per la ricerca semantica
#     4. Utilizza un LLM per generare risposte basate sui documenti recuperati
    
#     Attributes:
#         name (str): Nome del tool
#         description (str): Descrizione del tool per CrewAI
#         documents_path (str): Percorso della directory contenente i documenti
#         persist_dir (Path): Directory per la persistenza dell'indice FAISS
#         embeddings (Optional[AzureOpenAIEmbeddings]): Modello di embeddings
#         llm (Optional[object]): Modello linguistico per la generazione
#         vector_store (Optional[FAISS]): Store vettoriale FAISS
#         retriever (Optional[object]): Retriever per la ricerca semantica
#         chain (Optional[object]): Catena LangChain per RAG
#     """
    
#     name: str = "RAG Tool"
#     description: str = (
#         "Uno strumento CrewAI che implementa Retrieval-Augmented Generation (RAG) "
#         "con persistenza FAISS su disco. Carica documenti locali (.txt, .md), "
#         "e risponde alle domande citando le fonti."
#     )

#     load_dotenv()

#     # Campi input Pydantic
#     documents_path: str = Field(default="docs/dogs_docs")
#     persist_dir: Path = Field(default=Path("faiss_index_docs"))

#     embeddings: Optional[AzureOpenAIEmbeddings] = None
#     llm: Optional[object] = None
#     vector_store: Optional[FAISS] = None
#     retriever: Optional[object] = None
#     chain: Optional[object] = None

#     def __init__(self, **data):
#         """
#         Inizializza il RagTool configurando embeddings, LLM e sistema RAG.
        
#         Durante l'inizializzazione:
#         1. Configura gli embeddings Azure OpenAI
#         2. Configura il modello linguistico
#         3. Testa le connessioni
#         4. Carica o costruisce l'indice FAISS
#         5. Crea la catena RAG
        
#         Args:
#             **data: Dati di configurazione per Pydantic
#         """
#         super().__init__(**data)

#         # 1️⃣ Embeddings
#         self.embeddings = AzureOpenAIEmbeddings(
#             api_key=os.getenv("AZURE_API_KEY"),
#             azure_endpoint=os.getenv("AZURE_API_BASE"),
#             api_version=os.getenv("AZURE_API_VERSION"),
#             model=os.getenv("EMBEDDING_MODEL")
#         )
#         # 2️⃣ LLM
#         self.llm = init_chat_model(
#             os.getenv("LLM_MODEL"),
#             model_provider="azure_openai",
#             api_key=os.getenv("AZURE_API_KEY"),
#             api_version=os.getenv("AZURE_API_VERSION"),
#             azure_endpoint=os.getenv("AZURE_API_BASE")
#         )
#         try:
#         # Test embedding: genera embedding per una stringa di esempio
#             test_emb = self.embeddings.embed_query("test")
#             print("Connessione embedding OK")
#         except Exception as e:
#             print("Errore embedding:", e)

#         try:
#             # Test LLM: genera una risposta di esempio
#             response = self.llm.invoke("Ciao!")
#             print("Connessione LLM OK")
#         except Exception as e:
#             print("Errore LLM:", e)

#         # 3️⃣ Carica o costruisce FAISS
#         docs = self._load_documents()
#         self.vector_store = self._load_or_build_vectorstore(docs)
#         self.retriever = self._make_retriever(self.vector_store)

#         # 4️⃣ Catena RAG
#         self.chain = self._build_chain()


#     def _load_documents(self) -> List[Document]:
#         """
#         Carica tutti i documenti dalla directory specificata.
        
#         Scansiona ricorsivamente la directory documents_path e carica tutti i file
#         con estensione .txt e .md, aggiungendo metadati sulla fonte.
        
#         Returns:
#             List[Document]: Lista dei documenti caricati
#         """
#         folder = Path(self.documents_path)
#         documents: List[Document] = []
#         for file_path in folder.glob("**/*"):
#             if file_path.suffix.lower() not in [".txt", ".md"]:
#                 continue
#             loader = TextLoader(str(file_path), encoding="utf-8")
#             docs = loader.load()
#             for doc in docs:
#                 doc.metadata["source"] = file_path.name
#             documents.extend(docs)
#         return documents

#     def _split_documents(self, docs: List[Document]) -> List[Document]:
#         """
#         Suddivide i documenti in chunk più piccoli per l'indicizzazione.
        
#         Args:
#             docs (List[Document]): Lista dei documenti da suddividere
            
#         Returns:
#             List[Document]: Lista dei chunk di documenti
#         """
#         splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
#         return splitter.split_documents(docs)

#     def _build_vectorstore(self, chunks: List[Document]) -> FAISS:
#         """
#         Costruisce un nuovo indice FAISS dai chunk di documenti.
        
#         Args:
#             chunks (List[Document]): Chunk di documenti da indicizzare
            
#         Returns:
#             FAISS: Store vettoriale FAISS costruito e salvato
#         """
#         vs = FAISS.from_documents(chunks, self.embeddings)
#         self.persist_dir.mkdir(parents=True, exist_ok=True)
#         vs.save_local(str(self.persist_dir))
#         return vs

#     def _load_or_build_vectorstore(self, docs: List[Document]) -> FAISS:
#         """
#         Carica un indice FAISS esistente o ne costruisce uno nuovo.
        
#         Se l'indice esiste già, lo carica. Altrimenti, suddivide i documenti
#         e costruisce un nuovo indice.
        
#         Args:
#             docs (List[Document]): Documenti da indicizzare se necessario
            
#         Returns:
#             FAISS: Store vettoriale FAISS
#         """
#         index_file = self.persist_dir / "index.faiss"
#         meta_file = self.persist_dir / "index.pkl"
#         if index_file.exists() and meta_file.exists():
#             return FAISS.load_local(
#                 str(self.persist_dir),
#                 self.embeddings,
#                 allow_dangerous_deserialization=True
#             )
#         chunks = self._split_documents(docs)
#         return self._build_vectorstore(chunks)

#     def _make_retriever(self, vector_store: FAISS):
#         """
#         Crea un retriever dal store vettoriale FAISS.
        
#         Args:
#             vector_store (FAISS): Store vettoriale da cui creare il retriever
            
#         Returns:
#             object: Retriever configurato per recuperare i 5 documenti più rilevanti
#         """
#         return vector_store.as_retriever(search_kwargs={"k": 5})


#     def _format_docs(self, docs: List[Document]) -> str:
#         """
#         Formatta i documenti recuperati per il prompt del LLM.
        
#         Args:
#             docs (List[Document]): Documenti da formattare
            
#         Returns:
#             str: Documenti formattati con informazioni sulla fonte
#         """
#         return "\n\n".join(
#             [f"[source:{d.metadata.get('source','doc')}] {d.page_content}" for d in docs]
#         )

#     def _build_chain(self):
#         """
#         Costruisce la catena LangChain per il sistema RAG.
        
#         La catena combina:
#         1. Recupero dei documenti rilevanti
#         2. Formattazione del contesto
#         3. Prompt engineering
#         4. Generazione della risposta con LLM
        
#         Returns:
#             object: Catena LangChain configurata per RAG
#         """
#         system_prompt = (
#             "Sei un assistente esperto. Rispondi in italiano. "
#             "Usa solo le informazioni nei documenti forniti. "
#             "Se non trovi la risposta, scrivi: 'Non è presente nel contesto fornito.'"
#         )

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "Domanda:\n{question}\n\nContesto:\n{context}")
#         ])

#         chain = (
#             {
#                 "question": RunnablePassthrough(),
#                 "context": lambda q: self._format_docs(
#                     self.retriever.get_relevant_documents(q) if self.retriever else []
#                 )
#             }
#             | prompt
#             | self.llm
#             | StrOutputParser()
#         )
#         return chain

#     def _run(self, query: str) -> str:
#         """
#         Esegue la query attraverso il sistema RAG.
        
#         Args:
#             query (str): Domanda dell'utente
            
#         Returns:
#             str: Risposta generata dal sistema RAG
#         """
#         return self.chain.invoke(query)


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# LangChain Core components for prompt/chain construction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# Qdrant vector database client and models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

load_dotenv()

@dataclass
class Settings:
    """
    Comprehensive configuration settings for the RAG pipeline.
    
    This class centralizes all configurable parameters, allowing easy tuning
    of the system's behavior without modifying the core logic.
    """
    
    qdrant_url: str = "http://localhost:6333"
    
    collection: str = "rag_chunks"

    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 700
    
    chunk_overlap: int = 120

    top_n_semantic: int = 30
    
    top_n_text: int = 100
    
    final_k: int = 6
    
    alpha: float = 0.75
    
    text_boost: float = 0.20
    
    use_mmr: bool = True
    
    mmr_lambda: float = 0.6
    
    lm_base_env: str = "AZURE_API_BASE"
    
    lm_key_env: str = "AZURE_API_KEY"
    
    lm_model_env: str = "LLM_MODEL"


SETTINGS = Settings()

def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """
    Restituisce un modello di embedding deployato su Azure.
    """
    embeddings = AzureOpenAIEmbeddings(
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL")
    )
    print('EMBEDDING CREATO')
    return embeddings

def get_llm(settings: Settings):
    """
    Inizializza un ChatModel.
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
    """
    llm = init_chat_model(
        model = os.getenv("LLM_MODEL"),
        model_provider='azure_openai',
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE")
    )
    print('LLM CREATO')
    return llm

def _load_documents(self) -> List[Document]:
    """
    Carica tutti i documenti dalla directory specificata.
    
    Scansiona ricorsivamente la directory documents_path e carica tutti i file
    con estensione .txt e .md, aggiungendo metadati sulla fonte.
    
    Returns:
        List[Document]: Lista dei documenti caricati
    """
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

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)