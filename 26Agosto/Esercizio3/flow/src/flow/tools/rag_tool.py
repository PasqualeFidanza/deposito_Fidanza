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


from crewai.tools import BaseTool
from pydantic import Field
from typing import Optional, List, Any, Iterable, Tuple
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    OptimizersConfigDiff, ScalarQuantization, ScalarQuantizationConfig,
    PayloadSchemaType, FieldCondition, MatchText, Filter, SearchParams
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

class RagTool(BaseTool):
    name: str = "RAG Tool"
    description: str = (
        "Uno strumento CrewAI che implementa Retrieval-Augmented Generation (RAG) "
        "utilizzando Qdrant come vector store."
    )

    documents_path: str = Field(default="docs/dogs_docs")
    collection: str = Field(default="rag_chunks")

    embeddings: Optional[AzureOpenAIEmbeddings] = None
    llm: Optional[Any] = None
    client: Optional[QdrantClient] = None
    chain: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("🔧 Inizializzazione RagTool con Qdrant...")
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.client = self.get_qdrant_client()

        docs = self._load_documents()
        chunks = self.split_documents(docs)

        self.recreate_collection_for_rag(vector_size=len(self.embeddings.embed_query("test")))
        self.upsert_chunks(chunks)

        self.chain = self.build_rag_chain()
        print("✅ RagTool inizializzato con successo")

    def _run(self, query: str) -> str:
        results = self.hybrid_search(query)
        context = self.format_docs_for_prompt(results)
        return self.chain.invoke({"question": query, "context": context})

    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            model=os.getenv("EMBEDDING_MODEL"),
        )

    def get_llm(self):
        from langchain.chat_models import init_chat_model
        return init_chat_model(
            model=os.getenv("LLM_MODEL"),
            model_provider='azure_openai',
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )

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

    def split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SETTINGS.chunk_size,
            chunk_overlap=SETTINGS.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""]
        )
        return splitter.split_documents(docs)

    def get_qdrant_client(self) -> QdrantClient:
        return QdrantClient(url=SETTINGS.qdrant_url)

    def recreate_collection_for_rag(self, vector_size: int):
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
            optimizers_config=OptimizersConfigDiff(default_segment_number=2),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type="int8", always_ram=False)
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT
        )
        for key in ["doc_id", "source", "title", "lang"]:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD
            )

    def upsert_chunks(self, chunks: List[Document]):
        vecs = self.embeddings.embed_documents([c.page_content for c in chunks])
        points = self.build_points(chunks, vecs)
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def build_points(self, chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
        pts: List[PointStruct] = []
        for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
            payload = {
                "doc_id": doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "lang": doc.metadata.get("lang", "en"),
                "text": doc.page_content,
                "chunk_id": i - 1
            }
            pts.append(PointStruct(id=i, vector=vec, payload=payload))
        return pts
    
    def qdrant_semantic_search(self, query: str, limit: int, with_vectors: bool = False):
        """
        Esegue una ricerca semantica su Qdrant.

        Args:
            query: Query utente
            limit: Numero massimo di risultati
            with_vectors: Se includere i vettori nei risultati

        Returns:
            List[ScoredPoint]: Risultati della ricerca
        """
        qv = self.embeddings.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection,
            query=qv,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
            search_params=SearchParams(
                hnsw_ef=256,
                exact=False
            ),
        )
        return res.points
    
    def qdrant_text_prefilter_ids(self, query: str, max_hits: int) -> List[int]:
        """
        Usa l'indice full-text su 'text' per prefiltrare i punti che contengono parole chiave.
        Restituisce un sottoinsieme di ID da usare come boost.
        """
        matched_ids: List[int] = []
        next_page = None

        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text", match=MatchText(text=query))]
                ),
                limit=min(256, max_hits - len(matched_ids)),
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            matched_ids.extend([p.id for p in points])
            if not next_page or len(matched_ids) >= max_hits:
                break

        return matched_ids
    
    def mmr_select(self, query_vec: List[float], candidates_vecs: List[List[float]], k: int, lambda_mult: float) -> List[int]:
        """
        Seleziona risultati diversificati usando l'algoritmo Maximal Marginal Relevance (MMR).
        """
        import numpy as np
        V = np.array(candidates_vecs, dtype=float)
        q = np.array(query_vec, dtype=float)

        def cos(a, b):
            na = (a @ a) ** 0.5 + 1e-12
            nb = (b @ b) ** 0.5 + 1e-12
            return float((a @ b) / (na * nb))

        sims = [cos(v, q) for v in V]
        selected: List[int] = []
        remaining = set(range(len(V)))

        while len(selected) < min(k, len(V)):
            if not selected:
                best = max(remaining, key=lambda i: sims[i])
                selected.append(best)
                remaining.remove(best)
                continue

            best_idx = None
            best_score = -1e9
            for i in remaining:
                max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
                score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected




    def hybrid_search(self, query: str):
        """
        Effettua una ricerca ibrida combinando similarità semantica e matching testuale.

        Returns:
            List[ScoredPoint]: Documenti rilevanti con punteggio ibrido e diversificazione MMR opzionale.
        """
        # (1) Ricerca semantica
        sem = self.qdrant_semantic_search(
            query=query,
            limit=SETTINGS.top_n_semantic,
            with_vectors=True
        )
        if not sem:
            return []

        # (2) Prefiltro testuale (full-text match)
        text_ids = set(self.qdrant_text_prefilter_ids(
            query=query,
            max_hits=SETTINGS.top_n_text
        ))

        # (3) Normalizzazione score
        scores = [p.score for p in sem]
        smin, smax = min(scores), max(scores)

        def norm(x):  # robusto anche se tutti i punteggi sono uguali
            return 1.0 if smax == smin else (x - smin) / (smax - smin)

        # (4) Fusione semantico + boost testuale
        fused: List[Tuple[int, float, Any]] = []
        for idx, p in enumerate(sem):
            base = norm(p.score)
            fuse = SETTINGS.alpha * base
            if p.id in text_ids:
                fuse += SETTINGS.text_boost
            fused.append((idx, fuse, p))

        fused.sort(key=lambda t: t[1], reverse=True)

        # (5) MMR opzionale
        if SETTINGS.use_mmr:
            qv = self.embeddings.embed_query(query)
            N = min(len(fused), max(SETTINGS.final_k * 5, SETTINGS.final_k))
            cut = fused[:N]
            vecs = [sem[i].vector for i, _, _ in cut]
            mmr_idx = self.mmr_select(qv, vecs, SETTINGS.final_k, SETTINGS.mmr_lambda)
            picked = [cut[i][2] for i in mmr_idx]
            return picked

        # (6) Top-K finale
        return [p for _, _, p in fused[:SETTINGS.final_k]]


    def format_docs_for_prompt(self, points: Iterable[Any]) -> str:
        blocks = []
        for p in points:
            pay = p.payload or {}
            src = pay.get("source", "unknown")
            blocks.append(f"[source:{src}] {pay.get('text','')}")
        return "\n\n".join(blocks)

    def build_rag_chain(self):
        system_prompt = (
            "Sei un assistente tecnico. Rispondi in italiano, conciso e accurato. "
            "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO. "
            "Se non è presente, dichiara: 'Non è presente nel contesto fornito.' "
            "Cita sempre le fonti nel formato [source:FILE]."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
             "Domanda:\n{question}\n\n"
             "CONTENUTO:\n{context}\n\n"
             "Istruzioni:\n"
             "1) Risposta basata solo sul contenuto.\n"
             "2) Includi citazioni [source:...].\n"
             "3) Niente invenzioni.")
        ])
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
