from crewai.tools import tool
from pathlib import Path
from typing import List
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.persist_dir = "faiss_index_medical"
        self.k = 4
        
        self.embeddings = self._get_embeddings()
        self.llm = self._get_llm()
        self.vector_store = None
        self.chain = None
        
        self._initialize_rag()
    

    def _get_embeddings(self):
        """Inizializza embedding Azure OpenAI"""
        api_key = os.getenv("AZURE_API_KEY")
        return AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=api_key
        )
    

    def _get_llm(self):
        """Inizializza LLM Azure OpenAI"""
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_API_BASE")
        deployment = os.getenv("MODEL")

        return AzureChatOpenAI(
            deployment_name=deployment,
            openai_api_version="2024-02-15-preview",
            azure_endpoint=endpoint,
            openai_api_key=api_key,
            temperature=0.1
        )
    
    # Load documents form documents folder
    def _load_documents(self):
        """Carica i documenti dalla cartella documents"""
        documents = []
        for file in os.listdir("documents"):
            with open(os.path.join("documents", file), "r") as f:
                documents.append(Document(page_content=f.read()))
        return documents
    

    def _initialize_rag(self):
        """Inizializza o carica il sistema RAG"""
        if Path(self.persist_dir).exists():
            # Carica indice esistente
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Indice FAISS medico caricato da disco")
        else:
            # Crea nuovo indice
            documents = self._load_documents()
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            # Salva su disco
            self.vector_store.save_local(self.persist_dir)
            print("✅ Nuovo indice FAISS medico creato e salvato")
        
        # Costruisci la chain RAG
        self._build_rag_chain()
    

    def _format_docs(self, docs: List[Document]) -> str:
        """Formatta i documenti recuperati"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    

    def _build_rag_chain(self):
        """Costruisce la chain RAG con LangChain"""
        # Template del prompt
        template = """Sei un cinofilo esperto. Usa le seguenti informazioni dal database cinofilo per rispondere alla domanda.
        Se le informazioni non sono sufficienti, indicalo chiaramente.
        
        Contesto dal database cinofilo:
        {context}
        
        Domanda: {question}
        
        Fornisci una risposta dettagliata, precisa e in italiano."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Crea il retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        
        # Costruisci la chain
        self.chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def search(self, question: str) -> str:
        """Esegue una ricerca RAG"""
        if not self.chain:
            return "❌ Sistema RAG non inizializzato correttamente"
        
        try:
            result = self.chain.invoke(question)
            return result
            
        except Exception as e:
            return f"❌ Errore nella ricerca RAG: {str(e)}"

# Istanza globale del sistema RAG
_rag_system = None

def get_rag_system():
    """Restituisce l'istanza singleton del sistema RAG"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system

@tool
def search_rag(question: str) -> str:
    """
    Effettua una ricerca nel database cinofilo locale utilizzando RAG.
    Restituisce informazioni cinofile basate sui documenti hardcodati di esempio.
    """
    try:
        rag_system = get_rag_system()
        result = rag_system.search(question)
        return f"Risultato della ricerca cinofila per '{question}':\n\n{result}"
    except Exception as e:
        return f"Errore nella ricerca RAG: {str(e)}"