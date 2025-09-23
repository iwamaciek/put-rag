import sys
import os
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List
import bs4

from PyQt5.QtCore import QObject, pyqtSignal

class RAGWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    setup_ready = pyqtSignal(object)
    results_ready = pyqtSignal(str)

    def __init__(self, file_path: str = None, file_url: str = None, user_question: str = None, graph=None):
        super().__init__()
        self.file_path = file_path
        self.file_url = file_url
        self.user_question = user_question
        self.graph = graph

    def setup_rag(self):
        # Get Mistral API key
        if not os.environ.get("MISTRAL_API_KEY"):
            with open("C:\\Users\\iwama\\Desktop\\Osobiste\\rag_project\\key.txt") as key_file:
                os.environ["MISTRAL_API_KEY"] = key_file.read().strip()

        # Create MistralAI llm object
        llm = ChatMistralAI(model="mistral-large-latest")
        self.progress.emit("Mistral LLM loaded.")

        # Create an embedder object
        embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m")
        self.progress.emit("Embeddings model loaded.")

        # Load the document
        if self.file_path:
            loader = DoclingLoader(file_path=self.file_path)
            docs = loader.load()
            self.progress.emit("Document loaded from file.")
        elif self.file_url:
            bs4_strainer = bs4.SoupStrainer(class_=("scrtext"))
            loader = WebBaseLoader(
                web_paths=(self.file_url,),
                bs_kwargs={"parse_only": bs4_strainer},
            )
            docs = loader.load()
        else:
            self.progress.emit("No document source provided.")
            self.setup_ready.emit(None)
            self.finished.emit()
            return
        self.progress.emit("Document loaded.")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # chunk size (characters)
            chunk_overlap=256,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        self.progress.emit(f"Document split into {len(all_splits)} chunks.")

        # Put the embeddings into a vector store
        vector_store = InMemoryVectorStore(embeddings)

        document_ids = []
        for i in range(0, len(all_splits)):
            document_ids += vector_store.add_documents(documents=all_splits[i : i + 5])
        self.progress.emit("Vector store created.")

        # Prepare the query
        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"], k=16)
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        self.progress.emit("Set up complete!")
        self.setup_ready.emit(graph)
        self.finished.emit()

    def ask_question(self):
        result = self.graph.invoke({"question": self.user_question})
        self.results_ready.emit(result["answer"])
        self.finished.emit()
