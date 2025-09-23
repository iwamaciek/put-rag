import sys
import os
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List
import bs4
from time import sleep
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QLineEdit, QPushButton, QVBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Application")
        self.setGeometry(100, 100, 600, 300)

        self.explainer_label = QLabel("This application uses Retrieval-Augmented Generation (RAG) to answer questions based on the Shrek movie script.", self)
        self.question_label = QLabel("Please type your questions in the box below:", self)
        self.question_input = QLineEdit(self)
        self.confirmation_button = QPushButton("Submit", self)
        self.answer_label = QLabel("Answer will be displayed here.", self)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.explainer_label)
        main_layout.addWidget(self.question_label)
        main_layout.addWidget(self.question_input)
        main_layout.addWidget(self.confirmation_button)
        main_layout.addWidget(self.answer_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.question_input.setPlaceholderText("Type your question here...")
        self.confirmation_button.clicked.connect(self.on_submit)

        self.graph = self.setup_rag()

    def on_submit(self):
        question = self.question_input.text()
        response = self.graph.invoke({"question": question})
        self.answer_label.setText(response["answer"])
        self.question_input.clear()

    def setup_rag(self):
        # Get Mistral API key
        if not os.environ.get("MISTRAL_API_KEY"):
            os.environ["MISTRAL_API_KEY"] = open("C:\\Users\\iwama\\Desktop\\Osobiste\\rag_project\\key.txt").read().strip()

        # Create MistralAI llm object
        llm = ChatMistralAI(model="mistral-large-latest")

        # Create an embedder object
        embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m")

        # Load the Shrek script
        bs4_strainer = bs4.SoupStrainer(class_=("scrtext"))
        loader = WebBaseLoader(
            web_paths=("https://imsdb.com/scripts/Shrek.html",),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # chunk size (characters)
            chunk_overlap=256,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Loaded and split the Shrek script into {len(all_splits)} sub-documents.")

        # Put the embeddings into a vector store
        vector_store = InMemoryVectorStore(embeddings)

        document_ids = []
        for i in range(0, len(all_splits)):
            document_ids += vector_store.add_documents(documents=all_splits[i : i + 5])

        # Prepare the query
        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"], k=8)
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
        print("Set up complete!")
        return graph


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec_()