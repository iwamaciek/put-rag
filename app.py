import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import QThread
from worker import RAGWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Application")
        self.setGeometry(100, 100, 600, 300)

        self.explainer_label = QLabel("This application uses Retrieval-Augmented Generation (RAG) to answer questions based on the provided document.\n" \
        "The document can either be a file or a URL. Provide either a file path or a URL.", self)
        
        self.document_input = QLineEdit(self)
        self.document_input.setPlaceholderText("Enter file path or URL here...")
        self.load_document_button = QPushButton("Load Document", self)

        self.question_label = QLabel("Please type your questions in the box below:", self)
        self.question_label.hide()
        self.question_input = QLineEdit(self)
        self.question_input.hide()
        self.question_input.setPlaceholderText("Type your question here...")
        self.confirmation_button = QPushButton("Submit", self)
        self.confirmation_button.hide()

        self.answer_label = QLabel("Answer will be displayed here.", self)
        self.answer_label.hide()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.explainer_label)
        main_layout.addWidget(self.document_input)
        main_layout.addWidget(self.load_document_button)
        main_layout.addWidget(self.question_label)
        main_layout.addWidget(self.question_input)
        main_layout.addWidget(self.confirmation_button)
        main_layout.addWidget(self.answer_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # -- CONNECTIONS --
        self.load_document_button.clicked.connect(self.load_document)
        self.confirmation_button.clicked.connect(self.get_answer)

        self.thread = None
        self.worker = None

    def load_document(self):
        self.load_document_button.setText("Document Loading...")
        self.load_document_button.setEnabled(False)
        document_path = self.document_input.text()
        if document_path:
            self.thread = QThread()
            if document_path.startswith("http"):
                self.worker = RAGWorker(file_url=document_path)
            else:
                self.worker = RAGWorker(file_path=document_path)
            self.worker.moveToThread(self.thread)

            # Connect signals and slots
            self.thread.started.connect(self.worker.setup_rag)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.update_status)
            self.worker.setup_ready.connect(self.on_setup_complete)

            # Start the thread
            self.thread.start()

    def on_setup_complete(self, graph):
        self.graph = graph
        self.question_label.show()
        self.question_input.show()
        self.confirmation_button.show()
        self.load_document_button.setText("Load Document")
        self.load_document_button.setEnabled(True)

    def get_answer(self):
        self.confirmation_button.setText("Getting Answer...")
        self.confirmation_button.setEnabled(False)
        question = self.question_input.text()
        if question:
            self.thread = QThread()
            self.worker = RAGWorker(user_question=question, graph=self.graph)
            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.ask_question)
            self.worker.results_ready.connect(self.on_results_ready)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.thread.start()

    def on_results_ready(self, answer):
        self.answer_label.setText(answer)
        self.answer_label.show()
        self.confirmation_button.setText("Submit")
        self.confirmation_button.setEnabled(True)

    def update_status(self, message):
        self.load_document_button.setText(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec_()