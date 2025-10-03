## put-rag

This repository contains a graphical RAG application. The original implementation, done for MLOps classes can be seen in the project4.ipynb and project4.py files.

The app is contained in the app.py file, while the worker process (launched as a separate thread from the app) is in the worker.py file.

### Usage
To run the GUI, install the requirements first:
```bash
pip install -r requirements.txt
```
And then simply execute:
```bash
python app.py
```
A Mistral AI API key is required to run the RAG process. It needs to be placed in the key.txt file. The steps of the process are explained in the user interface.