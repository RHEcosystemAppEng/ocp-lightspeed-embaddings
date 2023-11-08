#!/usr/bin/env python
## Connects to a remote database server at localhost:8000

import sys
import json
import chromadb
from chromadb.utils import embedding_functions

from langchain.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
)
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

class RobustDirectoryLoader(DirectoryLoader):
    def load_file(self, index, path, docs, pbar):
        try:
            super().load_file(index, path, docs, pbar)
        except UnicodeDecodeError:
            print(f"Failed to decode file {path} due to UnicodeDecodeError.")
        except Exception as e:
            raise e

# Get .html files from ./scrape
loader = RobustDirectoryLoader(
    "./scrape",
    glob="*.html",
    loader_cls=BSHTMLLoader,
    show_progress=True,
    loader_kwargs={"get_text_separator": " "},
)
data = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(data)

# map sources from file directory to web source
## Read in mapping

with open("./scrape/sitemap.json", "r") as f:
    sitemap = json.load( f )

## Update source with URL instead of file location
doc_name = ""
doc_index = 0
doc_ids = []
for document in documents:
    if doc_name != document.metadata['title']:
        doc_name = document.metadata['title']
        doc_index = 0
    document.metadata["source"] = sitemap[document.metadata["source"].replace(".html", "").replace("scrape/", "")]
    doc_ids.append(document.metadata['title'] + str(doc_index))
    doc_index = doc_index + 1

chroma_collection = sys.argv[0]

print("Using collection: ", chroma_collection)

# Connect to ChromaDB server
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
# chroma_client = chromadb.HttpClient(host='chatbot-chroma-openshift-operators.apps.telco-edge.planolab.io', port=80)

# Print server version to verify connection
print("Server version: ", chroma_client.get_version())

#API Key = laptop
# embedder = embedding_functions.OpenAIEmbeddingFunction(
#      model_name="text-embedding-ada-002"
# )

embedder =  OllamaEmbeddings(model="llama2").embed_documents


# Connect or create collection in ChromaDB
collection = chroma_client.create_collection(
    name=chroma_collection,
    embedding_function=embedder,
    get_or_create = True
)

total_chunks = len(documents)
curr_chunk = 0
print("Total chunks: ", total_chunks)


# Add documents in chunks
chunk_size = 1
for i in range(0, len(documents), chunk_size):
    documents_to_process = documents[i:i + chunk_size]
    ids_to_process = doc_ids[i:i + chunk_size]
    print("Status: ", curr_chunk, "/", total_chunks)
    print("Working documents: ")
    process_docs = []
    process_meta = []
    for document in documents_to_process:
        process_docs.append(document.page_content)
        process_meta.append(document.metadata)
        print("  ", document.metadata['title'])
    
    collection.add(documents=process_docs, metadatas=process_meta, ids=ids_to_process)
    curr_chunk = curr_chunk + chunk_size

