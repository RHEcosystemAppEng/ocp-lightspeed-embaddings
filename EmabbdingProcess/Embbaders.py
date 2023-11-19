import json
import uuid
import os 
import chromadb
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

from CommonHandlers.DataLoader  import Loader
from CommonHandlers.ContextProvider import GeneralServiceContextLoader, GeneralStorageContextLoader
# from CommonHandlers.ContextProvider  import WatsonXContext

# import llama_index
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader,  get_response_synthesizer
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores import MilvusVectorStore
from llama_index import ServiceContext
from chromadb.config import Settings



from dotenv import load_dotenv
load_dotenv()


class ChromaEmbbader(): 
    
    def __init__(self) -> None:
        self.chroma_client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=os.getenv("CHROMADB_PORT"), headers={"Authorization": os.getenv('CHROMA_TOKEN')   }   )
        self.embedder =  OllamaEmbeddings(base_url=os.getenv("OLLAMA_SERVE") , model=os.getenv("CHROMADB_MODEL")).embed_documents
        self.loader = Loader()


    def save( self, collection_name):
        # get files 
        file_paths = Loader().load_docs()    
        # create collections
        collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.embedder,
            get_or_create = True
        )
        
        #save files
        try:
            total_files = len(file_paths)
            print(f" * Strting file loading to chromadb: loading {total_files} files ")
            for i, file in enumerate(file_paths): 
                with open(file, "r") as doc: 
                    doc = json.load(doc)
                    collection.add(documents=doc["document"], metadatas=doc["metadata"], ids=str(uuid.uuid4()))    
                    print(f" {i}/{total_files} Loaded to chromadb: {file} ")
            print(f"**  All files loaded to chromadb **")
        except Exception as e :
            print( str(e))
            
class LlmIndexEmbbader():

    def __init__(self,) -> None:
        self.loader = Loader()
        
    def save(self, collection_name, model_name="llama2" ,vector_db="local", service_context_name="local", file_paths=None): 
        
        '''
            saves trained data after embbadings in vectorDB  

            Args:
                collection_name: index collection name 
                model_name: the name of LLM model to be use
                vector_db: name of the supported vector db [local, chroma, milvus]
                service_context: llm server [local, Ollama,HuggingFace, openAI ]
    
            '''
        
        
        storage_context = GeneralStorageContextLoader().get_storage_context(vector_db=vector_db, model_name=model_name,collection_name=collection_name)
        service_context = GeneralServiceContextLoader().get_service_context(service_context=service_context_name, model_name=model_name)
        
        
        # load
        documents = SimpleDirectoryReader(input_files = file_paths).load_data()
        
        print(f"** Loaded docs  **")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context, show_progress=True)
        index.set_index_id(collection_name)
        index.storage_context.persist()
        print(f"**  Saved embeddings on {vector_db} storage **")
        return index
        