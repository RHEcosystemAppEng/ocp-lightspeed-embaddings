import json
import uuid
import os 
import chromadb
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

from CommonHandlers.DataLoader  import Loader
from CommonHandlers.ServiceContext  import WatsonXContext


from dotenv import load_dotenv
import llama_index
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings import TextEmbeddingsInference
from llama_index import ServiceContext
from llama_index.llms import Ollama   

from pymilvus import Collection, connections

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
    '''
    

    '''
    
    
    def __init__(self,) -> None:
        self.loader = Loader()
        
    def save(self,collection_name, model_name="llama2" ,vector_db="local", service_context="local"): 
        
        '''
            saves trained data after embbadings in vectorDB  

            Args:
                collection_name: index collection name 
                model_name: the name of LLM model to be use
                vector_db: name of the supported vector db [local, chroma, milvus]
                service_context: llm server [local, Ollama,HuggingFace, openAI ]
    
            '''
        
        file_paths = Loader().load_docs()  
        
        # TODO storage_context configuration 
        
        if vector_db == "local":
            storage_context = StorageContext.from_defaults()
        elif vector_db == "chroma":
            chroma_client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=os.getenv("CHROMADB_PORT"), headers={"Authorization": os.getenv('CHROMA_TOKEN')   }   )
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function= OllamaEmbeddings(base_url=os.getenv("OLLAMA_SERVE") , model=model_name) , 
                get_or_create = True            
                )
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        elif vector_db == "milvus": 
            collection = Collection(collection_name)     
            
            index_params= {
                'nlist': 2048
            }

            collection.create_index(
                        field_name="ocp", 
                        index_params=index_params
                        )
            
            ()
            vector_store = MilvusVectorStore( "http://0.0.0.0:19530" ,  collection_name==collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # service contxts   
        embed_model = "local"
        service_context = ServiceContext.from_defaults(embed_model=embed_model) 
        if service_context == "local": 
            embed_model = "local"
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
        elif service_context == "HuggingFace":
            embed_model = HuggingFaceBgeEmbeddings(model_name=model_name)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
        elif service_context == "Ollama":
            embed_model = OllamaEmbeddings(base_url=os.getenv("OLLAMA_SERVE") , model=model_name)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
        elif service_context == "WatsonX": 
            service_context = WatsonXContext.get_watsonx_context(model=model_name)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            
        
        # load
        documents = SimpleDirectoryReader(input_files = file_paths).load_data()
        
        print(f"** Loaded docs  **")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context, show_progress=True)
        index.set_index_id(collection_name)
        index.storage_context.persist()
        print(f"**  Saved embeddings on {vector_db} storage **")
