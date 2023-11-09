import json
import uuid
import os 
import chromadb
from langchain.embeddings.ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

class ChromaEmbbader(): 
    
    def __init__(self) -> None:
        self.chroma_client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=os.getenv("CHROMADB_PORT"), headers={"Authorization": os.getenv('CHROMA_TOKEN')   }   )
        self.embedder =  OllamaEmbeddings(base_url="http://ollama-serve-ollama.apps.cn-ai-lab.6aw6.p1.openshiftapps.com" , model=os.getenv("CHROMADB_MODEL")).embed_documents

    def load_docs(self,folder=os.getenv("TRINING_FOLDER")): 
        # get files 
        file_paths = []
        for folder_name, subfolders, files in os.walk(folder):
            for file in files:
                # Create the full path by joining folder path, folder name, and file name
                file_path = os.path.join(folder_name, file)
                file_paths.append(file_path)
        return file_paths    

    def save( self, collection_name):
        # get files 
        file_paths = self.load_docs()    
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
            
