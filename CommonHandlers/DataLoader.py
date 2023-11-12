import os
from dotenv import load_dotenv

load_dotenv()

class Loader():
    def __init__(self) -> None: 
        pass 
    
    def load_docs(self,folder=os.getenv("TRINING_FOLDER")): 
        # get files 
        file_paths = []
        for folder_name, subfolders, files in os.walk(folder):
            for file in files:
                # Create the full path by joining folder path, folder name, and file name
                file_path = os.path.join(folder_name, file)
                file_paths.append(file_path)
        return file_paths    
