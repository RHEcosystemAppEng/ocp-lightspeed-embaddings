#!/usr/bin/env python
## Connects to a remote database server at localhost:8000

import os 

from bs4 import BeautifulSoup
from ScrapeProcess.HtmlParser import Parser
from EmabbdingProcess.Embbaders import ChromaEmbbader, LlmIndexEmbbader
from dotenv import load_dotenv

load_dotenv()

SCRAPE_FOLDER = "./scrape"
TRINING_FOLDER = "./TrainingData"

def parse():
    
# Get .html files from ./scrape
    files = os.listdir(SCRAPE_FOLDER)
    total_files = len(files) 

    print(f"**  Parsing HTML file: {total_files} ")
    parser = Parser(True)
    for i, file in enumerate(files): 
        file_path = os.path.join(SCRAPE_FOLDER,file)
        with open(file_path, "r") as html:    
        # Create a BeautifulSoup object
            soup = BeautifulSoup(html.read(), 'html.parser')
            parser.parse(soup)
            print(f"Completed HTML parsing: {i}/{total_files}. just completed {file} ")
    print(f"**  Completed parsing HTML **")
    
def save_embedding(collection_name):
    ChromaEmbbader().save(collection_name)

def save(collection_name):
    LlmIndexEmbbader().save(collection_name,vector_db="milvus",service_context="local")


if __name__ == "__main__":
    #parse()
    save(collection_name="OCP_4.13")
