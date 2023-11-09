
import json
import os

class Parser( ):  
    """ parses the RH scrape HTML doc and returns json or db vector   

    Attributes:
        persist     save scrape data in folder 
        persist_folder  persist folder  
    """
    
    def __init__(self,persist=True, persist_folder="TrainingData" ) -> None:    
        
        self.persist = persist
        self.persist_folder = persist_folder 

        
    def save(self,metadata,file_name,data): 
        full_path = os.path.join(self.persist_folder, metadata["product_name"], metadata["product_version"], metadata["topic"] )
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        file_path = os.path.join(full_path, file_name)

        with open(file_path, 'w') as file:
            file.write(data)
    
    def parse( self, soup):
        product_version = soup.find('span', class_='productnumber')
        product_name = soup.find('span', class_='productname')
        subtitle = soup.find('h2', class_='subtitle')

        metadata = { } 
        metadata["product_name"] = product_name.text if product_name  else None
        metadata["product_version"] = product_version.text if product_version else None
        metadata["topic"] = subtitle.text if subtitle else None
        chapter = soup.find_all(class_='chapter')

        for sub in chapter: 
            for i, p in enumerate(sub.find_all('p')): 
                    metadata["title1"] = sub.h1.text if sub.h1 else None
                    metadata["title2"] = sub.h2.text if sub.h2 else None
                    metadata["title3"] = sub.h3.text if sub.h3 else None            
                    
                    
                    response = { 
                        "metadata":metadata,
                        "document":p.text.replace('\t','').replace('\xa0','').replace('\n','') if p else None
                    }
                    
                    # write to folder 
                    if self.persist == True : 
                        self.save(metadata,f"{metadata['topic']}_{i}.json",json.dumps(response))
