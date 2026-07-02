from langchain_community.document_loaders import PyPDFLoader 

def ingest_pdf(
        file_path: str,
        password:str = None, 
        mode:str = "page", 
        pages_delimiter:str = '', 
        extract_images:bool = False
        ):
    
    loader = PyPDFLoader(
        file_path = file_path, 
        password = password,
        mode = mode, 
        pages_delimiter = pages_delimiter, 
        extract_images = extract_images,
    )

    return loader.lazy_load()