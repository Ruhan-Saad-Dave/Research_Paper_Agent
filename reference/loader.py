from langchain_community.document_loaders import PyPDFLoader
import asyncio 

file_path = "papers//sample.pdf"

loader = PyPDFLoader(
    file_path = file_path,
    #headers = None, # Used for get request to download the file 
    #password = None,
    mode = "page", # single for entire document, page for page-wise
    pages_delimiter = '',
    extract_images = True,
)

docs = [] 
docs_normal = loader.load() #loads entire document onto RAM 
docs_lazy = loader.lazy_load() #returns a generator that loads requested page 


for doc in docs_lazy:
    docs.append(doc)

print(docs[0].page_content[:100])
print(docs[0].metadata)

async def main():
    docs_async = await loader.aload() 
    docs_async_lazy = await loader.alazy_load() 

    for doc in docs_async_lazy:
        docs.append(doc)

asyncio.run(main())