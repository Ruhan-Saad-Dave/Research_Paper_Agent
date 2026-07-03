from langchain_text_splitters import RecursiveCharacterTextSplitter #text based (sentence/paragraph) splitting

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 0,
)
texts = text_splitter.split_text("Put document in here")


from langchain_text_splitters import CharacterTextSplitter #token based splitting

text_splitter = CharacterTextSplitter(
    encoding_name="cl100k_base",
    chunk_size = 100,
    chunk_overlap = 0,
)

#other splitters: MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, RecursiveJSONSplitter, Language Splitters (python, c++ etc)

#example of chaining together (hierarchical)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter 

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on = headers_to_split_on,
)
section_documents = header_splitter.split_text("raw markdown text")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)
final_chunks = text_splitter.split_documents(section_documents)