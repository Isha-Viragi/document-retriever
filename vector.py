from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
from uuid import uuid4
import pandas as pd

from pages import final_description_doc
import chromadb


collection_name = "resume_collection"
db_location = "./chroma_resume_db"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


ids = [str(uuid4()) for _ in range(len(final_description_doc))]

persistent_client = chromadb.PersistentClient(path=db_location)

collection = persistent_client.get_or_create_collection(collection_name)


vector_store = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)
vector_store.add_documents(documents=final_description_doc, ids=ids)


# print("Collection count after add:", len(vector_store.get()))

# with open("output.txt", "a") as f:
#     print(collection.peek(), file=f)


retriever = vector_store.as_retriever(
    search_kwargs={
        'k': 8,
        "filter": {
            "section": "education"
        }}
)

# df = pd.read_csv("realistic_restaurant_reviews.csv")
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chroma_langchain_db"

# add_documents = not os.path.exists(db_location)

# if add_documents:
#   documents = []
#   ids = []

#   for i, row in df.iterrows():
#     document = Document(
#       page_content= row["Title"] + " " + row["Review"],
#       metadata = {"rating": row["Rating"], "date": row["Date"]},
#       id = str(i)
#     )
#     ids.append(str(i))
#     documents.append(document)

# vector_store = Chroma(
#   collection_name="restaurant_reviews",
#   persist_directory=db_location,
#   embedding_function=embeddings
# )

# if add_documents:
#   vector_store.add_documents(documents=documents, ids = ids)


# retriever = vector_store.as_retriever(
#   search_kwargs = {'k': 5}
# )
