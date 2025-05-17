from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_experimental.text_splitter import SemanticChunker
import re
import json
from pathlib import Path
from pprint import pprint

file_path = "./data/resume.json"
json_data = json.loads(Path(file_path).read_text())

sections = ["projects", "experience", "education"]
final_description_doc = []


for section in sections:
    def metadata_func(record: dict, metadata: dict):
        metadata["section"] = section
        metadata["id"] = record.get("id")
        metadata["date"] = record.get("date")
        metadata["location"] = record.get("location")

        if record.get("title"):
            metadata["title"] = record.get("title")

        if record.get("role"):
            metadata["role"] = record.get("role")

        if record.get("company"):
            metadata["company"] = record.get("company")

        if record.get("diploma"):
            metadata["diploma"] = record.get("diploma")

        if record.get("honors"):
            metadata["honors"] = record.get("honors")

        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=f".{section}[]",
        content_key="description",
        metadata_func=metadata_func,
        text_content=False
    )

    section = loader.load()

    for project in section:
        description_list_json = json.loads(project.page_content)

        for description in description_list_json:
            description_doc = Document(
                page_content=description["message"],
                metadata={
                    **project.metadata,
                    "description_index": description["index"]

                }
            )
            final_description_doc.append(description_doc)

# with open("output.txt", "a") as f:
#     print(final_description_doc, file=f)


# sections_raw = []
# full_text = ""
# for page in pages:
#   full_text += "".join(page.page_content)

# sections_raw = re.split(r"\n(?=###.+)", full_text)


# for section in sections_raw:
#   section_header = re.match(r"###(\w+)", section).group(1)
#   print(section_header)


# user_input = "default"

# while user_input != "q":
#   user_input = input(f"Which page out of {len(sections_raw)} (q to quit): ")
#   print('\n\n')
#   if user_input != "q":
#     print(sections_raw[int(user_input)])
#     print('\n\n')
#     section_header = re.match(r"(?=###.+)", sections_raw[int(user_input)])
#     print(section_header)
#     print('\n\n')
#     print('------------------------------------------------')


# splitter = RecursiveCharacterTextSplitter(
#   separators=["?", "!", "."],
#   chunk_size = 300,
#   chunk_overlap = 50,
# )

# raw_chunks = splitter.split_documents(pages)
# initial_chunks = []

# for chunk in raw_chunks:
#   if len(chunk.page_content.strip()) != 0:
#     initial_chunks.append(chunk)

# user_input = "default"

# while user_input != "q":
#   user_input = input(f"Which page out of {len(initial_chunks)} (q to quit): ")
#   print('\n\n')
#   if user_input != "q":
#     print(initial_chunks[int(user_input)].page_content)
#     print('\n\n')
#     print('------------------------------------------------')


# print(len(initial_chunks))
