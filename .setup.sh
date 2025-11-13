#!/bin/bash

# Setup script to load and store documents in vector store

python3 << 'EOF'
import os
import re
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from utils.splitter import Splitter
from store.vectorstore import vector_store

# Load environment variables
load_dotenv()

def find_json_files(folder_path):
    json_files = []
    json_pattern = re.compile(r'\.json$')
    try:
        for filename in os.listdir(folder_path):
            if json_pattern.search(filename):
                full_path = os.path.join(folder_path, filename)
                json_files.append(full_path)
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return json_files

def get_documents_from_json(file_path):
    docs_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for page in data:
                doc = Document(
                    page_content=page.get("text", ""),
                    metadata={
                        "source": page.get("url", ""),
                        "title": page.get("title", "")
                    }
                )
                docs_list.append(doc)
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
    return docs_list

def get_documents(json_file_paths):
    documents = []
    try:
        for file_path in json_file_paths:
            docs_from_file = get_documents_from_json(file_path)
            documents.extend(docs_from_file)
    except Exception as e:
        print(f"Error reading JSON files: {e}")
    return documents

# Main setup logic
folder_path = './data/iimb_data'
json_files = find_json_files(folder_path)
documents = get_documents(json_files)

if not documents:
    print(f"Warning: No documents found in {folder_path}")
    exit(1)

splitter = Splitter()
split_docs = splitter.split_documents(documents)
vector_store.store(split_docs)

print("Setup complete: Documents loaded and stored in vector store.")
EOF
