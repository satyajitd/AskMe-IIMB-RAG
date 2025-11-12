import os
import re
import json
from langchain_core.documents import Document

from splitter import Splitter
from vectorstore import VectorStore

def find_json_files(folder_path) -> list[str]:
    """
    Finds all files with a .json extension in a specified folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the found JSON files.
    """

    # Example usage:
    # folder_path = '/path/to/your/folder'
    # json_files = find_json_files(folder_path)
    # print(json_files) # Outputs a list of JSON file paths
    
    json_files = []
    # Define the regex pattern to match files ending with .json
    # The '\.' escapes the dot, and '$' anchors the pattern to the end of the string.
    json_pattern = re.compile(r'\.json$')

    try:
        # Iterate through all files and directories in the given folder
        for filename in os.listdir(folder_path):
            # Check if the filename matches the JSON pattern
            if json_pattern.search(filename):
                # Construct the full path to the JSON file
                full_path = os.path.join(folder_path, filename)
                json_files.append(full_path)
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return json_files

def get_documents(json_file_paths: list[str]) -> list[Document]:
    """
    Reads JSON files and converts their content into LangChain Document objects.
    Args:
        json_file_paths (list): A list of paths to JSON files.
    Returns:
        list: A list of LangChain Document objects.
    """

    documents = []
    try:
        
        for file_path in json_file_paths:
            docs_from_file = get_documents_from_json(file_path)
            documents.extend(docs_from_file)
    except Exception as e:
        print(f"Error reading JSON files: {e}")

    return documents

def get_documents_from_json(file_path: str) -> list[Document]:
    """
    Helper function to load documents from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        list: A list of LangChain Document objects.
    """
    
    docs_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            

            # Convert the raw JSON data back into LangChain Document objects
            # The Document object needs 'page_content' and 'metadata'
            for page in data:
                doc = Document(
                    # The 'text' field from your JSON goes into 'page_content'
                    page_content=page.get("text", ""), 
                    # 'url' and 'title' are great for metadata
                    metadata={
                        "source": page.get("url", ""),
                        "title": page.get("title", "")
                    }
                )
                docs_list.append(doc)
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
    
    return docs_list
    

if __name__ == "__main__":
    folder_path = './data'
    json_files = find_json_files(folder_path)
    documents = get_documents(json_files)

    splitter = Splitter()
    vector_store = VectorStore(collection_name="my_collection")  

    split_docs = splitter.split_documents(documents)
    vector_store.store(split_docs)
