'''
Load and ingest a document
'''

import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple

import langchain
import wandb
import openai
from getpass import getpass
from langchain.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)


def load_and_split_documents(file_path: str, chunk_size: int = 500, chunk_overlap=0) -> List[Document]:
    """Load documents from a directory of pdf files

    Args:
        data_dir (str): The directory containing the pdf files

    Returns:
        List[Document]: A list of documents
    """

    markdown_text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    documents = PyPDFLoader(file_path=file_path).load_and_split(
            text_splitter = markdown_text_splitter
        )

    return documents


def create_vector_store(
    documents,
    vector_store_path: str = "./vector_store",
) -> Chroma:
    """Create a ChromaDB vector store from a list of documents

    Args:
        documents (_type_): A list of documents to add to the vector store
        vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

    Returns:
        Chroma: A ChromaDB vector store containing the documents.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        if any(['VSCODE' in x for x in os.environ.keys()]):
            print('Please enter password in the VS Code prompt at the top of your VS Code window!')
        os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
        openai.api_key = os.getenv("OPENAI_API_KEY", "")

        assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
        print("OpenAI API key configured")

    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings(model = "text-embedding-ada-002",openai_api_key=api_key)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )
    vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb

    Args:
        documents (List[Document]): A list of documents to log to a wandb artifact
        run (wandb.run): The wandb run to log the artifact to.
    """
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json") as f:
        for document in documents:
            f.write(document.json() + "\n")

    run.log_artifact(document_artifact)


def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def log_prompt(prompt: dict, run: "wandb.run"):
    """Log a prompt to wandb

    Args:
        prompt (str): The prompt to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
    with prompt_artifact.new_file("prompt.json") as f:
        f.write(json.dumps(prompt))
    run.log_artifact(prompt_artifact)


def ingest_data(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_store_path: str,
) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of markdown files into a vector store

    Args:
        file_path (str):
        chunk_size (int):
        chunk_overlap (int):
        vector_store_path (str):


    """
    # load the documents
    documents = load_and_split_documents(file_path)
    # create document embeddings and store them in a vector store
    vector_store = create_vector_store(documents, vector_store_path)
    return documents, vector_store


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="The path containing the sample documentation",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--prompt_file",
        type=pathlib.Path,
        default="./chat_prompt.json",
        help="The path to the chat prompt to use",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmchat",
        type=str,
        help="The wandb project to use for storing artifacts",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)
    documents, vector_store = ingest_data(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )
    log_dataset(documents, run)
    log_index(args.vector_store, run)
    log_prompt(json.load(args.prompt_file.open("r")), run)
    run.finish()


if __name__ == "__main__":
    main()