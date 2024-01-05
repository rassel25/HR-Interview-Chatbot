import chromadb
from chromadb.api.types import Documents, Embeddings
from chromadb.api.models.Collection import Collection
from database.duckdb import get_duckdb_data, get_questions_answer
from typing import Tuple, List

CHROM_PERSISTENT_PATH = "/path/for/chroma/persistence"


def create_chroma_collection(documents: List[Tuple[str, str]], name: str) -> Collection:
    """
    Create chroma collection if not exists

    Parameters:
    -----------
    documents: Documents that needs to be added to collection
    name: Name of the collection

    Returns:
    -----------
    ChromaDB collection client
    """
    print("Working on creating collection ", name)
    chroma_client = chromadb.PersistentClient(path=CHROM_PERSISTENT_PATH)
    db = chroma_client.create_collection(name=name)
    for i, row in enumerate(documents):
        if row[0] == None:
            continue
        mapping_ids = ",".join(row[1])
        db.add(documents=row[0], ids=str(i), metadatas={"mapping_ids": mapping_ids})
    return db


def get_chroma_collection(name: str) -> Collection:
    """
    Get chroma collection client

    Parameters:
    -----------
    name: Name of the collection

    Returns:
    -----------
    ChromaDB collection client
    """
    chroma_client = chromadb.PersistentClient(path=CHROM_PERSISTENT_PATH)
    try:
        db = chroma_client.get_collection(name)
    except:
        print("Chroma collection not found")
        data = get_duckdb_data()
        db = create_chroma_collection(data, name)
    return db


def get_relevant_qa(
    db: Collection, question: str, sample_question_ids: List[str], top_k: int = 10
) -> List[Tuple[str, str, str]]:
    """
    Get relevant Questions and Answers

    Parameters:
    -----------
    db: Chroma collection client
    question: Question to which relevant questions needs to be searched for
    sample_question_ids: List of ids of sample questions
    name: Name of the collection
    top_k: No of similar questions that needs to be returned by chroma

    Returns:
    -----------
    List of relevant questions, answers and ratings
    """
    results = db.query(query_texts=[question], n_results=top_k)
    relevant_qs_ids = results["metadatas"][0]
    # print("Relevant qs_ids chroma ", relevant_qs_ids)
    res = []
    for qs_ids in relevant_qs_ids:
        mapping_ids = qs_ids["mapping_ids"].split(",")
        tmp = [i for i in mapping_ids if int(i) in sample_question_ids]
        res += tmp
    res = tuple(res)
    if len(res) == 0:
        raise Exception("No relevant questions found in the filtered dataset")
    # print("Length of relevaent ids ", len(res))
    data = get_questions_answer(res)
    return data
