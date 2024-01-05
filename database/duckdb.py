import duckdb
import pandas as pd
import os
from typing import List, Tuple

path = r"C:\Machine learning\End to End project\Omdena\HyderabadIndiaChapter_ChatbotInterviewPreparation\application\DuckDB_data\\"
filepaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
#path = "/path/for/duckdb/data/"
#filepaths = [path + f for f in os.listdir(path) if f.endswith(".csv")]


def get_duckdb_data() -> List[Tuple[str, List[str]]]:
    """
    Get all duckdb data that needs to be send to vector store

    Parameters:
    -----------
    None

    Returns:
    -----------
    List of questions and question_ids
    """
    print("Working on getting all duckdb data")
    df = pd.concat(map(pd.read_csv, filepaths))
    data = duckdb.sql(
        """SELECT interview_question,
    LIST(CAST (id AS VARCHAR)) AS question_ids
    FROM df group by company_name,job_title,job_category,skill_tested,interview_question"""
    ).fetchall()
    return data


def get_sample_questions(
    company_name: str, job_title: str, skill_tested: str, no_of_sample_questions: int
) -> Tuple[List[str], List[int]]:
    """
    Sample questions that needs to shared to question generator based on user inputs

    Parameters:
    -----------
    company_name: Name of the company
    job_title: Job title
    skill_tested: Skill that needs to be tested
    no_of_sample_questions: No. of sample questions that needs to returned

    Returns:
    -----------
    Tuple of relevant questions and question_ids
    """
    query_with_skill = f"SELECT id,interview_question FROM df WHERE trim(company_name)='{company_name}' AND trim(job_title)='{job_title}' AND trim(skill_tested)='{skill_tested}'"
    query_with_no_skill = f"SELECT id,interview_question FROM df WHERE trim(company_name)='{company_name}' AND trim(job_title)='{job_title}'"

    df = pd.concat(map(pd.read_csv, filepaths))
    res = duckdb.sql(query_with_skill).fetchall()
    if len(res) == 0:
        res = duckdb.sql(query_with_no_skill).fetchall()
    rel_ids = []
    rel_ques = set()
    for row in res:
        rel_ids.append(row[0])
        rel_ques.add(row[1])
    return list(rel_ques)[:no_of_sample_questions], rel_ids


def get_questions_answer(filter_ids: tuple) -> List[Tuple[str, str, int]]:
    """
    Get question and answers based on relevant ids

    Parameters:
    -----------
    filter_ids: Tuple of all the relevant ids that needs to be searched

    Returns:
    -----------
    List of question,answers and rating as tuple
    """
    df = pd.concat(map(pd.read_csv, filepaths))
    query = f"SELECT question,answer,rating FROM df WHERE id IN {filter_ids}"
    data = duckdb.sql(query).fetchall()
    return data
