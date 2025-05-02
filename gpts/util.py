import os 
import json 
from PyPDF2 import PdfReader

def read_all_text(file_path: str) -> str:
    """
    Read all text from a file and return it as a string.

    Parameters:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_all_text(file_path: str, text: str) -> None:
    """
    Write a string to a file.

    Parameters:
        file_path (str): The path to the file where the text will be written.
        text (str): The text to be written to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def read_tab_separated_file(file_path: str) -> list:
    """
    Read a tab-separated file and return its content as a list of lists.

    Parameters:
        file_path (str): The path to the tab-separated file to be read.

    Returns:
        list: A list of lists containing the content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file.readlines()]

def write_tab_separated_file(file_path: str, data: list) -> None:
    """
    Write a list of lists to a tab-separated file.

    Parameters:
        file_path (str): The path to the file where the data will be written.
        data (list): The data to be written to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for row in data:
            file.write('\t'.join(row) + '\n')

def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return its content as a dictionary.

    Parameters:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json_file(file_path: str, data: dict) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters:
        file_path (str): The path to the file where the data will be written.
        data (dict): The data to be written to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_jsonl_file(file_path: str) -> list:
    """
    Read a JSON Lines (JSONL) file and return its content as a list of dictionaries.

    Parameters:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list: A list of dictionaries containing the content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]

def write_jsonl_file(file_path: str, data: list) -> None:
    """
    Write a list of dictionaries to a JSON Lines (JSONL) file.

    Parameters:
        file_path (str): The path to the file where the data will be written.
        data (list): The data to be written to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_pdf_text(file_path: str) -> str:
    """
    Read a PDF file and return its content as a string.

    Parameters:
        file_path (str): The path to the PDF file to be read.

    Returns:
        str: The content of the PDF file as a string.
    """
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return ' '.join(text)

def read_pdf_pages(file_path: str) -> list:
    """
    Read a PDF file and return its content as a list of strings, one for each page.

    Parameters:
        file_path (str): The path to the PDF file to be read.

    Returns:
        list: A list of strings, each containing the content of a page.
    """
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return text