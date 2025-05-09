import logging
import time
from typing_extensions import List, TypedDict, Dict
import json
import enum

class MultipleChoice(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class AnswerItem(TypedDict):
    question: str
    answer: MultipleChoice

class QuestionAnswers(TypedDict):
    answers: List[AnswerItem]


def check_file_active(client, file):
    logging.info(f"Checking if file {file.name} is active")
    attempts = 0
    while not file.state or file.state.name != "ACTIVE":
        attempts += 1
        logging.info(f"Processing video... (attempt {attempts})")
        logging.info(f"File state: {file.state}")
        print("Processing video...")
        print("File state:", file.state)
        time.sleep(5)  # Wait 5 seconds before checking again
        file = client.files.get(name=file.name)
    logging.info(f"File {file.name} is now active")
    return file


def generate_content_with_retry(client, model, contents, max_retries=3):
    retry_count = 0
    logging.info(f"Generating content with model {model}")
    while retry_count < max_retries:
        try:
            logging.info(f"Attempt {retry_count + 1}/{max_retries} to generate content")
            result = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': QuestionAnswers,
                },
            )
            logging.info(f"Successfully generated content")
            return result
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            if "503 UNAVAILABLE" in str(e):
                retry_count += 1
                if retry_count < max_retries:
                    logging.warning(f"Model overloaded, retrying... (Attempt {retry_count}/{max_retries})")
                    print(f"Model overloaded, retrying... (Attempt {retry_count}/{max_retries})")
                    time.sleep(10)  # Wait 10 seconds before retrying
                else:
                    logging.error(f"Max retries reached. Moving to next video.")
                    print("Max retries reached. Moving to next video.")
                    return None
            else:
                logging.error(f"Unexpected error: {str(e)}")
                raise e  # Re-raise other exceptions
            
def load_api_key():
    with open("API_KEY.txt", "r") as f:
        api_key = f.read().strip()
    return api_key

def load_questions(questions_file):
    logging.info(f"Loading questions from {questions_file}")
    try:
        with open(questions_file, 'r') as file:
            questions_dict = json.load(file)
            logging.info(f"Successfully loaded {len(questions_dict)} questions")
    except FileNotFoundError:
        logging.error(f"File not found: {questions_file}")
        print("File not found!")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {questions_file}")
        print("Invalid JSON format!")
        return {}
    return questions_dict
