#%%
import os
import time
import json
import logging
from google import genai
from typing_extensions import List, TypedDict, Dict
import enum


os.environ['GOOGLE_RESUMABLE_MEDIA_CHUNK_SIZE'] = str(10 * 1024 * 1024)

# Set up logging
def setup_logger(cubicle):
    # Create log filename based on cubicle
    log_filename = os.path.splitext(cubicle)[0] + "_results.log"
    
    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite existing log file
    )
    
    # Add console handler to see logs in terminal too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting video understanding benchmark for {cubicle}")
    return log_filename

# CONSTANTS ------------------------------------------------------------
OBJECT_COUNTING = "Object Counting.json"
OBJECT_STATE = "Object State Change.json"   
OBJECT_LOCATION = "Object Location Change.json"
OBJECT_DETECTION = "Object Detection.json"


# FUNCTIONS ------------------------------------------------------------

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


def combine_questions(cubicle):
    logging.info(f"Combining questions from all files in {cubicle}")
    object_counting_dict = load_questions(os.path.join(cubicle, OBJECT_COUNTING))
    object_state_dict = load_questions(os.path.join(cubicle, OBJECT_STATE))
    object_location_dict = load_questions(os.path.join(cubicle, OBJECT_LOCATION))
    object_detection_dict = load_questions(os.path.join(cubicle, OBJECT_DETECTION))

    object_list = [object_counting_dict, object_state_dict, object_location_dict, object_detection_dict]

    object_combined_dict = {}
    i = 0
    for object_dict in object_list:
        for Q in object_dict:
            object_combined_dict[f"Q{i}"] = object_dict[Q]
            i += 1
    
    logging.info(f"Combined {len(object_combined_dict)} questions total")
    return object_combined_dict


# Load the questions from the json file

def generate_questions(Q_dict, video_number):
    logging.info(f"Generating questions prompt for video number {video_number}")
    prompt_start = """
You are taking a multiple-choice benchmark.  
  
To answer the questions, you will be provided with two videos. The first video is the initial state of the scene, and the second video is the final state of the scene.
Please pay attention to the changes of object in the scene between the two videos.
Futhermore, if we say on object had been removed, vanished or disappear from a cubicle it means it will not be present in the second video.
If we say an object has appeared in a cubicle it means it will be present in the second video.
If we say an object has been moved, it means it has moved within the cubicle or between cubicles.

Please answer the question to best of your ability. 

For each question below, reply with the single letter (A–E) that you believe is correct.  
Do not provide explanations—only the letter.

Please return a json object in the following format:

"answers": [
{
    "question": "Q1",
    "answer": "B"
},
{
    "question": "Q2",
    "answer": "C"
},
{
    "question": "Q3",
    "answer": "C"
},
{
    "question": "Q4",
    "answer": "B"
},
{
    "question": "Q5",
    "answer": "C"
},
{
    "question": "Q6",
    "answer": "C"
},
{
    "question": "Q7",
    "answer": "C"
}
....
]

Please use the Q# correspoding the questions provided.

"""

    # Find all the questions that have the video number as the initial video
    questions_list = [key for key, value in Q_dict.items() if value["Initial Video"] == video_number]
    logging.info(f"Found {len(questions_list)} questions for video number {video_number}")

    questions_prompt_list = []
    for i, question in enumerate(questions_list):
        questions_prompt_list.append(question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n")
        for choice in list(Q_dict[question]["Multiple Choice"].keys()):
            questions_prompt_list[i] += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """
"""

    questions_prompt = prompt_start + "\n".join(questions_prompt_list) + "\n" + prompt_end
    logging.info(f"Generated prompt with {len(questions_list)} questions")
    return questions_prompt

# Check file is active
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


# Define a schema that allows for dynamic number of questions for Gemini
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

def save_results(cubicle, prompt, video_paths, questions_dict, dict_results, total_correct, total_questions, model_name):
    """
    Save benchmark results to a JSON file.
    
    Args:
        questions_file: Original questions file name
        prompt: The prompt used for the model
        video_paths: List of video file paths
        questions_dict: Dictionary of questions
        dict_results: Dictionary of model answers and correct answers
        total_correct: Number of correct answers
        total_questions: Total number of questions
        model_name: Name of the Gemini model used
    """
    # Create output filename based on input filename
    output_filename = os.path.splitext(cubicle)[0] + "_results.json"
    
    logging.info(f"Saving results to {output_filename}")
    
    # Prepare the results dictionary
    results = {
        "model": model_name,
        "prompt": prompt,
        "video_paths": video_paths,
        "questions": questions_dict,
        "answers": dict_results,
        "score": {
            "correct": total_correct,
            "total": total_questions,
            "percentage": (total_correct / total_questions) * 100 if total_questions > 0 else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    try:
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results successfully saved to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
    
    print(f"Results saved to {output_filename}")



def load_api_key():
    with open("API_KEY.txt", "r") as f:
        api_key = f.read().strip()
    return api_key

# INPUTS ------------------------------------------------------------
cubicle = "Local Changes/Josh 2008S"

gemini_model = "gemini-2.5-pro-preview-03-25"


# CODE ------------------------------------------------------------

# Initialize logging
log_filename = setup_logger(cubicle)

api_key = load_api_key()

client = genai.Client(api_key=api_key)

logging.info(f"Log file created at {log_filename}")

logging.info("Initializing Gemini client")

logging.info(f"Using model: {gemini_model}")

logging.info("Starting benchmark execution")
questions_dict = combine_questions(cubicle)

initial_video = 0
final_video = questions_dict[list(questions_dict.keys())[-1]]["Final Video"]
logging.info(f"Video range: {initial_video} to {final_video}")

dict_answers = {}
for i in range(initial_video, final_video):
    logging.info(f"Processing video pair {i} -> {i+1}")
    
    prompt = generate_questions(questions_dict, i)

    video_file_1 = os.path.join(cubicle, 'videos', f'ep_{i:02d}_a.mp4')
    video_file_2 = os.path.join(cubicle, 'videos', f'ep_{i+1:02d}_a.mp4')
    
    logging.info(f"Uploading video 1: {video_file_1}")
    try:
        video_1 = client.files.upload(file=video_file_1)
        logging.info(f"Successfully uploaded video 1: {video_file_1}")
    except Exception as e:
        logging.error(f"Error uploading video 1: {str(e)}")
        continue
    
    logging.info(f"Uploading video 2: {video_file_2}")
    try:
        video_2 = client.files.upload(file=video_file_2)
        logging.info(f"Successfully uploaded video 2: {video_file_2}")
    except Exception as e:
        logging.error(f"Error uploading video 2: {str(e)}")
        continue

    # Poll until the video files are completely processed (state becomes ACTIVE)
    video_1 = check_file_active(client, video_1)
    video_2 = check_file_active(client, video_2)
    # Once ACTIVE, use the file

    result = generate_content_with_retry(client, gemini_model, [video_1, video_2, prompt])
    # result = generate_content_with_retry(client, "gemini-2.5-flash-preview-04-17", [video_1,  video_2, prompt])
    
    if result:
        logging.info(f"Got result: {result.text}")
        print(result.text)
        
        try:
            answers_json = json.loads(result.text)
            logging.info(f"Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {str(e)}")
            logging.error(f"Raw response: {result.text}")
            continue

        for answer in answers_json['answers']:
            dict_answers[answer['question']] = answer['answer']
            logging.info(f"Recorded answer: {answer['question']} -> {answer['answer']}")
    else:
        logging.warning(f"No result received for video pair {i} -> {i+1}")
    
    logging.info("Cleaning up files")
    for f in client.files.list():
        try:
            client.files.delete(name=f.name)
            logging.info(f"Deleted: {f.name}")
        except Exception as e:
            logging.error(f"Error deleting file {f.name}: {str(e)}")
        
    print("Deleted:", f.name)

logging.info("Answers: --------------------------------")
logging.info(dict_answers)
print("Answers: --------------------------------")
print(dict_answers)    

total_correct = 0
total_questions = 0

dict_results = {}

logging.info("Calculating results")
for question in questions_dict:
    if question in dict_answers:
        total_questions += 1
        model_choice = dict_answers[question]
        correct_choice = questions_dict[question]["Correct Choice"]
        
        if correct_choice == model_choice:
            total_correct += 1
            logging.info(f"{question}: CORRECT - {model_choice}")
        else:
            logging.info(f"{question}: WRONG - Expected {correct_choice}, got {model_choice}")

        dict_results[question] = {
            "Correct Choice": correct_choice,
            "Model Choice": model_choice
        }
    else:
        logging.warning(f"{question}: NO ANSWER RECEIVED")

logging.info(f"Total correct: {total_correct}/{total_questions}")
print(f"Total correct: {total_correct}/{total_questions}")

# Collect video paths
video_paths = []
for i in range(initial_video, final_video):
    video_paths.append(os.path.join(cubicle, 'videos', f'ep_{i:02d}_a.mp4'))
    video_paths.append(os.path.join(cubicle, 'videos', f'ep_{i+1:02d}_a.mp4'))

logging.info(f"Saving final results for {total_questions} questions")
# Save all results
save_results(
    cubicle=cubicle,
    prompt=prompt,
    video_paths=video_paths,
    questions_dict=questions_dict,
    dict_results=dict_results,
    total_correct=total_correct,
    total_questions=total_questions,
    model_name=gemini_model
)

logging.info("Benchmark completed successfully")

