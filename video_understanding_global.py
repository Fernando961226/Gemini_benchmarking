#%%
import os
import time
import json
from google import genai
import logging
from pathlib import PurePath
from google.genai.types import HttpOptions, Part


from utils import generate_content_with_retry, load_questions


# Set environment variables for Google Cloud and Vertex AI
os.environ['GOOGLE_CLOUD_PROJECT'] = 'gen-lang-client-0982482224'  # Replace with your actual project ID
os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
os.environ['GOOGLE_RESUMABLE_MEDIA_CHUNK_SIZE'] = str(10 * 1024 * 1024)


# FUNCTIONS ------------------------------------------------

def generate_questions(Q_dict,video_number):
    logging.info(f"Generating questions for video number {video_number}")
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
    logging.info(f"Found {len(questions_list)} questions for video {video_number}")

    questions_prompt_list = []
    for i, question in enumerate(questions_list):
        questions_prompt_list.append(question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n")
        for choice in list(Q_dict[question]["Multiple Choice"].keys()):
            questions_prompt_list[i] += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """
"""

    questions_prompt = prompt_start + "\n".join(questions_prompt_list) + "\n" + prompt_end
    logging.debug(f"Generated prompt: {questions_prompt}")
    return questions_prompt

def setup_logger(questions_file):
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join("Global Changes", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create log filename based on cubicle and timestamp
    base_name = PurePath(questions_file).stem
    log_filename = os.path.join(logs_dir, f"{timestamp}_{base_name}_results.log")
    
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
    
    logging.info(f"Starting video understanding benchmark for {questions_file}")
    return log_filename

def save_results(questions_file, prompt, video_paths, questions_dict, dict_answers, total_correct, total_questions, model_name):
    logging.info(f"Saving results for {questions_file}")
    # Create output filename based on input filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.splitext(questions_file)[0] + f"_{timestamp}_results.json"
    
    # Prepare the results dictionary
    results = {
        "model": model_name,
        "prompt": prompt,
        "video_paths": video_paths,
        "questions": questions_dict,
        "answers": dict_answers,
        "score": {
            "correct": total_correct,
            "total": total_questions,
            "percentage": (total_correct / total_questions) * 100 if total_questions > 0 else 0
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_filename}")


# INPUTS ------------------------------------------------

questions_file = "Global_Changes/Object_Counting_Questions.json"


gemini_model = "gemini-2.5-pro-preview-03-25"




# CODE ------------------------------------------------

log_filename = setup_logger(questions_file)
logging.info("Logger initialized")
logging.info(f"Using Gemini model: {gemini_model}")
logging.info(f"Using questions file: {questions_file}")
# Initialize client without API key since we're using Vertex AI
client = genai.Client(http_options=HttpOptions(api_version="v1"))
logging.info("GenAI client initialized with Vertex AI")
questions_dict = load_questions(questions_file)
logging.info(f"Loaded {len(questions_dict)} questions from {questions_file}")

initial_video = 0
final_video = questions_dict[list(questions_dict.keys())[-1]]["Final Video"]
logging.info(f"Processing videos from {initial_video} to {final_video}")

dict_answers = {}
for i in range(initial_video, final_video):
    logging.info(f"Processing video pair {i} and {i+1}")
    
    video_1 = Part.from_uri(
        file_uri=f"gs://gemini-api-global-videos/episode_{i}_720p_10fps.mp4",
        mime_type="video/mp4")
    logging.info(f"Loaded initial video: episode_{i}_720p_10fps.mp4")
    
    video_2 = Part.from_uri(
        file_uri=f"gs://gemini-api-global-videos/episode_{i+1}_720p_10fps.mp4",
        mime_type="video/mp4")
    logging.info(f"Loaded final video: episode_{i+1}_720p_10fps.mp4")

    prompt = generate_questions(questions_dict,i)
    logging.info("Generated questions prompt")
    
    result = generate_content_with_retry(client, gemini_model, [video_1, video_2, prompt])
    logging.info("Received response from Gemini model")
    
    print(result.text)
    
    answers_json = json.loads(result.text)
    logging.info(f"Parsed JSON response with {len(answers_json['answers'])} answers")

    for answer in answers_json['answers']:
        dict_answers[answer['question']] = answer['answer']
    
    # for f in client.files.list():
    #     client.files.delete(name=f.name)
    #     logging.info(f"Deleted temporary file: {f.name}")

logging.info("Completed processing all video pairs")
logging.info(f"Final answers dictionary: {dict_answers}")

total_correct = 0
total_questions = 0

for question in questions_dict:
    total_questions += 1
    if questions_dict[question]["Correct Choice"] == dict_answers[question]:
        total_correct += 1

logging.info(f"Scoring complete: {total_correct} correct out of {total_questions} questions")

# Collect video paths
video_paths = []
for i in range(initial_video, final_video):
    video_paths.append(f'gs://gemini-api-global-videos/episode_{i}_720p_10fps.mp4')
    video_paths.append(f'gs://gemini-api-global-videos/episode_{i+1}_720p_10fps.mp4')
logging.info(f"Collected {len(video_paths)} video paths")

# Save all results
save_results(
    questions_file=questions_file,
    prompt=prompt,
    video_paths=video_paths,
    questions_dict=questions_dict,
    dict_answers=dict_answers,
    total_correct=total_correct,
    total_questions=total_questions,
    model_name=gemini_model
)
logging.info("Benchmark execution completed successfully")


