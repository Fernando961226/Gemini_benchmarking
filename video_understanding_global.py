#%%
import io, os
import time
import json
from google import genai
import logging
from pathlib import PurePath



from utils import check_file_active, QuestionAnswers, generate_content_with_retry, load_api_key, load_questions


os.environ['GOOGLE_RESUMABLE_MEDIA_CHUNK_SIZE'] = str(10 * 1024 * 1024)





def generate_questions(Q_dict,video_number):
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


    questions_prompt_list = []
    for i, question in enumerate(questions_list):
        questions_prompt_list.append(question + ":\n" + "Question: " + Q_dict[question]["Question"] + "\n")
        for choice in list(Q_dict[question]["Multiple Choice"].keys()):
            questions_prompt_list[i] += choice + ": " + Q_dict[question]["Multiple Choice"][choice] + "\n"

    prompt_end = """
"""

    questions_prompt = prompt_start + "\n".join(questions_prompt_list) + "\n" + prompt_end

    return questions_prompt

def setup_logger(questions_file):
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join("Local Changes", "logs")
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





questions_file = "Object_State_Questions.json"

gemini_model = "gemini-2.5-pro-preview-03-25"


api_key = load_api_key()

client = genai.Client(api_key=api_key)

questions_dict = load_questions(questions_file)


initial_video = 0
final_video = questions_dict[list(questions_dict.keys())[-1]]["Final Video"]

dict_answers = {}
for i in range(initial_video, final_video):

    video_1 = client.files.upload(file=f'/home/fernando/Gemini Benchmark/Global_Changes/episode_{i}_720p_10fps.mp4')
    video_2 = client.files.upload(file=f'/home/fernando/Gemini Benchmark/Global_Changes/episode_{i+1}_720p_10fps.mp4')


    # Poll until the video files are completely processed (state becomes ACTIVE)
    video_1 = check_file_active(client, video_1)
    video_2 = check_file_active(client, video_2)
    # Once ACTIVE, use the file

    prompt = generate_questions(questions_dict,i)

    result = generate_content_with_retry(client, gemini_model, [video_1,  video_2, prompt])

    
    print(result.text)
    
    answers_json = json.loads(result.text)

    for answer in answers_json['answers']:
        dict_answers[answer['question']] = answer['answer']
    
    for f in client.files.list():
        client.files.delete(name=f.name)
    print("Deleted:", f.name)
print("Answers: --------------------------------")
print(dict_answers)    

def save_results(questions_file, prompt, video_paths, questions_dict, dict_answers, total_correct, total_questions, model_name):
    """
    Save benchmark results to a JSON file.
    
    Args:
        questions_file: Original questions file name
        prompt: The prompt used for the model
        video_paths: List of video file paths
        questions_dict: Dictionary of questions
        dict_answers: Dictionary of model answers
        total_correct: Number of correct answers
        total_questions: Total number of questions
        model_name: Name of the Gemini model used
    """
    # Create output filename based on input filename
    output_filename = os.path.splitext(questions_file)[0] + "_results.json"
    
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
    
    print(f"Results saved to {output_filename}")

total_correct = 0
total_questions = 0

for question in questions_dict:
    total_questions += 1
    if questions_dict[question]["Correct Choice"] == dict_answers[question]:
        total_correct += 1

print(f"Total correct: {total_correct}/{total_questions}")

# Collect video paths
video_paths = []
for i in range(initial_video, final_video):
    video_paths.append(f'/home/fernando/Gemini Benchmark/Global_Changes/episode_{i}_720p_10fps.mp4')
    video_paths.append(f'/home/fernando/Gemini Benchmark/Global_Changes/episode_{i+1}_720p_10fps.mp4')

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


