#%%
from google import genai
from pydantic import BaseModel
from typing_extensions import List, TypedDict, Dict
import re
import enum
import json
prompt = """ Please answer the following questions:
    
    **Multiple‑Choice Quiz (FFmpeg Performance & Usage)**  
*(Choose the best option for each question. Answers are **not** provided.)*

---

**Q1.** Which FFmpeg flag enables hardware decoding with automatic device selection?  
A. `-threads 0`  
B. `-hwaccel auto`  
C. `-codec copy`  
D. `-preset ultrafast`  

---

**Q2.** In FFmpeg, what does the option `-crf` primarily affect when encoding with `libx264`?  
A. Output frame rate  
B. Bit‑depth of the pixel format  
C. Constant rate factor (quality vs. size trade‑off)  
D. Audio sampling rate  

---

**Q3.** You need to generate a short color‑bar test clip without reading an input file. Which input specification is correct?  
A. `-i testbars.mp4`  
B. `-i null`  
C. `-f lavfi -i testsrc=size=1280x720`  
D. `-f pipe -i -`  

---

**Q4.** What is the main difference between presets `slow`, `medium`, and `fast` in `libx265`?  
A. They change the container format only  
B. They trade encoding speed for compression efficiency  
C. They switch from software to hardware encoding  
D. They alter the color space from BT.601 to BT.709  

---

**Q5.** When using `ffmpeg -benchmark`, which metric in the final summary indicates the total real‑world elapsed time?  
A. `user`  
B. `sys`  
C. `wall`  
D. `fps`

**Q6.** When using `ffmpeg -benchmark`, which metric in the final summary indicates the total real‑world elapsed time?  
A. `user`  
B. `sys`  
C. `wall`  
D. `fps`

**Q7.** When using `ffmpeg -benchmark`, which metric in the final summary indicates the total real‑world elapsed time?  
A. `user`  
B. `sys`  
C. `wall`  
D. `fps`

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

    """




class MultipleChoice(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

# Define a schema that allows for dynamic number of questions
class AnswerItem(TypedDict):
    question: str
    answer: MultipleChoice

class QuestionAnswers(TypedDict):
    answers: List[AnswerItem]

client = genai.Client(api_key="AIzaSyB0Q1KYjPjus3efd6zt4YX5RfjbkH1JJLQ")
response = client.models.generate_content(
    model='gemini-2.5-flash-preview-04-17',
    contents=prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': QuestionAnswers,
    },
)
# Use the response as a JSON string.
print(response.text)


answers_dict = json.loads(response.text)

# %%
dict_answers = {}
for answer in answers_dict['answers']:
    dict_answers[answer['question']] = answer['answer']

# %%
print(dict_answers)
# %%
