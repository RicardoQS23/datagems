# Application call:
In order to run the code, one has to install the correct dependencies by doing:
- pip3 install -r requirements.txt
Then run the cmd python3 -m app.main under the app directory.

# Description of routes:
## **1. quizzesGen** 
### This service takes as input a set of MCQs, the number of quizzes to be generated, the number of MCQs to be included in each quiz,the list of topics to address (or an arbitrary number of random topics to sample), and some properties for the quizzes and it outputs
### Input:
- **MCQs**: List[str] - A list of string URLs with path to CSV files with dataset of MCQs  
- **numQuizzes**: int - Number of quizzes to be generated.
- **numMCQs**: int - Number of MCQs to be included in each quiz.
- **numTopics**: int - Number of topics sampled from the initial MCQs dataset to choose randomly (DEFAULT)
- **listTopics**: List[str] - list of topics to filter from the initial MCQs dataset (OPTIONAL)
- **topicMode** : boolean - 0 for same topic and 1 for different topics (the function should not return an exception when numQuizzes is not met)
- **levelMode** : boolean - 0 for same level and 1 for different levels (11 is the random mode)
- **orderLevel** : boolean - 0 for ascending and 1 for descending order of MCQs in quizzes based on difficulty level
### Output:
- **PathToQuizzes**: str - Path to the folder with the CSV files.(output a message to the user to show what happend with the schema of the CSVs)
- **RequestID**: str - UUID Identifier of the method call
---
## **2. automaticQuiz**
### This service takes as input the topic proportions and the difficulty level proportions and returns the best quiz for the requested case using reinforcement learning. Specifically,
### Input:
- **dataUUID**: UUID - Identifier of the universe of quizzes generated from a call to the quizzesGen method to use for quiz inference.
- **teacherTopic**: List[float] - A list of floats, each corresponding to a topic and representing its percentage participation in the quiz.
- **teacherLevel**: List[float] - A list of floats, each corresponding to a level and representing its percentage participation in the quiz.
- **pathToModel**: str - string URL with path to the RL agent model invoked for inference
- **alfaValue**: float [0, 0.25, 0.5, 0.75, 1] - Objective weight regarding the teacher objectives (closer to 0 to maximize difficulty level, closer to 1 to maximize topic coverage).
### Output:
- **PathToQuiz**: str - Path to the folder with the JSON file
- **RequestID**: str - UUID Identifier of the method call
---

# Examples of call requests:
## **1. quizzesGen**
### Input
{
  "MCQs": [
    "data/math.csv"
  ],
  "numQuizzes": 10000,
  "numMCQs": 10,
  "listTopics": [
  ],
  "numTopics": 10,
  "topicMode": 1,
  "levelMode": 1,
  "orderLevel": 2
}

The previous input object simulates a request that generates an universe of quizzes from questions from the universe of quizzes identified by UUID **"868e14ff-5f92-408f-be95-11f8c265175a"** and with a teacher topic coverage of **"[0, 0, 0, 0, 0.5, 0, 0, 0.25, 0, 0.25]"** and teacher difficulty level goal of **"[0, 0.25, 0, 0.25, 0.5, 0]"**. The Reinforcement learning model to be called is present in **""models/dqn_t4_math_r2""** and the objective weight alfa is set to **0.5**.

### Output
{
  "PathToQuizzes": "data/868e14ff-5f92-408f-be95-11f8c265175a/universe_868e14ff-5f92-408f-be95-11f8c265175a.json",
  "RequestID": "868e14ff-5f92-408f-be95-11f8c265175a"
}

Along with this JSON object, the route also outputs a YAML file inside the **data/{UUID}** folder with some informations regarding the request call. It also stores the sampled MCQs and universe of quizzes in CSV files.

## **2. automaticQuiz**
### Input
{
  "dataUUID": "868e14ff-5f92-408f-be95-11f8c265175a",
  "teacherTopic": [
    0, 0, 0, 0, 0.5, 0, 0, 0.25, 0, 0.25
  ],
  "teacherLevel": [
    0, 0.25, 0, 0.25, 0.5, 0
  ],
  "pathToModel": "models/dqn_t4_math_r2",
  "alfaValue": 0.5
}

The previous input object simulates a request that asks for a quiz from the universe of quizzes identified by UUID **"868e14ff-5f92-408f-be95-11f8c265175a"** and with a teacher topic coverage of **"[0, 0, 0, 0, 0.5, 0, 0, 0.25, 0, 0.25]"** and teacher difficulty level goal of **"[0, 0.25, 0, 0.25, 0.5, 0]"**. The Reinforcement learning model to be called is present in **""models/dqn_t4_math_r2""** and the objective weight alfa is set to **0.5**.

### Output
{
  "PathToQuiz": "output_quizzes/5ed95a45-46ac-42b3-b0a0-ad3da1f87365/best_quiz_5ed95a45-46ac-42b3-b0a0-ad3da1f87365.json",
  "RequestID": "5ed95a45-46ac-42b3-b0a0-ad3da1f87365"
}

Along with this JSON object, the route also outputs a YAML file inside the **output_quizzes** folder with some informations regarding the request call.