import os
import json
import numpy as np
import pandas as pd
from app.schemas.generation import QuizGenerationRequest
from uuid import UUID

def generate_quiz_universe(req: QuizGenerationRequest, uuid: UUID) -> str:
    """
    Generate a universe of quizzes.
    
    Args:
        size (int): Size of the universe
        
    Returns:
        list: List of quizzes
    """
    # create a directory for the data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    os.makedirs(f'data/{uuid}')
    # write a yml file with the request parameters
    with open(f'data/{uuid}/request_{uuid}.yml', 'w') as f:
        for key, value in req.dict().items():
            f.write(f"{key}: {value}\n")


    try:
        # Load MCQs from the provided URLs
        mcqs_list, num_topics = load_mcqs(req.MCQs, req.numTopics, req.listTopics, uuid)
    except Exception as e:
        raise ValueError(f"Error loading MCQs: {e}")
    
    universe, topic_to_idx = sample_combinations(mcqs_list, req.numQuizzes, req.numMCQs, num_topics, req.topicMode, req.levelMode, req.orderLevel)
    output_path = create_quiz_dataframe(universe, num_topics, topic_to_idx, req.numMCQs, uuid)
    
    return output_path


def load_mcqs(urls: list[str], numTopics: int, listTopics: list[str], uuid: UUID) -> tuple[list[dict], int]:
    """
    Load MCQs from the provided URLs and parse them into a list of dictionaries.
    Args:
        urls (list[str]): List of URLs pointing to CSV files containing MCQs.
    Returns:
        tuple: A tuple containing a list of MCQs and the number of topics.
    """
    if not urls:
        raise ValueError("No URLs provided for loading MCQs.")
    
    dataframes = []
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

    for path in urls:
        for encoding in encodings:
            try:
                df = pd.read_csv(path, sep=';', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not read file {path} with any of the tried encodings: {encodings}")
        dataframes.append(df)

    mcqs_df = pd.concat(dataframes, ignore_index=True)
    if mcqs_df.empty:
        raise ValueError("No data found in the provided dataset.")
    try:
        mcqs_list, num_topics = parse_dataset(mcqs_df, numTopics, listTopics, uuid)
    except ValueError as e:
        print(f"Error parsing dataset: {e}")
        mcqs_list = []
        num_topics = 0
    print(f"Generated {len(mcqs_list)} unique MCQs from {len(urls)} files with {num_topics} topics.")
    return mcqs_list, num_topics

def parse_dataset(df: pd.DataFrame, numTopics: int, listTopics: list[str], uuid: UUID) -> tuple[list[dict], int]:
    """
    Parse the dataset to create a dictionary of MCQs categorized by topic and difficulty.
    
    Args:
        dataset (pd.DataFrame): DataFrame containing MCQs with 'topic', 'difficulty', and 'id' columns.
        
    Returns:
        dict: Dictionary with topics as keys and dictionaries of difficulties as values.
    """

    df.rename(columns={'topic_id': 'topic', 'Level': 'difficulty'}, inplace=True)

    df.rename(columns={'id': 'mcq_id'}, inplace=True)
    df.rename(columns={'difficulty_level': 'difficulty'}, inplace=True)
    df['option_a'] = df['correct_answer']
    df.rename(columns={'correct_answer': 'correct_option', 'answer2': 'option_b', 'answer3': 'option_c', 'answer4': 'option_d'}, inplace=True)
    df['topic'] = df['topic'].astype(int)

    df['id'] = df.index
    df['difficulty'] = df['difficulty']
    # create a mapping between the values of column topic_id and column topic_name
    topic_mapping = df[['topic', 'topic_name']].drop_duplicates().set_index('topic')['topic_name'].to_dict()
    # Get unique topics
    unique_topics = df['topic'].unique()
    unique_topic_names = df['topic_name'].unique()
    
    if listTopics:  # If a list of topics is provided prioritize it
        # Filter topics based on provided list
        selected_topic_names = [topic for topic in unique_topic_names if str(topic).lower() in [t.lower() for t in listTopics]]
        # use the mapping to get the topic ids
        selected_topics = [topic for topic in unique_topics if topic_mapping.get(topic, '').lower() in [t.lower() for t in listTopics]]

        num_topics = len(selected_topics)
        if not selected_topics:
            raise ValueError("No valid topics found in the provided list.")
    else:  # If no list is provided, use the number of topics specified
        num_topics = numTopics
        if num_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")
        if num_topics > len(unique_topics):
            print(f"Warning: Requested {num_topics} topics but only {len(unique_topics)} available. Using all available topics.")
            num_topics = len(unique_topics)
        selected_topics = np.random.choice(unique_topics, num_topics, replace=False)
        print(f"Selected topics: {selected_topics}")

    # Filter MCQs by selected topics
    mcqs = df[df['topic'].isin(selected_topics)]

    mcqs.to_csv(f"data/{uuid}/mcqs_{uuid}.csv", index=False)
    print(f"Generated {len(mcqs)} unique MCQs")
    # Convert to list of dictionaries in the format required by RealDataGenerator
    mcqs_list = []
    for _, row in mcqs.iterrows():
        mcq_dict = {
            'id': row['id'],
            'topic': row['topic'],
            'difficulty': row['difficulty']
        }
        mcqs_list.append(mcq_dict)

    return mcqs_list, num_topics

def order_level_quizzes(universe: list[list[dict]], orderLevel: int) -> list[list[dict]]:
    """
    Order quizzes by difficulty level.
    
    Args:
        universe (list): List of quizzes, each represented as a list of MCQs.
        orderLevel (int): Order of difficulty levels (0 for ascending, 1 for descending).
        
    Returns:
        list: Ordered list of quizzes.
    """
    if orderLevel == 0:  # Ascending order
        return sorted(universe, key=lambda quiz: sum(mcq['difficulty'] for mcq in quiz))
    elif orderLevel == 1:  # Descending order
        return sorted(universe, key=lambda quiz: sum(mcq['difficulty'] for mcq in quiz), reverse=True)
    else:
        return universe  # No ordering applied

def sample_combinations(mcq_list: list[dict], numQuizzes: int, numMCQs: int, num_topics: int, topicMode: bool, levelMode: bool, orderLevel: int) -> tuple[list[list[dict]], dict]:
        """
        Generate a sample of the universe of quizzes.
        
        Args:
            numQuizzes (int): Number of quizzes to generate.
            topicMode (bool): Whether to sample from different topics.
            levelMode (bool): Whether to sample from different difficulty levels.
            orderLevel (int): Order of difficulty levels (0 for ascending, 1 for descending).
        Returns:
            list: List of quizzes, each represented as a list of MCQs.
        """
        data_dict = initialize_generator(mcq_list, numMCQs, num_topics, num_difficulties=6)
        print(
            f"Initialized generator with the following parameters:\n"
            f"  - Number of topics: {data_dict['num_topics']}\n"
            f"  - Number of difficulties: {data_dict['num_difficulties']}\n"
            f"  - Quiz size: {data_dict['quiz_size']}\n"
            f"  - Topics to index mapping: {data_dict['topic_to_idx']}\n"
            f"  - Difficulty to index mapping: {data_dict['difficulty_to_idx']}"
        )
        universe = []
        seen_quizzes = set()  # Track unique quizzes
        
        max_attempts = 1000  # Maximum attempts to generate a unique quiz
        attempts = 0
        
        while len(universe) < numQuizzes and attempts < max_attempts:
            quiz = generate_quiz(data_dict, topicMode, levelMode)
            # Create a unique key for the quiz by sorting MCQ IDs
            quiz_key = tuple(sorted(mcq['id'] for mcq in quiz))
            
            if quiz_key not in seen_quizzes:
                universe.append(quiz)
                seen_quizzes.add(quiz_key)
                attempts = 0  # Reset attempts counter on success
            else:
                attempts += 1
        
        if len(universe) < numQuizzes:
            print(f"Warning: Could only generate {len(universe)} unique quizzes out of requested {numQuizzes}")

        universe = order_level_quizzes(universe, orderLevel)
        
        return universe, data_dict['topic_to_idx']
        
def initialize_generator(mcq_list: list[dict], quiz_size: int, num_topics: int, num_difficulties: int = 6) -> dict:
    mcqs = mcq_list
    quiz_size = quiz_size
    num_topics = num_topics
    num_difficulties = num_difficulties
    
    # Create topic and difficulty mappings
    topic_to_idx = {topic: i for i, topic in enumerate(sorted(set(m['topic'] for m in mcqs)))}
    difficulty_to_idx = {i: i for i in range(num_difficulties)}
    
    # Group MCQs by topic and difficulty
    mcqs_by_topic_diff = {}
    for mcq in mcqs:
        topic = mcq['topic']
        difficulty = int(mcq['difficulty']) - 1  # Convert 1-based to 0-based
        if topic not in mcqs_by_topic_diff:
            mcqs_by_topic_diff[topic] = {}
        if difficulty not in mcqs_by_topic_diff[topic]:
            mcqs_by_topic_diff[topic][difficulty] = []
        mcqs_by_topic_diff[topic][difficulty].append(mcq)

    return {
        'mcqs_by_topic_diff': mcqs_by_topic_diff,
        'quiz_size': quiz_size,
        'num_topics': num_topics,
        'num_difficulties': num_difficulties,
        'topic_to_idx': topic_to_idx,
        'difficulty_to_idx': difficulty_to_idx
    }

def generate_quiz(data_dict: dict, topicMode: bool, levelMode: bool) -> list[dict]:
    """
    Generate a single quiz with diverse topic distribution.
    Returns:
        list: List of MCQs in the quiz
    """
    quiz = []
    mcqs_by_topic_diff = data_dict['mcqs_by_topic_diff']
    quiz_size = data_dict['quiz_size']
    num_topics = data_dict['num_topics']

    topics = list(mcqs_by_topic_diff.keys())
    # Distribute remaining MCQs randomly across sampled topics
    max_attempts = 100
    quiz_built = False
    counter = 0
    max_counter = 100

    while not quiz_built and counter < max_counter:
        attempts = 0
        chosen_difficulty = None
        if topicMode:
            # Randomly sample topics for the quiz (without replacement)
            sampled_topics = np.random.choice(topics, size=num_topics, replace=False)
        else:
            # Only cover a single random topic
            sampled_topics = [np.random.choice(topics)]
        quiz.clear()  # Clear quiz before each attempt

        while len(quiz) < quiz_size and attempts < max_attempts:
            topic = np.random.choice(sampled_topics)
            if levelMode:
                difficulties = list(mcqs_by_topic_diff[topic].keys())
                if not difficulties:
                    attempts += 1
                    continue
                difficulty = np.random.choice(difficulties)
            else:
                # For levelMode == False, pick a single difficulty for the whole quiz (first MCQ decides)
                if chosen_difficulty is None:
                    # Find intersection of available difficulties across all sampled topics
                    available_difficulties = set(mcqs_by_topic_diff[sampled_topics[0]].keys())
                    for t in sampled_topics[1:]:
                        available_difficulties &= set(mcqs_by_topic_diff[t].keys())
                    if not available_difficulties:
                        attempts += 1
                        continue
                    chosen_difficulty = np.random.choice(list(available_difficulties))
                difficulty = chosen_difficulty
            mcqs_pool = mcqs_by_topic_diff[topic].get(difficulty, [])
            if not mcqs_pool:
                attempts += 1
                continue
            mcq = np.random.choice(mcqs_pool)
            if mcq in quiz:
                attempts += 1
                continue
            quiz.append(mcq)
            attempts = 0  # Reset attempts after successful addition

        if len(quiz) == quiz_size:
            quiz_built = True
        else:
            # Try again with a different chosen_difficulty
            continue
        counter += 1
    if len(quiz) < quiz_size:
        raise ValueError(f"Could not generate a quiz of size {quiz_size} after multiple attempts. Not enough MCQs available.")
    return quiz

def create_quiz_dataframe(universe: list[str], num_topics: int, topic_to_idx: dict, numMCQs: int, uuid: UUID) -> str:
    # Create lists to store data for DataFrame
    quiz_data = []
    universe_array = []
    num_difficulties = 6 # This should match the number of difficulty levels in your dataset, right now its set for mathE
    for quiz_idx, quiz in enumerate(universe):
        # Calculate topic and difficulty distributions
        topic_dist = np.zeros(num_topics)
        difficulty_dist = np.zeros(num_difficulties)
        
        # Get MCQ IDs and calculate distributions
        mcq_ids = [mcq['id'] for mcq in quiz]
        for mcq in quiz:
            topic = mcq['topic']
            difficulty = mcq['difficulty'] - 1  # Convert to 0-based
            topic_dist[topic_to_idx[topic]] += 1
            difficulty_dist[difficulty] += 1
        
        # Normalize distributions
        topic_dist = topic_dist / len(quiz)
        difficulty_dist = difficulty_dist / len(quiz)
        
        # Create row data for DataFrame
        row_data = {
            'quiz_id': quiz_idx,
            **{f'mcq_{i+1}': mcq_id for i, mcq_id in enumerate(mcq_ids)},
            **{f'topic_coverage_{i}': cov for i, cov in enumerate(topic_dist)},
            **{f'difficulty_coverage_{i}': cov for i, cov in enumerate(difficulty_dist)}
        }
        quiz_data.append(row_data)
        
        # Add to universe array
        combined_dist = np.concatenate([topic_dist, difficulty_dist])
        universe_array.append(combined_dist)
    
    # Create and save DataFrame
    quizzes_df = pd.DataFrame(quiz_data)
    quizzes_df.to_csv(f'data/{uuid}/quizzes_{uuid}.csv', index=False)
    
    universe_array = np.array(universe_array)
    #save the array to a json file
    output_path = f"data/{uuid}/universe_{uuid}.json"
    with open(output_path, "w") as f:
        json.dump(universe_array.tolist(), f)
    
    print(f"Generated universe with {len(universe_array)} quizzes, each containing {numMCQs} MCQs.")
    
    return output_path 