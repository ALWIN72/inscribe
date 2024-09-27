import os
from groq import Groq
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Groq API client
client = Groq()

# Function to generate a topic and question based on user prompt
def generate_topic_and_question(prompt, difficulty_level):
    try:
        difficulty_descriptions = {
            1: "easy",
            2: "medium",
            3: "hard"
        }
        
        difficulty_description = difficulty_descriptions.get(difficulty_level, "easy")
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a {difficulty_description} topic and a question for short essay about: {prompt}"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        
        
    except Exception as e:
        print(f"Error calling the Groq API: {e}")
        return None, None

    response_text = response.choices[0].message.content.strip()
    topic, question = None, None
    
    if "Topic:" in response_text and "Question:" in response_text:
        try:
            topic = response_text.split("Topic:")[1].split("\n")[0].strip()
            question = response_text.split("Question:")[1].split("\n")[0].strip()
            
            if not topic or not question:
                raise ValueError("Topic or question is empty.")
                
        except (IndexError, ValueError) as e:
            print(f"Error parsing response: {e}")
            topic, question = None, None
    else:
        print("Couldn't generate topic and question. Please try again.")

    return topic, question

def check_spelling(text):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"Check for spelling in: {text}, provide suggestions"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        if response:
            response_text = response.choices[0].message.content.strip()
            errors = []
            if "Error" in response_text or "error" in response_text.lower():
                error_messages = response_text.split("\n")
                for error in error_messages:
                    if "Error" in error or "error" in error.lower():
                        errors.append(error.strip())
            return response, errors
        else:
            return None, []
    
    except Exception as e:
        print(f"Error checking spelling: {e}")
        return None, [str(e)]

def check_grammar(text):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": f"Check for grammar in: {text}"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        if response:
            response_text = response.choices[0].message.content.strip()
            errors = []
            if "Error" in response_text or "error" in response_text.lower():
                error_messages = response_text.split("\n")
                for error in error_messages:
                    if "Error" in error or "error" in error.lower():
                        errors.append(error.strip())
            return response, errors
        else:
            return None, []
    
    except Exception as e:
        print(f"Error checking grammar: {e}")
        return None, [str(e)]

# Cosine Similarity Function
def cosine_similarity_check(correct_answer, user_answer):
    vectorizer = CountVectorizer().fit_transform([correct_answer, user_answer])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]  # Return similarity score

# Function to check knowledge percentage on the topic
def knowledge_check(user_answer, correct_answer):
    # Calculate cosine similarity
    similarity_ratio = cosine_similarity_check(correct_answer, user_answer)
    knowledge_percentage = similarity_ratio * 100  # Convert to percentage
    
    print(f"Your knowledge percentage on the topic is: {knowledge_percentage:.2f}%")
    return knowledge_percentage

# Function to provide hints based on user's answer
def provide_hints(correct_answer, user_answer):
    similarity_ratio = cosine_similarity_check(correct_answer, user_answer)
    
    if similarity_ratio > 0.7:  # Adjust threshold as needed
        print("You're very close! Consider refining your answer.for the required 80% knowledge")
    elif similarity_ratio > 0.4:
        print("Good effort! Try to include more relevant details.")
    else:
        print("It seems like your answer could be more aligned with the question. Revisit the topic for better understanding.")


# Main quiz game loop
def quiz_game():
    user_prompt = input("Enter a topic or prompt (or type 'exit' to quit): ")
    
    # Exit if user types 'exit'
    if user_prompt.lower() == 'exit':
        print("Thank you for playing! Goodbye!")
        return
    
    # Set the initial difficulty level
    difficulty_level = 1
    
    # Generate the first topic and question
    topic, question = generate_topic_and_question(user_prompt, difficulty_level)
    
    if topic and question:
        print(f"Topic: {topic}")
        print(f"Question: {question}")
    else:
        print("Failed to generate topic and question.")
        return
    
    score = 0
    correct_answer = generate_correct_answer(topic)  # Generate correct answer based on the topic
    
    
    
    
    while True:
        user_answer = input("Enter your answer (or type 'exit' to quit): ")
        
        # Exit if user types 'exit'
        if user_answer.lower() == 'exit':
            print("Thank you for playing! Goodbye!")
            return
        
        # Check grammar and spelling
        grammar_response, grammar_errors = check_grammar(user_answer)
        spelling_response, spelling_errors = check_spelling(user_answer)
        
        if grammar_response is None or spelling_response is None:
            print("Error checking grammar or spelling. Please try again.")
            continue
        # Display errors and ask user to correct
        if grammar_errors or spelling_errors:
            print("Errors detected:")
            if grammar_errors:
                print("Grammar errors:")
                for error in grammar_errors:
                    print(f"- {error}")
            if spelling_errors:
                print("\nSpelling errors:")
                for error in spelling_errors:
                    print(error)
            
            correct = input("Do you want to correct these errors? (yes/no): ")
            if correct.lower() == 'yes':
                if grammar_response:
                    print("\nAPI Response (Grammar Check):")
                    print(grammar_response)
                if spelling_response:
                    print("\nAPI Response (Spelling Check):")
                    print(spelling_response)
                
                # Extract corrected answer from API response
                if grammar_response:
                    response_text = grammar_response.choices[0].message.content.strip()
                    corrected_grammar_answer = response_text.split("Corrected text: ")[-1].strip()
                    print(f"\nCorrected answer (grammar): {corrected_grammar_answer}")
                    user_answer = corrected_grammar_answer
                if spelling_response:
                    response_text = spelling_response.choices[0].message.content.strip()
                    corrected_spelling_answer = response_text.split("Corrected text: ")[-1].strip()
                    print(f"\nCorrected answer (spelling): {corrected_spelling_answer}")
                    user_answer = corrected_spelling_answer
        else:
            print("No errors detected.")
        # Check user's knowledge percentage
        knowledge_percentage = knowledge_check(user_answer, correct_answer)
        
        # Set similarity thresholds based on the level
        if difficulty_level == 1:
            similarity_threshold = 0.2  # 20% similarity for level 1
        elif difficulty_level == 2:
            similarity_threshold = 0.5  # 50% similarity for level 2
        elif difficulty_level == 3:
            similarity_threshold = 0.7  # 70% similarity for level 3 
        
        # Check user's answer against the correct answer using similarity
        if cosine_similarity_check(correct_answer, user_answer) > similarity_threshold:
            score += 1
            print(f"Correct! You earned {score} points.")
            difficulty_level += 1  # Increase difficulty level
            
            topic, question = generate_topic_and_question(topic, difficulty_level)
            if question:
                correct_answer = generate_correct_answer(topic)  # Generate a new correct answer
                print(f"Level {difficulty_level}: {question}")
            else:
                print("Couldn't generate a new question.")
        else:
            print("Incorrect. Try again!")
            provide_hints(correct_answer, user_answer)

def generate_correct_answer(topic):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": f"Provide a short essay about: {topic}"
            }
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        stream=False,
    )

    # Extract the content of the response
    response_text = response.choices[0].message.content.strip()  # Clean up the response
    
    # Check if the response is empty
    if not response_text:
        print("No answer generated. Please try again.")
        return None
    
    return response_text

if __name__ == "__main__":
    quiz_game()

    ''' while compliling in terminal -execute these only
    1)$env:GROQ_API_KEY='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'
    2)python app.py'''