import asyncio
from multiprocessing import Pool
import os
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import time

from agents.response_evaluator import ResponseEvaluator
from agents.question_generator import QuestionGenerator
from database.chroma import get_chroma_collection, get_relevant_qa
from database.duckdb import get_sample_questions

UPLOAD_FOLDER = "uploads"

from typing import TypedDict

class UserData(TypedDict):
    api_key: str
    role: str
    company: str

@st.cache_data(show_spinner=False)
def welcome(role):
    st.write(f"Welcome to the HR interview for the " f"{role} position.")
    speak(f"Welcome to the HR interview for the " f"{role} position.")
    st.write("This app will ask you a series of HR interview questions.")
    speak("This app will ask you a series of HR interview questions.")
    st.write("Please provide your answers either by typing in the text box or by uploading an audio file.")
    speak("Please provide your answers either by typing in the text box or by uploading an audio file.")
    speak("When you are ready, Please click the Generate Questions button below to start your interview")

@st.cache_data(show_spinner=False)
def thanks():
    speak("Thank you very much for filling out the details. Please wait for the introduction page")

@st.cache_data(show_spinner=False)
def generate_questions(skill_tested, sample_questions, api_key):
    question_generator = QuestionGenerator(api_key)
    question_generator_question = question_generator.generate_question(skill_tested, sample_questions)
    return question_generator_question

def sidebar():
    # Omdena Logo
    with st.sidebar:
        st.image(image="images/omdena.png")
        st.markdown(
            body=(
                "<h2><center>Hyderabad, India</center></h2>"
                "<center><code>Interview Preparation Chatbot</code></center>"
                "<hr>"
            ),
            unsafe_allow_html=True,
        )

        with st.form(key="user_data_form", clear_on_submit=False):
            
            api_key = st.text_input(
                label="Your GOOGLE API KEY",
                help="Enter your Google api key",
            )
            # Role
            role = st.text_input(
                label="Role",
                placeholder="Data Scientist",
                help="Enter your role you are applying for (Data Scientist, Data Analyst, etc.)",
            )

            # Resume uploader
            company = st.text_input(
                label="Company Name",
                placeholder="Amazon",
                help="Enter your company name you are applying for (Google, Amazon, etc.)",
            )

            # Submit button
            submit = st.form_submit_button(label="Submit")

            # Validation
            if submit:
                if not role:
                    st.toast("Tell us the role you are applying for")
                if not company:
                    st.toast("Tell us your company name you are applying for")
                if not api_key:
                    st.toast("Tell us your API KEY")

                if role and company and api_key:
                    st.toast("Your data has been submitted successfully")

                    # Save user data
                    st.session_state.user_data = UserData(
                        role=role,
                        company=company,
                        api_key=api_key,
                    )

        st.divider()

        # Footer - Copyright info
        st.markdown(body="`©2023 by Omdena. All rights reserved.`", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    audio = AudioSegment.from_mp3("output.mp3")
    play(audio)

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    
async def feedback(questionnaire, api_key):
    st.header("Feedback")
    tasks = []
    for i, question_dict in enumerate(questionnaire):
        skill_to_test = question_dict['skills']
        user_response = st.session_state.user_responses[i]
        question = question_dict['question']
        relevant_qa = question_dict['answer']

        evaluator = ResponseEvaluator(api_key)
        task = asyncio.create_task(evaluator.evaluate_response(question, user_response, skill_to_test, relevant_qa))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    
    return results

@st.cache_data(show_spinner=False)
def generate_questionnaire(company, role, api_key):
    skills_to_test = ["social", "speaking", "management", "technical"]
    skill_question_map = {}
    for skill_to_test in skills_to_test:
        skill_question_map[skill_to_test] = get_sample_questions(company, role, skill_to_test, 5)

    sample_questions_list = [(k, v[0]) for k, v in skill_question_map.items()]

    with Pool(4) as p:
        questions = p.starmap(generate_questions, [(skill, sample, api_key) for skill, sample in sample_questions_list])
    db = get_chroma_collection("question_embeddings_v2")
    questionnaire_list = []
    for question, skill_to_test in zip(questions, skills_to_test):
        try:
            relevant_qa = get_relevant_qa(db, question, skill_question_map[skill_to_test][1])
        except:
            relevant_qa = []

        person_dict = {
            "question": question,
            "answer": relevant_qa,
            "skills": skill_to_test,
            "user_response": "",
        }
        questionnaire_list.append(person_dict)

    return questionnaire_list


def main():
    st.title("HR Interviewer")

    sidebar()
    user_data = st.session_state.get("user_data", {})

    if user_data:
        company = user_data['company']
        role = user_data['role']
        api_key = user_data['api_key']
        thanks()
        questionnaire = generate_questionnaire(company, role, api_key)

        if 'user_responses' not in st.session_state:
            st.session_state.user_responses = []
            welcome(role)
            if st.button("Generate Questions"):
                st.session_state.current_question_index = 0
        else:
            current_question_index = st.session_state.get('current_question_index', 0)

            if current_question_index < len(questionnaire):
                question_dict = questionnaire[current_question_index]
                user_answer_text = st.text_input(f"{question_dict['question']}")
                speak(f"{question_dict['question']}")
                user_answer_audio = st.file_uploader(f"Upload file for {question_dict['question']}", type=["wav", "mp3"])
                user_answer_file = None

                if user_answer_audio:
                    user_answer_file = save_uploaded_file(user_answer_audio)

                question_dict['user_response'] = user_answer_text if user_answer_text else user_answer_file

                if st.button("Next Question"):
                    if not question_dict['user_response']:
                        st.warning("Please provide an answer before moving to the next question.")
                    else:
                        current_question_index += 1
                        st.session_state.current_question_index = current_question_index
                        st.session_state.user_responses.append(question_dict['user_response'])

                        display_text = question_dict['user_response']
                        if question_dict['user_response'].startswith("uploads"):
                            display_text = transcribe_audio(question_dict['user_response'])

                        st.success(f"Your answer: {display_text}")

                        # Set the flag to True after speaking the text
                        question_dict['text_spoken'] = True
            else:
                st.warning("You have completed the questionnaire. Please find the feedback of your interview below.")
                speak("Congratulation you have finished the interview. Please click the feedback button to get the feedback of your interview.")
                
                if st.button("Feedback"):
                    feedback_results = asyncio.run(feedback(questionnaire, api_key))
                    st.write(feedback_results)
    else:
        st.warning("Please fill up the user data to start the interview.")
        time.sleep(0.5)
        speak("Please fill up the user data to start the interview.")

if __name__ == "__main__":
    main()