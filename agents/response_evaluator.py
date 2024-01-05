import google.generativeai as genai
from typing import Dict


class ResponseEvaluator:
    def __init__(self, api_key) -> None:
        genai.configure(api_key = api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    async def evaluate_response(
        self, user_question: str, user_response: str, skill_tested: str, examples: list
    ) -> Dict:
        """
        Evalute the response and provide the feedback

        Parameters:
        -----------
        user_question: Question that is posed to the user
        user_response: Answer provided by the user
        skill_tested: Skill that is being tested
        examples: List of similar questions and answers from the internal dataset

        Returns:
        -----------
        Dictionary of the feedback containing question,answer and feedback
        """
        prompt = f"""You are a response evaluator responsible for providing rating and feedback for a particular
        question and answer provided to test {skill_tested}. If there are any relevant examples found, they will be shared otherwise you can evaluate based on your knowledge"""
        if len(examples) == 0:
            print("No relevant examples found")
        else:
            example_prompt = """
            ----------------
            Examples:
            """
            prompt += example_prompt
            for question, response, rating in examples:
                tmp = f"""
                Question: {question}
                Response: {response}
                Rating: {rating}

                """
                prompt += tmp
            prompt += "----------------"
        ending = f"""
        Now, provide the feeback and rating for the below question and response
        Question: {user_question}
        Response: {user_response}
        """
        prompt += ending
        response = await self.model.generate_content_async(prompt)
        result = {
            "Question": user_question,
            "Answer": user_response,
            "Feedback": response.text,
        }
        return result
