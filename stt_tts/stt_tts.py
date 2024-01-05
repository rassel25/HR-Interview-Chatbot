from tempfile import NamedTemporaryFile
from gtts import gTTS
import speech_recognition as sr


class STT_TTS:
    def __init__(self):
        """
        Initializes the Moderator with a speech recognizer.
        """
        self.speech_recognizer = sr.Recognizer()

    def text_to_speech(self, text: str) -> str:
        """
        Converts text to speech and saves it to a temporary WAV file.

        Parameters
        ----------
        text : str
            The text to convert into audio.

        Returns
        -------
        str
            Filename of the temporary WAV file.
        """
        # Create a gTTS (Google Text-to-Speech) object for the given text
        speech = gTTS(text)

        # Create a named temporary file with a ".wav" extension for audio
        with NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp_file:
            # Write the speech into the temporary file
            speech.write_to_fp(audio_temp_file)

        # Return the filename of the temporary WAV file
        return audio_temp_file.name

    def speech_to_text(self, audio_filename: str) -> str:
        """
        Converts speech from an audio file to text.

        Parameters
        ----------
        audio_filename : str
            The filename of the audio file.

        Returns
        -------
        str
            The recognized text from the audio.
        """
        # Open the audio file for recognition
        with sr.AudioFile(audio_filename) as audio_file:
            # Record the audio data from the file
            audio_data = self.speech_recognizer.record(audio_file)

        recognized_text = ""
        try:
            # Recognize the text from the recorded audio using Google Speech Recognition
            recognized_text = self.speech_recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Error in requesting results from Google Speech Recognition service: {e}")

        return recognized_text
