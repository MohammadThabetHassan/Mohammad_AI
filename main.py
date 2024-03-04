import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# API key and configuration (replace with your own)
genai.configure(api_key="AIzaSyBtXDeujD8MIhE4jJODOBLiejs8J_DUcPY")
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[])

# Initialize conversation history
conversation_history = {}


def search_pdf(chunks, question):
  relevant_chunks = []
  for chunk in chunks:
    if question.lower() in chunk.lower():
      relevant_chunks.append(chunk)
  return '\n'.join(relevant_chunks)


def add_history(user_input, response):
  conversation_history[user_input] = response
  return conversation_history


def main():
  st.title("Mohammad AI Chatbot")
  pdf = st.file_uploader("upload file ", type="pdf")

  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
    chunks = splitter.split_text(text)

    for chunk in chunks:
      if chunk.strip():
       convo.send_message(chunk)

  st.write(
      "Hi there! How can I assist you today? Ask about the PDF by starting with 'QUESTION:'"
  )
  x = str(st.text_input("Enter message or question here"))

  if x.startswith("QUESTION:"):
    question = x[9:]
    relevant_chunks = search_pdf(chunks, question)

    prompt = "Answer the question based on the provided context: {} \n Context: {}".format(
        question, relevant_chunks)
    response = model.call(prompt)
    st.write(response)
    conversation_history[question] = {
        'context': relevant_chunks,
        'response': response
    }

  else:
     if x.strip():
      convo.send_message(x)
      if convo.last is not None:
         st.write(convo.last.text)
         conversation_history[x] = convo.last.text


if __name__ == '__main__':
  main()
