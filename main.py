import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
genai.configure(api_key="AIzaSyBtXDeujD8MIhE4jJODOBLiejs8J_DUcPY")
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                              generation_config=generation_config,
                              )

convo = model.start_chat(history=[])


def search_pdf(chunks, question):
  relevant_chunks = []
  for chunk in chunks:
    if question.lower() in chunk.lower():
      relevant_chunks.append(chunk)
  return '\n'.join(relevant_chunks)

def add_history(user_input, response):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({"sender": "ChatBot", "content": response})
    st.session_state.messages.append({"sender": "You", "content": user_input})

    reversed_messages = list(reversed(st.session_state.messages))

    for id, message in enumerate(reversed_messages):
        st.sidebar.text_area(f"{id}. {message['sender']}",
                             message["content"],
                             disabled=True,
                             height=int(len(message["content"]) / 3))
st.set_page_config(
    page_title="RAM AI Chatbot",
    page_icon="m.jpeg",
)
def main():
  try:
      st.markdown("""
            <style>
            .reportview-container {
                background: #002b36;
            }
            .main {
                color: #fafafa;
                background-color: #586e75;
            }
            body {
                color: #fafafa;
                font-family: "Sans Serif";
            }
            </style>
            """, unsafe_allow_html=True)
      st.title("RAM AI Chatbot")

      pdf = st.file_uploader("upload file ", type="pdf")

      if pdf is not None:

          pdf_reader = PdfReader(pdf)
          text = ""
          for page in pdf_reader.pages:
              text += page.extract_text()
          splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                    chunk_overlap=300)
          chunks = splitter.split_text(text)

          for chunk in chunks:
              if chunk.strip():
                  if isinstance(chunk, str) and chunk.strip() != "":
                      try:
                          convo.send_message(chunk)
                          break  # If the request is successful, break the loop
                      except Exception as e:
                          if "contents.parts must not be empty" in str(e):
                              print(f"Skipping empty chunk.")
                              break
                          else:
                              print(f"Error: {e}. Retrying...")
                              time.sleep(5)  # Wait for 5 seconds before retrying
                  else:
                      print(f"Invalid chunk: {chunk}")
      st.write(
          "Hi there! How can I assist you today? Ask about the PDF by starting with 'QUESTION:'"
      )
      col1, col2 = st.columns([3, 1])

      x = col1.text_input("Enter message or question here")
      col2.markdown("""
        <style>
        /* CSS */
        .button-85 {
          padding: 0.6em 2em;
          border: none;
          outline: none;
          color: rgb(255, 255, 255);
          background: #111;
          cursor: pointer;
          position: relative;
          z-index: 0;
          border-radius: 10px;
          user-select: none;
          -webkit-user-select: none;
          touch-action: manipulation;
          margin-top: 25px;
        }

        .button-85:before {
          content: "";
          background: linear-gradient(
            45deg,
            #ff0000,
            #ff7300,
            #fffb00,
            #48ff00,
            #00ffd5,
            #002bff,
            #7a00ff,
            #ff00c8,
            #ff0000
          );
          position: absolute;
          top: -2px;
          left: -2px;
          background-size: 400%;
          z-index: -1;
          filter: blur(5px);
          -webkit-filter: blur(5px);
          width: calc(100% + 4px);
          height: calc(100% + 4px);
          animation: glowing-button-85 20s linear infinite;
          transition: opacity 0.3s ease-in-out;
          border-radius: 10px;
        }

        @keyframes glowing-button-85 {
          0% {
            background-position: 0 0;
          }
          50% {
            background-position: 400% 0;
          }
          100% {
            background-position: 0 0;
          }
        }

        .button-85:after {
          z-index: -1;
          content: "";
          position: absolute;
          width: 100%;
          height: 100%;
          background: #222;
          left: 0;
          top: 0;
          border-radius: 10px;
        }
        </style>
        <button class="button-85" role="button">Enter</button>
        """, unsafe_allow_html=True)
  except Exception as e:
    st.write(f"Error reading PDF {e}")

  if x.startswith("QUESTION:"):
      try:
          question = x[9:]
          relevant_chunks = search_pdf(chunks, question)

          prompt = "Answer the question based on the provided context: {} \n Context: {}".format(
              question, relevant_chunks)
          response = model.generate(prompt)
          st.markdown(
              f'<div style="color: white; background-color: black; padding: 10px; border-radius: 5px;">{response}</div>',
              unsafe_allow_html=True)
      except Exception as e:
            st.write(f"Error processing question: {e}")
  else:
      try:
          if x.strip():
              convo.send_message(x)
              if convo.last is not None:
                  st.markdown(
                      f'<div style="color: white; background-color: black; padding: 10px; border-radius: 5px;">{convo.last.text}</div>',
                      unsafe_allow_html=True)
                  add_history(x, convo.last.text)
      except Exception as e:
            st.write(f"Error processing message: {e}")

if __name__ == '__main__':
  main()
