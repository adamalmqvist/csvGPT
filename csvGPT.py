import os
import pickle
import streamlit as st
import tempfile
import pandas as pd
import asyncio

from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS


st.set_page_config(layout="wide", page_icon="🤖", page_title="csvGTP")

st.markdown("<h1 style='text-align: center;'>csvGTP</h1>", unsafe_allow_html=True)
async def main():
   
   user_api_key = st.sidebar.text_input(label="#### Your OpenAI API key", placeholder="Paste your openAI API key, sk-", type="password")
               
   os.environ["OPENAI_API_KEY"] = user_api_key
   uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="hidden")
   
   if uploaded_file is not None:
       def show_user_file(uploaded_file):
           file_container = st.expander("Your CSV file :")
           shows = pd.read_csv(uploaded_file)
           uploaded_file.seek(0)
           file_container.write(shows)
           
       show_user_file(uploaded_file)
    
   if uploaded_file :
            try :
                async def storeDocumentEmbeddings(file, filename):
                    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
                        tmp_file.write(file)
                        tmp_file_path = tmp_file.name

                    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                    data = loader.load()

                    embeddings = OpenAIEmbeddings()
                    vectors = FAISS.from_documents(data, embeddings)
                    os.remove(tmp_file_path)

                    with open(filename + ".pkl", "wb") as f:
                        pickle.dump(vectors, f)

                async def getDocumentEmbeddings(file, filename):
                    if not os.path.isfile(filename + ".pkl"):
                        await storeDocumentEmbeddings(file, filename)

                    with open(filename + ".pkl", "rb") as f:
                        global vectores
                        vectors = pickle.load(f)

                    return vectors

                async def conversational_chat(query):
                    result = chain({"question": query, "chat_history": st.session_state['history']})
                    st.session_state['history'].append((query, result["answer"]))

                    print("Log: ")
                    print(st.session_state['history'])

                    return result["answer"]

                with st.sidebar.expander("Settings", expanded=False):
                    if st.button("Reset Chat"):
                        st.session_state['reset_chat'] = True

                    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','gpt-4'])

                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'ready' not in st.session_state:
                    st.session_state['ready'] = False

                if 'reset_chat' not in st.session_state:
                    st.session_state['reset_chat'] = False

                if uploaded_file is not None:
                    with st.spinner("Processing..."):
                        uploaded_file.seek(0)
                        file = uploaded_file.read()
                        vectors = await getDocumentEmbeddings(file, uploaded_file.name)

                        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name=MODEL),
                                                                      retriever=vectors.as_retriever())

                    st.session_state['ready'] = True

                if st.session_state['ready']:
                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

                    if 'past' not in st.session_state:
                        st.session_state['past'] = ["Hey ! 👋"]

                    response_container = st.container()
                    container = st.container()

                    with container:
                        with st.form(key='my_form', clear_on_submit=True):
                            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                            submit_button = st.form_submit_button(label='Send')

                            if st.session_state['reset_chat']:
                                st.session_state['history'] = []
                                st.session_state['past'] = ["Hey ! 👋"]
                                st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]
                                response_container.empty()
                                st.session_state['reset_chat'] = False

                        if submit_button and user_input:
                            output = await conversational_chat(user_input)
                            st.session_state['past'].append(user_input)
                            st.session_state['generated'].append(output)

                    if st.session_state['generated']:
                        with response_container:
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state["past"][i], is_user=True, key=str(i) +'_user', avatar_style="big-smile")
                                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            except Exception as e:
             st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())