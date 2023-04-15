# 🤖 Conversational Chatbot for CSV Data 🤖
This project is an implementation of a conversational chatbot that can answer questions based on the data in a CSV file. The chatbot uses Streamlit for the web interface, Langchain for conversation logic, FAISS for storing and retrieving document embeddings, and OpenAI's GPT models for generating responses.

## Features
- Upload CSV files to use as a data source for the chatbot
- Select between different GPT models for response generation
- Reset the chat history at any time
- Simple and user-friendly interface

## Getting Started
Ensure that you have the following dependencies installed:
- Python 3.6 or higher
- Streamlit
- Langchain
- FAISS
- OpenAI

```
pip install requirements.txt
```

## Running the Application
```
streamlit run csvGPT.py
```
