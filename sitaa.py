import langchain

#opensource models are easy to fine tune
#secure data can used to integrate
#customization
#can deploy on your base
#llamma
#mistral
#can acess using HuggingFace

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    #task="text-generation"
)

model = ChatHuggingFace(llm=llm)


result = model.invoke("what is the financial capital of india")

print(result.content)


import streamlit as st
from langchain_core.prompts import PromptTemplate

st.header("Research Tool")

user_input = st.text_input("enter prompt")

if st.button("summarize"):
             results = model.invoke(user_input)
             st.write(results.content)
    


