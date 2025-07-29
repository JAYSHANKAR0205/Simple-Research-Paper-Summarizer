# Normal Ui Based Project Based on LangChain and Ollama
# This code is a simple Streamlit application that allows users to summarize research papers using the Ollama LLM.

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import streamlit as st
llm = OllamaLLM(model="llama3", temperature=1.2)

st.header("Research Paper Summarizer")
paper_input = st.selectbox("Select a research paper to summarize:",["Attention Is All You Need", "BERT: Pre-training of  Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners"])
style_input = st.selectbox("Select a summary style:", ["Beginnners Friendly", "Technical", "Mathematical","Code-oriented"])
length_input = st.selectbox("Select summary length:", ["Short(1-2 paragraphs)", "Medium(5-6 paragraphs)", "Long(detailed explanation)"])
template=PromptTemplate(
    template="""
    You are a research paper summarizer. Your task is to summarize the {paper} research paper in a {style} style with a {length} length.
    """,
    input_variables=["paper", "style", "length"],
    validate_template=True
)


chain = template | llm
if st.button("Generate Summary"):
    result = chain.invoke({
        "paper": paper_input,
        "style": style_input,
        "length": length_input
    })
    st.write(result)