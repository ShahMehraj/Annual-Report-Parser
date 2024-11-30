import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from io import StringIO

load_dotenv()

# Set OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Set OpenAI API key globally
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Update the prompt template
    prompt_template = """
    Extract the following details from the provided context. Focus on operational activities or product categories for business segments, and avoid listing geographical regions unless no other information is available:
    1. **Company Name**: The official name of the company.
    2. **Business Segments and Revenue**: List the various major operational activities or product categories that the company generates revenue from, along with the corresponding revenue for each segment. Avoid using geographical regions like "Americas" or "Europe" unless no specific business segments are available. For example:
    - For Apple Inc:
        - iPhone: $120 billion
        - Mac: $35 billion
        - iPad: $29 billion
        - Services: $80 billion
    - For Microsoft:
        - Software: $60 billion
        - Cloud Services: $70 billion
        - Gaming: $20 billion
    - For Abbott:
        - Established Pharmaceutical Products: $5.1 billion
        - Nutritional Products: $8.2 billion
        - Diagnostic Products: $10.0 billion
        - Medical Devices: $16.9 billion
    3. **Currency**: The currency in which the company reports its financials.

    If any of these details are not explicitly available in the context, respond with "Not Available."

    ### Context:
    {context}

    ### Question:
    "Provide the financial details as mentioned above, including revenue corresponding to each business segment."

    ### Answer Format:
    Output the information in the format:
    | company_name | business_segment              | currency | revenue |
    If the answer is not in the context, return "Not Available."
    """

    # Initialize the OpenAI chat model
    model = ChatOpenAI(model="gpt-4", temperature=0.3)  # Specify the model and temperature

    # Create the prompt template
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load the question-answering chain
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def process_pdf_and_extract_data(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]

def main():
    st.set_page_config("Chat PDF and Extract Financial Data")
    st.header("Extract Financial Data from Annual Reports 💁")

    user_question = "Extract company code, business segment, currency, and revenue from this document."

    if "data" not in st.session_state:
        st.session_state["data"] = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your Annual Reports (PDF Files)", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for pdf in pdf_docs:
                    raw_text = get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                    extracted_data = process_pdf_and_extract_data(user_question)
                    if extracted_data not in st.session_state['data']:
                        st.session_state['data'].append(extracted_data)
                st.success("Processing complete! Financial data extracted.")

    if st.button("Generate Excel File"):
        if st.session_state["data"]:
            frames = [pd.read_csv(StringIO(d.strip()), sep="|", skipinitialspace=True).iloc[1:] for d in st.session_state["data"]]
            result_df = pd.concat(frames, ignore_index=True)
            # Clean up column names and remove leading/trailing whitespace from values
            result_df.columns = [col.strip() for col in result_df.columns]
            result_df = result_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            result_df = result_df.iloc[:, 1:5]
    
            excel_path = "financial_data.xlsx"
            result_df.to_excel(excel_path, index=False)
            st.success("Excel file generated successfully!")
            st.download_button(
                label="Download Excel File",
                data=open(excel_path, "rb").read(),
                file_name="financial_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("No data available. Please process the PDFs first.")

if __name__ == "__main__":
    main()
