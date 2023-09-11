import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare():
    df = pd.read_csv("./data/amazon_fashion_review_tags.csv")
    product_list = df['asin'].unique()
    for p_id in product_list:
        print(p_id)
        chunk = df[df['asin'] == p_id]['reviewText']
        review_list = chunk.tolist()
        reviews = '\n'.join(review_list)

        text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1000)
        documents = text_splitter.create_documents([reviews])

        db = FAISS.from_documents(documents, HuggingFaceEmbeddings())
        print("saving...")
        db.save_local(f"db/{p_id}")
        print("==============")

def get_ai_answer(product_id):
    selected_db = FAISS.load_local(f"db/{product_id}", HuggingFaceEmbeddings())

    question = "how does the button fit?"

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=selected_db.as_retriever(), return_source_documents=True)
    res = qa_chain({"query": question})
    # print(res)
    return res["result"], res["source_documents"]

if __name__ == "__main__":
    # prepare()
    
    st.title("Review Search Engine")

    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        product_list = []
        for dirname in os.listdir("db"):
            if not dirname.startswith("."):
                product_list.append(dirname)

        # print(product_list)
        product_id = st.selectbox('Select a product', product_list)
        if product_id:
            answer, docs = get_ai_answer(product_id)
            with st.container():
                st.subheader("AI Answer")
                st.write(answer)
            
                st.subheader("The answer is based on")
                for i in docs:
                    st.write(i)
    else:
        st.warning("WARNING: Enter your OpenAI API key!")

    






    