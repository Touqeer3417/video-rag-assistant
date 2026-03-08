import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

st.title("YouTube RAG Chatbot")

video_url = st.text_input("Paste YouTube Video URL")

if st.button("Process Video") and video_url.strip():

    try:
        # Load transcript using YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False
        )

        docs = loader.load()

        full_text = docs[0].page_content

        st.write("Transcript successfully fetched!")

        # Split transcript
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.create_documents([full_text])

        # Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Vector DB
        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4}
        )

        st.session_state.retriever = retriever

        st.success("Video processed successfully! Ask questions below.")

    except Exception as e:
        st.error(f"Error loading video: {e}")


# Question section
question = st.text_input("Ask a question about the video")

if question and "retriever" in st.session_state:

    retriever = st.session_state.retriever
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        template="""
    You are an AI assistant answering questions about a YouTube video.

    Answer the question ONLY using the transcript context provided below.

    Rules:
    - Do NOT make up information.
    - If the answer is not present in the transcript, say: "The transcript does not contain this information."
    - Keep the answer clear and concise.
    - Use only the information from the transcript.

    Transcript Context:
    {context}

    User Question:
    {question}

    Helpful Answer:
    """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    chain = parallel_chain | prompt | llm | parser

    answer = chain.invoke(question)

    st.write("### Answer")
    st.write(answer)