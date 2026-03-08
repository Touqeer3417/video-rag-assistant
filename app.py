import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

st.title("YouTube RAG Chatbot")


video_id = st.text_input("Paste YouTube Video ID")

if st.button("Process Video") and video_id.strip():

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        full_text = ""

        for line in transcript:
            full_text += line.text + " "

       

        st.write("Transcript successfully fetched!")

        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.create_documents([full_text])

        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4}
        )

        # Save retriever in session state
        st.session_state.retriever = retriever

        st.success("Video processed successfully! You can now ask questions below.")

    except TranscriptsDisabled:
        st.error("Transcript is disabled for this video.")

    except NoTranscriptFound:
        st.error("No transcript found for this video.")

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")


# Question input
question = st.text_input("Ask a question about the video")

if question and "retriever" in st.session_state:
    retriever = st.session_state.retriever
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt template
    prompt = PromptTemplate(
        template="""
You are an AI assistant that answers questions about a YouTube video.

Use ONLY the transcript context below. Do NOT make up any answers.

Transcript Context:
{context}

User Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    # Format retrieved chunks
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser

    answer = chain.invoke(question)

    st.write("**Answer:**")
    st.write(answer)