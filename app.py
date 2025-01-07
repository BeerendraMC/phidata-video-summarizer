import tempfile
import time
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from google.generativeai import get_file, upload_file
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide",
)

st.title("Phidata AI Video Summarizer Agent")
st.header("Powered by Gemini 2.0 Flash Exp")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="AI Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


## Initialize the Agent
multimodal_agent = initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis",
    accept_multiple_files=False,
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from this video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and summarize it for you.",
        help="Provide specific questions or insights you want from the video",
    )

    if st.button("Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please provide a question or insights to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Prompt generation for analysis
                    analysis_prompt = f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research,
                        {user_query}
                        
                        Provide a detailed, user-friendly, and actionable response.
                        """

                    # AI agent processing
                    response = multimodal_agent.run(
                        analysis_prompt, videos=[processed_video]
                    )

                # Display results
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary files
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to analyze.")

# Customize text area
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
