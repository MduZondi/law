import streamlit as st
import streamlit.components.v1 as components
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Set up Streamlit
st.set_page_config(page_title='Transcription and Summarization', layout="wide")

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'transcript_summary' not in st.session_state:
    st.session_state.transcript_summary = ""
if 'key_insights' not in st.session_state:
    st.session_state.key_insights = ""

# Initialize the ChatGoogleGenerativeAI model
google_api_key = "AIzaSyByN144UKV06XVeEm32okgCuos9HwhIv5Y"
llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=google_api_key, streaming=True)

# Function to stream responses
def stream_response(response):
    response_placeholder = st.empty()
    full_response = ""
    for chunk in response:
        if isinstance(chunk, dict) and 'text' in chunk:
            full_response += chunk['text']
        elif hasattr(chunk, 'content'):
            full_response += chunk.content
        else:
            full_response += str(chunk)
        response_placeholder.markdown(full_response + "▌")
    response_placeholder.markdown(full_response)
    return full_response

# Function to summarize transcript and highlight key insights
def summarize_and_highlight(transcript):
    summary_prompt = """
    Please summarize the following transcript and highlight key insights. The summary should be concise, and the key insights should be clearly identified:

    Transcript: {transcript}

    Summary:
    """
    insights_prompt = """
    Based on the following transcript, identify and highlight the most critical insights. Ensure that these insights are actionable and relevant:

    Transcript: {transcript}

    Key Insights:
    """
    
    summary_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=summary_prompt, input_variables=["transcript"]))
    insights_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=insights_prompt, input_variables=["transcript"]))
    
    summary = stream_response(summary_chain.stream({"transcript": transcript}))
    insights = stream_response(insights_chain.stream({"transcript": transcript}))
    
    return summary, insights

# Sidebar menu
menu = st.sidebar.selectbox("Choose a functionality", ["Transcription", "Summarization"])

if menu == "Transcription":
    st.title("Transcription Tool")

    # Embed the HTML for real-time transcription
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-time Audio Transcription</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 150px; margin-top: 10px; }
            button { font-size: 16px; padding: 10px 20px; margin-right: 10px; margin-top: 10px; }
            #status { margin-top: 10px; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Real-time Audio Transcription Tool</h1>
        <button id="startButton">Start Recording</button>
        <button id="stopButton" style="display:none;">Stop Recording</button>
        <div id="status">Status: Ready</div>
        <h2>Transcription:</h2>
        <textarea id="transcription" readonly></textarea>

        <script>
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const status = document.getElementById('status');
            const transcriptionArea = document.getElementById('transcription');

            let recognition;
            let transcription = '';

            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;

                recognition.onstart = () => {
                    status.textContent = 'Status: Listening...';
                };

                recognition.onresult = (event) => {
                    let interimTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            transcription += event.results[i][0].transcript + ' ';
                        } else {
                            interimTranscript += event.results[i][0].transcript;
                        }
                    }
                    transcriptionArea.value = transcription + interimTranscript;
                };

                recognition.onerror = (event) => {
                    console.error('Speech recognition error', event.error);
                    status.textContent = `Status: Error - ${event.error}`;
                };

                recognition.onend = () => {
                    status.textContent = 'Status: Stopped';
                    const streamlitTranscriptionEvent = new CustomEvent("transcriptionCompleted", { detail: transcription });
                    document.dispatchEvent(streamlitTranscriptionEvent);
                };

                startButton.onclick = () => {
                    transcription = '';
                    transcriptionArea.value = '';
                    recognition.start();
                    startButton.style.display = 'none';
                    stopButton.style.display = 'inline-block';
                    status.textContent = 'Status: Listening...';
                };

                stopButton.onclick = () => {
                    recognition.stop();
                    startButton.style.display = 'inline-block';
                    stopButton.style.display = 'none';
                };
            } else {
                startButton.style.display = 'none';
                stopButton.style.display = 'none';
                status.textContent = 'Web Speech API is not supported in this browser.';
            }
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=600)

    # Save and Download buttons for the transcript
    if st.session_state.transcript_text:
        st.subheader("Transcribed Text")
        st.text_area("Transcription", st.session_state.transcript_text, height=200)

        if st.button("Save Transcript"):
            st.success("Transcript saved successfully.")

        st.download_button(
            label="Download Transcript",
            data=st.session_state.transcript_text.encode(),
            file_name="transcript.txt",
            mime="text/plain"
        )

elif menu == "Summarization":
    st.title("Summarization Tool")

    transcript = st.text_area("Paste your transcript here or use the saved one:", st.session_state.transcript_text, height=200)

    if transcript:
        if st.button("Generate Summary and Insights"):
            with st.spinner('Processing transcript...'):
                try:
                    summary, insights = summarize_and_highlight(transcript)
                    st.session_state.transcript_summary = summary
                    st.session_state.key_insights = insights
                    st.success("Summary and insights generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        # Display results
        if st.session_state.transcript_summary:
            st.subheader("Transcript Summary")
            st.write(st.session_state.transcript_summary)
        
        if st.session_state.key_insights:
            st.subheader("Key Insights")
            st.write(st.session_state.key_insights)

        # Download button for summary and insights
        if st.session_state.transcript_summary or st.session_state.key_insights:
            combined_text = f"Summary:\n{st.session_state.transcript_summary}\n\nKey Insights:\n{st.session_state.key_insights}"
            st.download_button(
                label="Download Summary and Insights",
                data=combined_text.encode(),
                file_name="summary_and_insights.txt",
                mime="text/plain"
            )
