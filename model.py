import streamlit as st
from PIL import Image
from fpdf import FPDF
import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import torch
import tempfile
import os
import gc  # Garbage collector to free memory

# Set environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Function to load Model 1
def load_model1():
     # Free GPU memory before loading the new model
    processor1 = AutoProcessor.from_pretrained("shashank23088/llava-onevision-qwen2-7b-traffic-merged")
    model1 = AutoModelForImageTextToText.from_pretrained("shashank23088/llava-onevision-qwen2-7b-traffic-merged")
    return processor1, model1


# Global variable for model and processor
model = None
processor = None

# Load the selected model dynamically
def load_selected_model(model_option):
    global model, processor
    if model_option == "Model 1 (llava-7b-traffic)":
        processor, model = load_model1()
# Page Configurations
st.set_page_config(page_title="🚗 Traffic Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS for customization
st.markdown(
    """
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .stSidebar {
            background-color: #f0f2f6;
        }
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            text-align: center;
            background-color: #f7f7f7;
            padding: 10px 0;
            color: #666;
        }
        .report-title {
            color: #1e88e5;
        }
        .analysis-section {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            background-color: #e3f2fd;
            margin-top: 20px;
        }
        .chat-bubble {
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            display: inline-block;
            max-width: 80%;
        }
        .user-message {
            background-color: #d1c4e9;
            text-align: left;
        }
        .system-message {
            background-color: #bbdefb;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application Header
st.title("🚦 Traffic Analysis Dashboard with Video Q&A")
st.subheader("Analyze Traffic Patterns, Incidents, and More with Advanced AI Models")

# Sidebar: User Inputs
st.sidebar.title("Upload Traffic Data")
uploaded_video = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4"])

# Model Selection
model_option = st.sidebar.selectbox(
    "Select AI Model",
    ["Model 1 (llava-7b-traffic)"]
)

# Load selected model
load_selected_model(model_option)

# Analysis Type Selector
analysis_type = st.sidebar.selectbox(
    "Select Type of Analysis",
    [
        "Real-Time Traffic Pattern Analysis",
        "Incident and Object Detection",
        "Traffic Flow Prediction",
        "Vehicle and License Plate Recognition",
        "Environmental Impact Analysis",
        "Traffic Violation Detection",
    ]
)

# Sidebar: Generate Report
st.sidebar.title("Generate Insights")
generate_report = st.sidebar.button("Generate Report")

# Function to process and get output based on a given question
def ask_question(question, uploaded_video, model, processor, analysis_result):
    question_with_context = f"{analysis_result}\n\nQuestion: {question}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_video.getbuffer())
        video_path = temp_video_file.name

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": question_with_context},
            ],
        }
    ]

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Move inputs to GPU (cuda)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")  # Move all inputs to GPU
    
    # Move the model to GPU (if not already)
    model = model.to("cuda")
    
    # Generate the answer
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the output text from the model

# Placeholder for analysis
def perform_analysis(analysis_type, model, processor):
    results = {
        "Real-Time Traffic Pattern Analysis": "Detected heavy traffic patterns in downtown areas. Peak congestion expected at 5-7 PM.",
        "Incident and Object Detection": "Detected 2 minor accidents and one roadblock on 5th Avenue.",
        "Traffic Flow Prediction": "Predicted congestion on major highways from 8-10 AM. Suggested rerouting options include Route A and Route B.",
        "Vehicle and License Plate Recognition": "Identified 5 vehicles with expired registration tags. Notified law enforcement for follow-up.",
        "Environmental Impact Analysis": "Increased emission levels detected in high-traffic zones. Suggested emission control measures for Route 10.",
        "Traffic Violation Detection": "Detected 15 speeding violations and 3 instances of red-light running. Generated violation alerts for registered owners."
    }
    return results.get(analysis_type, "Analysis type not recognized.")

# Function to generate PDF report
def generate_pdf_report(analysis_type, analysis_result):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Traffic Analysis Report", ln=True, align="C")
    
    pdf.set_font("Arial", "I", 12)
    pdf.cell(200, 10, txt=f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt=f"Analysis Type: {analysis_type}", ln=True)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, txt=f"Report Summary:\n{analysis_result}")
    
    pdf_output = "./Report.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Display uploaded video
if uploaded_video is not None:
    st.video(uploaded_video)

# Trigger analysis when the user clicks "Generate Report"
if generate_report and uploaded_video is not None:
    analysis_result = perform_analysis(analysis_type, model, processor)
    pdf_file = generate_pdf_report(analysis_type, analysis_result)
    
    st.write(f"PDF Report Generated Successfully!")
    st.download_button(label="Download PDF Report", data=open(pdf_file, "rb"), file_name="traffic_analysis_report.pdf")
    
    st.subheader("Traffic Analysis Report")
    st.write(f"Analysis Type: {analysis_type}")
    st.write(f"Report Summary: {analysis_result}")
    
    st.success("Report generated successfully!")

# Video Q&A Feature
st.subheader("Video Q&A")
st.write("Ask questions about the video, and our system will answer based on the analysis results.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Enter your question about the traffic analysis:")

if question and uploaded_video:
    analysis_result = perform_analysis(analysis_type, model, processor)  # Get analysis results based on the selected type
    answer = ask_question(question, uploaded_video, model, processor, analysis_result)

    # Save chat history and display
    st.session_state.chat_history.append((question, answer))
    for user_question, assistant_answer in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble user-message'>{user_question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble system-message'>{assistant_answer}</div>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f7f7f7;
            padding: 10px 0;
            text-align: center;
            font-size: 16px;  /* Increase font size */
            font-weight: bold;  /* Make the text bold */
            color: #666;
        }
    </style>
    <footer>
        © 2024 Team Visionary, LLM Course, IIIT Delhi
    </footer>
    """,
    unsafe_allow_html=True,
)

