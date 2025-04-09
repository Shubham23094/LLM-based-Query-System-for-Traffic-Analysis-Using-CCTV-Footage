<h1>LLM-based Query System for Traffic Analysis Using CCTV Footage</h1>
<p>This repository contains the code and resources for developing a <strong>LLM-based Query System</strong> for analyzing CCTV footage to extract traffic statistics and detect accidents. The pipeline integrates state-of-the-art models for video analysis, including YOLO for object detection and LLAVA Onevision for multimodal inference.</p>

<h2>Table of Contents</h2>
<ul>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#pipeline-overview">Pipeline Overview</a></li>
<li><a href="#features">Features</a></li>
<li><a href="#requirements">Requirements</a></li>
<li><a href="#setup">Setup</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#dataset-details">Dataset Details</a></li>
<li><a href="#evaluation-metrics">Evaluation Metrics</a></li>
<li><a href="#results">Results</a></li>
<li><a href="#model-deployment">Model Deployment</a></li>
<li><a href="#future-work">Future Work</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#references">References</a></li>
</ul>

<hr>

<h2 id="introduction">Introduction</h2>
<p>This project aims to develop a system that processes natural language queries on CCTV footage to:</p>
<ul>
<li>Identify specific vehicles based on attributes (e.g., type, color).</li>
<li>Count and classify vehicles.</li>
<li>Detect traffic accidents.</li>
<li>Generate statistical reports for traffic management.</li>
</ul>
<p>The system combines transformer-based large language models (LLMs) with computer vision pipelines for robust and efficient traffic analysis.</p>

<h2 id="pipeline-overview">Pipeline Overview</h2>
<p>The main pipeline consists of the following steps:</p>
<ol>
<li>
<strong>Frame Extraction</strong>:
<ul>
<li>Extract frames from video files at regular intervals.</li>
</ul>
</li>
<li>
<strong>Vehicle Detection and Annotation</strong>:
<ul>
<li>Use YOLOv5 to detect and classify vehicles into categories such as cars, trucks, bikes, and bicycles.</li>
<li>Generate annotations in JSON format for training and validation.</li>
</ul>
</li>
<li>
<strong>Inference</strong>:
<ul>
<li>Use the <code>LlavaOnevisionForConditionalGeneration</code> model to process annotated frames and generate responses.</li>
</ul>
</li>
<li>
<strong>Finetuning</strong>:
<ul>
<li>Employ QLoRA for finetuning on custom datasets.</li>
<li>Optimize recall and precision for accident detection and vehicle classification tasks.</li>
</ul>
</li>
<li>
<strong>Evaluation</strong>:
<ul>
<li>Evaluate using metrics like Accuracy, F1 Score, Precision, Recall, and Mean Absolute Error (MAE).</li>
</ul>
</li>
</ol>

<h2 id="features">Features</h2>
<ul>
<li><strong>Accident Detection</strong>: Identify traffic accidents in video footage.</li>
<li><strong>Vehicle Counting</strong>: Count vehicles by type (car, truck, bike, bicycle).</li>
<li><strong>Interactive Video QA</strong>: Query video data using natural language.</li>
<li><strong>Efficient Finetuning</strong>: Use QLoRA to adapt large LLMs efficiently.</li>
<li><strong>Dataset Annotation</strong>: Automatically generate and preprocess dataset annotations.</li>
</ul>

<h2 id="requirements">Requirements</h2>
<ul>
<li>Python 3.9+</li>
<li>CUDA-enabled GPU for faster processing.</li>
<li>Libraries:
<ul>
<li>PyTorch</li>
<li>Hugging Face Transformers</li>
<li>OpenCV</li>
<li>YOLOv5</li>
<li>WandB</li>
<li>Jupyter Notebook</li>
</ul>
</li>
</ul>

<h2 id="setup">Setup</h2>
<h3>Clone the Repository</h3>
<pre><code>git clone https://github.com/Shubham23094/LLM-based-Query-System-for-Traffic-Analysis-Using-CCTV-Footage.git
cd LLM-based-Query-System-for-Traffic-Analysis-Using-CCTV-Footage</code></pre>

<h3>Install Dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>Download the Datasets</h3>
<ul>
<li><strong>DoTA</strong>: Download and place the dataset in the datasets/dota directory.</li>
<li><strong>HEVI</strong>: Download and place the dataset in the datasets/hevi directory.</li>
</ul>

<h3>Configure Paths in the Notebook</h3>
<ul>
<li>Open <code>/llava_finetune_lora_hf.ipynb</code> and update the paths for the datasets and outputs.</li>
</ul>

<h2 id="usage">Usage</h2>
<h3>Running the Pipeline</h3>
<ol>
<li>
<strong>Open the Jupyter Notebook</strong>:
<pre><code>jupyter notebook</code></pre>
</li>
<li>
<strong>Navigate to <code>/llava_finetune_lora_hf.ipynb</code></strong> and run the cells sequentially to:
<ul>
 <li>Preprocess datasets.</li>
 <li>Train and finetune the model using QLoRA.</li>
 <li>Generate inference results.</li>
</ul>
</li>
</ol>

<h3>Command Line Usage</h3>
<p>To run the finetuning script directly:</p>
<pre><code>python llava_finetune.py --data_dir ./datasets --output_dir ./llava_finetuned</code></pre>

<h2 id="dataset-details">Dataset Details</h2>
<h3>DoTA Dataset</h3>
<ul>
<li><strong>Description</strong>: Accident detection dataset with 1,450 annotated videos.</li>
<li><strong>Classes</strong>: Cars, Trucks, Bikes, Bicycles, Accident/Non-Accident.</li>
</ul>

<h3>HEVI Dataset</h3>
<ul>
<li><strong>Description</strong>: High-resolution videos focusing on traffic scenarios (250 videos).</li>
<li><strong>Annotations</strong>: Includes labels for accident and non-accident cases.</li>
</ul>

<h3>Dataset Split</h3>
<ul>
<li><strong>Training Set</strong>: 100 accident and 100 non-accident videos.</li>
<li><strong>Test Set</strong>: 50 accident and 50 non-accident videos.</li>
</ul>

<h2 id="evaluation-metrics">Evaluation Metrics</h2>
<p>The following metrics are used to evaluate the system:</p>
<ul>
<li><strong>Accuracy</strong></li>
<li><strong>F1 Score</strong></li>
<li><strong>Precision and Recall</strong></li>
<li><strong>Mean Absolute Error (MAE)</strong></li>
</ul>

<h2 id="results">Results</h2>
<h3>Baseline vs Finetuned Model</h3>
<table>
<thead>
<tr>
 <th>Metric</th>
 <th>Baseline Model</th>
 <th>Finetuned Model</th>
</tr>
</thead>
<tbody>
<tr>
 <td><strong>Accuracy</strong></td>
 <td>0.65</td>
 <td>0.80</td>
</tr>
<tr>
 <td><strong>MAE (Car)</strong></td>
 <td>4.30</td>
 <td>2.68</td>
</tr>
<tr>
 <td><strong>MAE (Truck)</strong></td>
 <td>0.90</td>
 <td>0.72</td>
</tr>
<tr>
 <td><strong>MAE (Bike)</strong></td>
 <td>0.50</td>
 <td>0.70</td>
</tr>
<tr>
 <td><strong>MAE (Bicycle)</strong></td>
 <td>0.65</td>
 <td>0.55</td>
</tr>
</tbody>
</table>

<h2 id="model-deployment">Model Deployment</h2>
<p>The finetuned model is deployed as a chatbot for interactive querying. Users can query:</p>
<ul>
<li>Vehicle counts.</li>
<li>Traffic accidents.</li>
<li>Time intervals with specific traffic patterns.</li>
</ul>

<h2 id="future-work">Future Work</h2>
<ul>
<li>Deploy the model for real-time analysis.</li>
<li>Expand the dataset to improve generalization across diverse scenarios.</li>
<li>Optimize the pipeline for lower latency and scalability.</li>
</ul>

<h2 id="contact">Contact</h2>
<p>For more information or questions regarding this project, feel free to reach out:</p>
<ul>
<li>Shashank Sharma: <a href="mailto:shashank23088@iiitd.ac.in">shashank23088@iiitd.ac.in</a></li>
<li>Nilanjana Chatterjee: <a href="mailto:nilanjanac@iiitd.ac.in">nilanjanac@iiitd.ac.in</a></li>
<li>Argharupa Adhikary: <a href="mailto:argharupa23020@iiitd.ac.in">argharupa23020@iiitd.ac.in</a></li>
<li>Arunoday Ghorai: <a href="mailto:arunoday23023@iiitd.ac.in">arunoday23023@iiitd.ac.in</a></li>
<li>Shubham Kale: <a href="mailto:shubham23094@iiitd.ac.in">shubham23094@iiitd.ac.in</a></li>
<li>Neeraj: <a href="mailto:neeraji@iiitd.ac.in">neeraji@iiitd.ac.in</a></li>
</ul>

<h2 id="references">References</h2>
<ol>
<li><strong>Jangam et al.</strong>, "Leveraging LLMs for Video Querying," DOI.</li>
<li><strong>Tami et al.</strong>, "Using Multimodal LLMs for Automated Detection of Traffic Safety-Critical Events," DOI.</li>
<li><strong>Kong et al.</strong>, "WTS: A Pedestrian-Centric Traffic Video Dataset," DOI.</li>
<li><strong>Patel et al.</strong>, "Multimodal Video Analysis Using LLMs for Traffic Safety and Efficiency," DOI.</li>
</ol>

<h2>Acknowledgements</h2>
<p>This project was developed as part of the coursework at IIIT Delhi under the guidance of <strong>Prof. Rajiv Ratn Shah</strong>.</p>
</body>
</html>
