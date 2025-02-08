# Multimodal-Expert-Tutor

Multimodal Expert Tutor
An AI assistant which leverages expertise from other sources for you.

Features:

Multimodal
Uses tools
Streams responses
Reads out the responses after streaming
Coverts voice to text during input
Scope for Improvement

Read response faster (as streaming starts)
code optimization
UI enhancements
Make it more real time
# imports

import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import gradio as gr
import google.generativeai
import anthropic
# constants

MODEL_GPT = 'gpt-4o-mini'
MODEL_CLAUDE = 'claude-3-5-sonnet-20240620'
MODEL_GEMINI = 'gemini-1.5-flash'
# set up environment

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')
# Connect to OpenAI, Anthropic and Google

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()
import tempfile
import subprocess
from io import BytesIO
from pydub import AudioSegment
import time

def play_audio(audio_segment):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        audio_segment.export(temp_path, format="wav")
        subprocess.call([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
 
def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",  # Also, try replacing onyx with alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play_audio(audio)

talker("Well hi there")
# prompts
general_prompt = "Please be as technical as possible with your answers.\
Only answer questions about topics you have expertise in.\
If you do not know something say so."

additional_prompt_gpt = "Analyze the user query and determine if the content is primarily related to \
coding, software engineering, data science and LLMs. \
If so please answer it yourself else if it is primarily related to \
physics, chemistry or biology get answers from tool ask_gemini or \
if it belongs to subject related to finance, business or economics get answers from tool ask_claude."

system_prompt_gpt = "You are a helpful technical tutor who is an expert in \
coding, software engineering, data science and LLMs."+ additional_prompt_gpt + general_prompt
system_prompt_gemini = "You are a helpful technical tutor who is an expert in physics, chemistry and biology." + general_prompt
system_prompt_claude = "You are a helpful technical tutor who is an expert in finance, business and economics." + general_prompt

def get_user_prompt(question):
    return "Please give a detailed explanation to the following question: " + question
def call_claude(question):
    result = claude.messages.create(
        model=MODEL_CLAUDE,
        max_tokens=200,
        temperature=0.7,
        system=system_prompt_claude,
        messages=[
            {"role": "user", "content": get_user_prompt(question)},
        ],
    )
    
    return result.content[0].text
def call_gemini(question):
    gemini = google.generativeai.GenerativeModel(
        model_name=MODEL_GEMINI,
        system_instruction=system_prompt_gemini
    )
    response = gemini.generate_content(get_user_prompt(question))
    response = response.text
    return response
# tools and functions

def ask_claude(question):
    print(f"Tool ask_claude called for {question}")
    return call_claude(question)
def ask_gemini(question):
    print(f"Tool ask_gemini called for {question}")
    return call_gemini(question)
ask_claude_function = {
    "name": "ask_claude",
    "description": "Get the answer to the question related to a topic this agent is faimiliar with. Call this whenever you need to answer something related to finance, marketing, sales or business in general.For example 'What is gross margin' or 'Explain stock market'",
    "parameters": {
        "type": "object",
        "properties": {
            "question_for_topic": {
                "type": "string",
                "description": "The question which is related to finance, business or economics.",
            },
        },
        "required": ["question_for_topic"],
        "additionalProperties": False
    }
}

ask_gemini_function = {
    "name": "ask_gemini",
    "description": "Get the answer to the question related to a topic this agent is faimiliar with. Call this whenever you need to answer something related to physics, chemistry or biology.Few examples: 'What is gravity','How do rockets work?', 'What is ATP'",
    "parameters": {
        "type": "object",
        "properties": {
            "question_for_topic": {
                "type": "string",
                "description": "The question which is related to physics, chemistry or biology",
            },
        },
        "required": ["question_for_topic"],
        "additionalProperties": False
    }
}
tools = [{"type": "function", "function": ask_claude_function},
        {"type": "function", "function": ask_gemini_function}]
tools_functions_map = {
    "ask_claude":ask_claude,
    "ask_gemini":ask_gemini
}
def chat(history):
    messages = [{"role": "system", "content": system_prompt_gpt}] + history
    stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, tools=tools, stream=True)
    
    full_response = ""
    history += [{"role":"assistant", "content":full_response}]
    
    tool_call_accumulator = ""  # Accumulator for JSON fragments of tool call arguments
    tool_call_id = None  # Current tool call ID
    tool_call_function_name = None # Function name
    tool_calls = []  # List to store complete tool calls

    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content or ""
            history[-1]['content']=full_response
            yield history
        
        if chunk.choices[0].delta.tool_calls:
            message = chunk.choices[0].delta
            for tc in chunk.choices[0].delta.tool_calls:
                if tc.id:  # New tool call detected here
                    tool_call_id = tc.id
                    if tool_call_function_name is None:
                        tool_call_function_name = tc.function.name
                
                tool_call_accumulator += tc.function.arguments if tc.function.arguments else ""
                
                # When the accumulated JSON string seems complete then:
                try:
                    func_args = json.loads(tool_call_accumulator)
                    
                    # Handle tool call and get response
                    tool_response, tool_call = handle_tool_call(tool_call_function_name, func_args, tool_call_id)
                    
                    tool_calls.append(tool_call)

                    # Add tool call and tool response to messages this is required by openAI api
                    messages.append({
                        "role": "assistant",
                        "tool_calls": tool_calls
                    })
                    messages.append(tool_response)
                    
                    # Create new response with full context
                    response = openai.chat.completions.create(
                        model=MODEL_GPT, 
                        messages=messages, 
                        stream=True
                    )
                    
                    # Reset and accumulate new full response
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content or ""
                            history[-1]['content'] = full_response
                            yield history
                    
                    # Reset tool call accumulator and related variables
                    tool_call_accumulator = ""
                    tool_call_id = None
                    tool_call_function_name = None
                    tool_calls = []

                except json.JSONDecodeError:
                    # Incomplete JSON; continue accumulating
                    pass

    # trigger text-to-audio once full response available
    talker(full_response)
# We have to write that function handle_tool_call:
def handle_tool_call(function_name, arguments, tool_call_id):
    question = arguments.get('question_for_topic')
 
    # Prepare tool call information
    tool_call = {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments)
        }
    }
    
    if function_name in tools_functions_map:
        answer = tools_functions_map[function_name](question)
        response = {
            "role": "tool",
            "content": json.dumps({"question": question, "answer" : answer}),
            "tool_call_id": tool_call_id
        }

        return response, tool_call
def transcribe_audio(audio_file_path):
    try:
        audio_file = open(audio_file_path, "rb")
        response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)        
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages", label="Multimodal Technical Expert Chatbot")
    with gr.Row():
        entry = gr.Textbox(label="Ask our technical expert anything:")
        audio_input = gr.Audio(
            sources="microphone", 
            type="filepath",
            label="Record audio",
            editable=False,
            waveform_options=gr.WaveformOptions(
                show_recording_waveform=False,
            ),
        )

        # Add event listener for audio stop recording and show text on input area
        audio_input.stop_recording(
            fn=transcribe_audio, 
            inputs=audio_input, 
            outputs=entry
        )
            
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        yield "", history
        
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry,chatbot]).then(
        chat, inputs=chatbot, outputs=chatbot)
    
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)
 
