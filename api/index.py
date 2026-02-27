from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List
import json
import os
import re
import sys
import traceback
from io import StringIO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Q2: Sentiment Analysis ----
class Comment(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.post("/comment")
async def analyze_comment(body: Comment):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = f"""You are a sentiment analysis expert. Analyze the sentiment of the following comment and classify it.

Rules:
- sentiment must be EXACTLY one of: 'positive', 'negative', or 'neutral'
- 'positive': comment expresses satisfaction, happiness, praise, or excitement
- 'negative': comment expresses dissatisfaction, complaint, sadness, or anger
- 'neutral': comment is factual, neither positive nor negative
- rating: integer 1-5 (5=highly positive, 3=neutral, 1=highly negative)

Comment: {body.comment}

Respond with JSON only."""
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SentimentResponse,
                temperature=0.1,
            )
        )
        result = json.loads(response.text)
        return result
    except Exception as e:
        return {"sentiment": "neutral", "rating": 3, "error": str(e)}

# ---- Q3: Code Interpreter ----
class CodeRequest(BaseModel):
    code: str

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

def analyze_error_with_ai(code: str, tb: str) -> list:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = f"""Analyze this Python code and its error traceback. Return the line number(s) where the error occurred as a JSON object with key 'error_lines' containing an array of integers.

CODE:
{code}

TRACEBACK:
{tb}"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"error_lines": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER))},
                    required=["error_lines"]
                )
            )
        )
        data = json.loads(response.text)
        return data.get("error_lines", [])
    except Exception:
        lines = re.findall(r'line (\d+)', tb)
        return [int(l) for l in lines[-1:]] if lines else []

@app.post("/code-interpreter")
async def code_interpreter(body: CodeRequest):
    result = execute_python_code(body.code)
    if result["success"]:
        return {"error": [], "result": result["output"]}
    else:
        error_lines = analyze_error_with_ai(body.code, result["output"])
        return {"error": error_lines, "result": result["output"]}

# ---- Q7: YouTube Timestamp Finder ----
class AskRequest(BaseModel):
    video_url: str
    topic: str

@app.post("/ask")
async def find_timestamp(body: AskRequest):
    try:
        from google import genai
        from google.genai import types
        import subprocess
        import tempfile
        import time
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", audio_path, body.video_url
            ], check=True, capture_output=True)
            uploaded = client.files.upload(path=audio_path, config={"mime_type": "audio/mpeg"})
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            prompt = f"""Listen to this audio and find the timestamp when the topic '{body.topic}' is first spoken or discussed. Return a JSON object with key 'timestamp' in HH:MM:SS format."""
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"timestamp": types.Schema(type=types.Type.STRING)},
                        required=["timestamp"]
                    )
                )
            )
            data = json.loads(response.text)
            ts = data.get("timestamp", "00:00:00")
            if re.match(r'^\d{2}:\d{2}$', ts):
                ts = "00:" + ts
            elif re.match(r'^\d+$', ts):
                secs = int(ts)
                ts = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
            return {"timestamp": ts, "video_url": body.video_url, "topic": body.topic}
    except Exception as e:
        return {"timestamp": "00:00:00", "video_url": body.video_url, "topic": body.topic, "error": str(e)}

# ---- Q12: Function Calling ----
@app.get("/execute")
async def execute_function(q: str):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://aipipe.org/openai/v1")
        tools = [
            {"type": "function", "function": {"name": "get_ticket_status", "parameters": {"type": "object", "properties": {"ticket_id": {"type": "integer"}}, "required": ["ticket_id"]}}},
            {"type": "function", "function": {"name": "schedule_meeting", "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "meeting_room": {"type": "string"}}, "required": ["date", "time", "meeting_room"]}}},
            {"type": "function", "function": {"name": "get_expense_balance", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}}, "required": ["employee_id"]}}},
            {"type": "function", "function": {"name": "calculate_performance_bonus", "parameters": {"type": "object", "properties": {"employee_id": {"type": "integer"}, "current_year": {"type": "integer"}}, "required": ["employee_id", "current_year"]}}},
            {"type": "function", "function": {"name": "report_office_issue", "parameters": {"type": "object", "properties": {"issue_code": {"type": "integer"}, "department": {"type": "string"}}, "required": ["issue_code", "department"]}}}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            tools=tools,
            tool_choice="required"
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
    except Exception as e:
        return {"error": str(e)}
