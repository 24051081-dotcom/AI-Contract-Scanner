from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import httpx

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContractRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_contract(req: ContractRequest):
    prompt = f"""You are a legal risk analyst. Analyze the following contract text and identify risky clauses.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation, just JSON):
{{
  "score": <number 0-100, higher = more risky>,
  "verdict": "<one of: SAFE TO SIGN | REVIEW BEFORE SIGNING | DO NOT SIGN WITHOUT LAWYER>",
  "high": [{{"title": "...", "desc": "..."}}],
  "medium": [{{"title": "...", "desc": "..."}}],
  "low": [{{"title": "...", "desc": "..."}}],
  "suggestion": "..."
}}

Rules:
- high = clauses that significantly harm the user
- medium = clauses worth negotiating
- low = standard clauses that are normal but worth knowing
- suggestion = one actionable recommendation
- If a category has no items, return an empty array
- Keep descriptions concise (1-2 sentences max)

Contract text:
{req.text[:3000]}"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        data = response.json()

    raw = data["choices"][0]["message"]["content"]
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result
class NegotiateRequest(BaseModel):
    title: str
    desc: str

@app.post("/negotiate")
async def negotiate_clause(req: NegotiateRequest):
    prompt = f"""You are a legal expert. A contract clause has been flagged as risky:

Clause: {req.title}
Issue: {req.desc}

Suggest a fairer, rewritten version of this clause that protects the employee/user better.
Keep it concise (2-3 sentences), professional, and realistic.
Return ONLY the suggested clause text, nothing else."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        data = response.json()
    return {"suggestion": data["choices"][0]["message"]["content"]}

app.mount("/", StaticFiles(directory=".", html=True), name="static")