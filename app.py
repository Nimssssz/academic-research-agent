from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from agent import research  # import your function from agent.py

app = FastAPI(
    title="Academic Research Agent API",
    description="Backend API for the Academic Research Assistant",
    version="1.0.0",
)

class ResearchRequest(BaseModel):
    query: str
    max_per_db: int = 10
    visualize: bool = False   # keep False for API speed
    cluster: bool = True

class ResearchResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    query: Optional[str] = None
    report_markdown: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    papers: Optional[List[Dict[str, Any]]] = None

@app.post("/research", response_model=ResearchResponse)
def run_research(body: ResearchRequest):
    # Call your existing research() function
    report, papers, analysis = research(
        body.query,
        max_per_db=body.max_per_db,
        visualize=body.visualize,
        cluster=body.cluster,
    )

    # If no papers
    if not papers:
        return ResearchResponse(
            success=False,
            message="No papers found for this query. Try a different topic.",
        )

    # Convert ResearchPaper dataclasses to plain dicts
    papers_dict = []
    for p in papers:
        papers_dict.append({
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "doi": p.doi,
            "publication_date": p.publication_date,
            "journal": p.journal,
            "url": p.url,
            "citations": p.citations,
            "keywords": p.keywords,
            "source": p.source,
            "pdf_url": p.pdf_url,
        })

    return ResearchResponse(
        success=True,
        query=body.query,
        report_markdown=report,
        analysis=analysis,
        papers=papers_dict,
    )

@app.get("/")
def root():
    return {
        "message": "Academic Research Agent API is running.",
        "endpoints": ["/research", "/docs"],
    }
