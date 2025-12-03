                                   ╔══════════════════════════╗
                                   ║      ACADEMICAI 🧠       ║
                                   ╚══════════════════════════╝
                                     Automated Research Agent
        
**AcademicAI** is an automated research engine that retrieves, analyzes and summarizes academic papers using open-access scholarly databases.



✨ **Features**
	•	🔍 Multi-database academic search
	•	📑 Extracts titles, authors, abstracts, citations & publication info
	•	📎 Provides open-access PDF links when available
	•	📊 Generates structured analysis and a formatted research report
	•	🤖 Optional semantic clustering (if supported by environment)



🏗️ **Structure**

**File	Purpose**
agent.py	Core engine for search, filtering, processing & report generation
app.py	FastAPI server providing the /query endpoint
requirements.txt	Dependencies required for backend runtime




🧵 **Connected Databases**

Source	Focus Area
OpenAlex	General academic metadata + citations
PubMed	Medical & life sciences
arXiv	CS, AI, physics, engineering, mathematics
CORE	Open-access research archives




**🚀 Deployment**

This backend is deployed on Hugging Face Spaces and functions as the main API.

The frontend is built using Lovable (not included in this repository).





Made with 🧠 + ☕ by Nimish Warghat


