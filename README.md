My portfolio website describing my interests, work experience, and projects.

## AI Twin chatbot backend

The chatbot UI on the page can be connected to a FastAPI + OpenAI RAG backend.

- Backend code: `backend/`
- Setup and run instructions: `backend/README.md`
- API endpoints:
  - `POST /api/reindex` builds recursive chunks + embeddings from resume/research docs.
  - `POST /api/chat` answers in a friendly "Ada, Jayanth's AI assistant" voice.
