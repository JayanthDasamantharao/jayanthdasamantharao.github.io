# AI Twin Backend

## 1) Create environment and install dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Postgres (required for chat persistence)

Run local Postgres via Docker:

```bash
docker run --name jayanth-chat-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=portfolio_chat \
  -p 5432:5432 \
  -d postgres:16
```

## 2) Configure OpenAI key

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.
Set `DATABASE_URL` in `.env`, for example:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/portfolio_chat
```

Protect admin chat logs in `.env`:

```env
CHAT_ADMIN_ENABLED=true
CHAT_ADMIN_USERNAME=your_admin_user
CHAT_ADMIN_PASSWORD=your_strong_password
CHAT_ADMIN_SESSION_SECONDS=43200
```

## 3) Run the app

```bash
cd ..
source backend/.venv/bin/activate
export $(grep -v '^#' backend/.env | xargs)
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## 4) Build vector index

Call:

```bash
curl -X POST http://localhost:8000/api/reindex
```

### Resumes (nested paths + chat download)

- Put PDF/DOCX files under `Resume/` anywhere, including `Resume/<company_name>/resume.pdf`. With `RAG_RESUME_LIMIT=0` (default in `.env.example`), **all** of these files are indexed for RAG.
- Optional: copy `backend/resume_manifest.example.json` to `Resume/resume_manifest.json` and edit paths/tags. If the manifest is missing, every PDF/DOCX under `Resume/` is discovered automatically; tags are inferred from folder and file names.
- When a user asks for a resume/CV file, the model classifies `resume_request`, picks the best file (manifest tags + conversation), and returns a **Download resume** link in chat. The browser saves it as `resume_jayanthd.pdf` (or `.docx` if the source is Word) — configurable via `RESUME_DOWNLOAD_FILENAME`.
- Short-lived download URLs: `GET /api/resume/download/<token>` (see `RESUME_TOKEN_TTL_SECONDS`).

## 5) Chat endpoint

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What are you currently working on?","history":[]}'
```

`/api/chat` also accepts:

```json
{
  "session_id": "optional-client-generated-id"
}
```

## 5.1) Retrieve saved chats

```bash
curl -u your_admin_user:your_strong_password http://localhost:8000/api/chats/sessions
curl -u your_admin_user:your_strong_password http://localhost:8000/api/chats/<session_id>
```

Login page:

```bash
http://localhost:8000/admin/login
```

Admin table UI (after login):

```bash
http://localhost:8000/admin/chats
```

Inside `/admin/chats`, use the "Postgres Connection Tester" panel to validate host/port/database/user/password and view discovered tables. Credentials entered there are used only for that test request and are not stored.

## 6) Meeting scheduling flow (optional)

If someone asks to connect/schedule a call in chat, backend now does:

1. Collect name, profession, email, and preferred time.
2. Validate the requester’s email (syntax + optional DNS/MX deliverability). With `EMAIL_VERIFICATION_ENABLED` (default), send a one-time code to that inbox before creating a slot request—so deliverability is proven by receipt. Set `EMAIL_CHECK_DELIVERABILITY=false` in `.env` only if DNS checks fail in your environment (e.g. strict firewalls).
3. After verification (or if verification is off), send acknowledgement email to requester.
4. Email Jayanth a link to propose an exact slot.
5. Send proposal email to requester with accept/reject links.
6. On requester acceptance, automatically send final confirmation email with Google Meet link.

Set these in `backend/.env`:

- `SMTP_HOST`, `SMTP_PORT`
- `SMTP_USERNAME`, `SMTP_PASSWORD` (for Gmail use app password)
- `SMTP_FROM_EMAIL`
- `SMTP_USE_TLS`
- `JAYANTH_NOTIFY_EMAIL`
- `API_PUBLIC_URL` — public **FastAPI** base URL (e.g. `https://your-service.onrender.com`). Used for `/api/meeting/...` links in emails. **Required** if `PUBLIC_BASE_URL` is GitHub Pages (Pages cannot serve `/api/`).
- `PUBLIC_BASE_URL` — fallback for those links if `API_PUBLIC_URL` is unset (local dev: `http://localhost:8000`)
- `DEFAULT_GMEET_LINK` (optional static meet link to include in final confirmation)
- `EMAIL_CHECK_DELIVERABILITY` (default `true` — verify domain can receive mail before accepting an address)

Owner slot proposal URL format:

```bash
http://localhost:8000/api/meeting/propose/<token>
```
