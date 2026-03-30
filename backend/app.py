from __future__ import annotations

import html
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from email_validator import EmailNotValidError, validate_email
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat_store import ChatStore
from .email_verification import (
    ConnectEmailVerificationStore,
    extract_six_digit_code,
    looks_like_resend_request,
)
from .meeting_coordinator import MeetingCoordinator, _api_public_base, ensure_est_in_slot_text
from .rag_engine import ResumeRagEngine


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    session_id: Optional[str] = None


class PostgresTestRequest(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str
    username: str
    password: str
    sslmode: str = "disable"


class AdminLoginRequest(BaseModel):
    username: str
    password: str


app = FastAPI(title="Jayanth AI Twin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
            os.environ[key] = value

_load_env_file(REPO_ROOT / "backend" / ".env")
engine = ResumeRagEngine(repo_root=REPO_ROOT, data_dir=REPO_ROOT / "backend" / "data")
meeting_coordinator = MeetingCoordinator(data_dir=REPO_ROOT / "backend" / "data")
connect_verification = ConnectEmailVerificationStore(REPO_ROOT / "backend" / "data")
database_url = os.getenv("DATABASE_URL", "").strip()


def _validate_requester_email(raw: Optional[str]) -> Tuple[bool, str, str]:
    """Returns (ok, user_facing_error_message, normalized_email)."""
    if raw is None or not str(raw).strip():
        return (
            False,
            "I'm sorry — I need an email address to send your confirmation and slot options. "
            "Could you share one you check regularly?",
            "",
        )

    check_deliverability = os.getenv("EMAIL_CHECK_DELIVERABILITY", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        info = validate_email(
            str(raw).strip(),
            check_deliverability=check_deliverability,
        )
    except EmailNotValidError:
        return (
            False,
            "I'm sorry — I'm not sure that email will work: it might be mistyped, or the domain doesn't look set up "
            "to receive mail. Could you double-check it and send a work or personal address you use often? "
            "(Company domains like you@yourcompany.com are totally fine when they're correct.)",
            "",
        )

    normalized = info.normalized or str(raw).strip()
    return True, "", normalized


chat_store = ChatStore(database_url=database_url) if database_url else None
chat_admin_enabled = os.getenv("CHAT_ADMIN_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
chat_admin_username = os.getenv("CHAT_ADMIN_USERNAME", "").strip()
chat_admin_password = os.getenv("CHAT_ADMIN_PASSWORD", "").strip()
chat_admin_security = HTTPBasic(auto_error=False)
chat_admin_session_cookie_name = "chat_admin_session"
chat_admin_session_seconds = max(
    300,
    int(os.getenv("CHAT_ADMIN_SESSION_SECONDS", "43200") or "43200"),
)
chat_admin_sessions: Dict[str, float] = {}

# Short-lived tokens for resume downloads (path -> not exposed in URLs beyond token).
resume_download_tokens: Dict[str, Tuple[float, Path]] = {}


def _cleanup_resume_download_tokens() -> None:
    now = time.time()
    expired = [key for key, (expiry, _) in resume_download_tokens.items() if expiry <= now]
    for key in expired:
        resume_download_tokens.pop(key, None)


def _issue_resume_download_token(path: Path) -> str:
    _cleanup_resume_download_tokens()
    token = secrets.token_urlsafe(24)
    ttl = max(60, int(os.getenv("RESUME_TOKEN_TTL_SECONDS", "900") or "900"))
    resume_download_tokens[token] = (time.time() + ttl, path.resolve())
    return token


def _resume_path_for_token(token: str) -> Optional[Path]:
    _cleanup_resume_download_tokens()
    item = resume_download_tokens.get(token)
    if not item:
        return None
    expiry, path = item
    if time.time() > expiry:
        resume_download_tokens.pop(token, None)
        return None
    if not path.is_file():
        resume_download_tokens.pop(token, None)
        return None
    return path


def _email_verification_enabled() -> bool:
    return os.getenv("EMAIL_VERIFICATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _reply_linked_existing_meeting(
    entry: Dict[str, Any],
    details: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, str]]],
    *,
    context: str = "email_verify",
    base_reply: Optional[str] = None,
) -> str:
    """
    Chat reply when the same email already has an active meeting: session re-linked, no duplicate request.
    context: email_verify | connect_direct | resume_no_fit
    """
    jayanth_sent = meeting_coordinator.notify_jayanth_session_relinked(entry, details)
    requester_sent = meeting_coordinator.send_requester_relink_ack(entry, details)

    status = (entry.get("status") or "").strip()
    proposed = (entry.get("proposed_time") or "").strip()
    preferred = (entry.get("preferred_time") or "").strip()
    # Single source of truth for "what's on file": proposed slot beats original preference.
    canonical_meeting_time = ensure_est_in_slot_text(
        (proposed or preferred or "").strip() or "the time on file for this meeting"
    )
    vd_pref = (details.get("preferred_time") or "").strip()
    verification_time = ensure_est_in_slot_text(vd_pref) if vd_pref else ""
    times_may_differ = bool(
        verification_time
        and verification_time.lower() not in canonical_meeting_time.lower()
        and canonical_meeting_time.lower() not in verification_time.lower()
    )

    meet_raw = (entry.get("meeting_link") or "").strip()
    placeholder = meeting_coordinator.is_placeholder_meet_link(meet_raw)
    meet_link = "" if placeholder else meet_raw

    raw_name = (details.get("name") or entry.get("name") or "").strip()
    visitor_name = raw_name if raw_name and raw_name.lower() not in {"unknown", "there"} else ""

    greet = f"Welcome back, {visitor_name}!" if visitor_name else "Welcome back!"
    email_note = ""
    if jayanth_sent:
        email_note += " Jayanth has been emailed a quick heads-up."
    if requester_sent:
        email_note += " You should also see a short confirmation in your inbox."
    fb = (
        f"{greet} Your email matches an **existing meeting** on file — I’ve linked this chat. "
        f"**Scheduled time on file:** **{canonical_meeting_time}** (status: **{status}**).{email_note} "
        "You can **cancel**, **reschedule**, or leave a note for Jayanth here."
    )
    if times_may_differ and verification_time:
        fb += f" (You mentioned **{verification_time}** in this chat — if that doesn’t match what’s on file, say **reschedule**.)"
    if status == "confirmed" and not placeholder and meet_link:
        fb += f" Meet link (same as in your confirmation): {meet_link}"
    elif status == "confirmed":
        fb += " Use the Google Meet link from your confirmation email (we don’t repeat generic “new meeting” links here)."

    prior_turns = len(history or [])
    facts: Dict[str, Any] = {
        "returning_visitor_with_existing_meeting": True,
        "visitor_name": visitor_name or None,
        "meeting_status": status,
        "canonical_meeting_time": canonical_meeting_time,
        "original_preference_on_record": preferred or None,
        "proposed_agreed_time": proposed or None,
        "verification_preferred_time_from_this_chat": verification_time or None,
        "times_may_differ": times_may_differ,
        "meeting_link": meet_link or None,
        "meet_link_is_generic_placeholder": placeholder,
        "chat_relinked_to_session": True,
        "link_context": context,
        "jayanth_notified_by_email": jayanth_sent,
        "requester_confirmation_email_sent": requester_sent,
        "prior_messages_in_thread_count": prior_turns,
    }
    if base_reply:
        facts["prior_assistant_context"] = base_reply

    instr = (
        "They returned with the same email as an EXISTING active meeting (new browser/tab or days later). "
        "Welcome them back warmly. "
        "STRICT: The authoritative scheduled / on-file time is **canonical_meeting_time** only (proposed time wins over original preference). "
        "If times_may_differ is true, explain briefly: they mentioned a different time in this chat, but the booking on file is canonical_meeting_time — "
        "they can ask to reschedule if needed. Never mix up two different times as if both were confirmed. "
        "If meet_link_is_generic_placeholder is true, NEVER paste meet.google.com/new or similar; say the real Meet link is in their confirmation email. "
        "If jayanth_notified_by_email is true, say clearly that Jayanth was just emailed a heads-up about the reconnect. "
        "If requester_confirmation_email_sent is true, mention they should see a short confirmation email too. "
        "Reassure them their earlier messages in this chat are still in the thread (prior_messages_in_thread_count helps). "
        "Invite cancel, reschedule, or note. 2 short paragraphs, 1–2 emojis."
    )
    if base_reply:
        instr = (
            "There was prior assistant context about resume/fit — acknowledge briefly without repeating it verbatim. " + instr
        )

    return engine.compose_flow_reply(
        instruction=instr,
        message=message,
        history=history,
        facts=facts,
        fallback_reply=fb,
    )


def _complete_connect_after_verification(
    *,
    details: Dict[str, Any],
    source_message: str,
    message: str,
    history: Optional[List[Dict[str, str]]],
    chat_session_id: str,
) -> str:
    details_dict = {
        "name": details.get("name") or "Unknown",
        "profession": details.get("profession") or "Unknown",
        "email": details.get("email") or "",
        "preferred_time": details.get("preferred_time") or "Not provided",
    }
    outcome = meeting_coordinator.create_request_or_link_existing_session(
        details_dict,
        source_message,
        chat_session_id,
    )
    if outcome.get("kind") == "linked":
        return _reply_linked_existing_meeting(
            outcome["entry"],
            details,
            message,
            history,
            context="email_verify",
        )

    request = outcome["request"]
    requester_ack_sent = meeting_coordinator.send_requester_ack(request)
    sent = meeting_coordinator.notify_jayanth_for_slot_selection(request)
    raw_name = (details.get("name") or "").strip()
    visitor_name = raw_name if raw_name and raw_name.lower() not in {"unknown", "there"} else ""
    preferred_time_display = ensure_est_in_slot_text(
        (details.get("preferred_time") or "").strip() or "the time you shared"
    )
    name_bit = f"{visitor_name}, " if visitor_name else ""
    fb_ok = (
        f"Yes — {name_bit}you're verified, and I'm genuinely thrilled for you! 🎉✨ "
        f"Jayanth's been looped in on your meeting for **{preferred_time_display}**, and your confirmation email is headed to your inbox. "
        "Want me to pass any extra note to him?"
    )
    fb_partial = (
        f"Yes — {name_bit}you're verified — I'm so excited we got this locked in! 🎉 "
        f"I've flagged Jayanth about your **{preferred_time_display}** request and he's on it. "
        "Want me to pass any extra note to him?"
    )
    fb = fb_ok if (sent and requester_ack_sent) else fb_partial
    return engine.compose_flow_reply(
        instruction=(
            "Email verification succeeded. Use a VERY excited, happy, celebratory tone—like you're genuinely thrilled for them. "
            "Address them by visitor_name when appropriate. "
            "State clearly that Jayanth has been notified about their meeting request for preferred_time_display (use that exact string; it is US Eastern when applicable). "
            "Use only the facts for whether a confirmation email was sent and whether Jayanth was notified. "
            "Ask if they want you to pass an extra note to Jayanth. "
            "Use 2–3 short paragraphs; 2–3 emojis max. Sound human and warm, not corporate."
        ),
        message=message,
        history=history,
        facts={
            "visitor_name": visitor_name or None,
            "preferred_time_display": preferred_time_display,
            "confirmation_email_sent_to_requester": bool(requester_ack_sent),
            "jayanth_notified": bool(sent),
        },
        fallback_reply=fb,
    )


def _handle_meeting_change(
    session_id: str,
    payload: ChatRequest,
    intent_confidence: Optional[float],
    chat_store: Optional[ChatStore],
) -> Dict[str, Any]:
    """Cancel or reschedule an active meeting; email requester and Jayanth."""
    details = engine.extract_connect_details(payload.message, payload.history)
    email_hint = (details.get("email") or "").strip()
    entry = meeting_coordinator.find_active_request_for_chat(session_id, email_hint or None)

    if not entry:
        fb = (
            "I couldn’t match an active meeting to this chat just yet. "
            "If you booked from another browser, check your email from Ada — "
            "or share the email you used to connect and ask again."
        )
        reply = engine.compose_flow_reply(
            instruction=(
                "No meeting record linked to this chat. Be gentle; suggest checking email or sharing the address they used."
            ),
            message=payload.message,
            history=payload.history,
            facts={"meeting_found": False},
            fallback_reply=fb,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    parsed = engine.extract_meeting_change_action(payload.message, payload.history)
    action = str(parsed.get("action") or "unclear").lower()
    note = parsed.get("note")
    new_availability = parsed.get("new_availability")

    if action == "unclear":
        fb = (
            "I can handle that — do you want to **cancel** the meeting entirely, or **reschedule** for another time? "
            "You can add a short note for Jayanth if you’d like."
        )
        reply = engine.compose_flow_reply(
            instruction="Ask warmly whether they want to cancel or reschedule; mention optional note for Jayanth.",
            message=payload.message,
            history=payload.history,
            facts={"clarification_needed": True},
            fallback_reply=fb,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    rid = entry.get("request_id")
    if not rid:
        fb = "I couldn’t update that meeting just now — please try again in a moment."
        reply = engine.compose_flow_reply(
            instruction="Brief apology; suggest retry.",
            message=payload.message,
            history=payload.history,
            facts={"error": "missing_request_id"},
            fallback_reply=fb,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    if action == "cancel":
        result = meeting_coordinator.cancel_meeting_request(rid, note=note)
    else:
        result = meeting_coordinator.request_reschedule_meeting(
            rid, new_availability=new_availability, note=note
        )

    st = result.get("status")
    if st == "not_found":
        fb = "I couldn’t find that meeting to update. If this keeps happening, use your latest email from Ada."
        reply = engine.compose_flow_reply(
            instruction="Gentle apology; suggest email thread.",
            message=payload.message,
            history=payload.history,
            facts={"update_status": "not_found"},
            fallback_reply=fb,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    if st == "already_cancelled":
        fb = "That meeting was already cancelled — you’re all set. Let me know if you need anything else."
        reply = engine.compose_flow_reply(
            instruction="Acknowledge already cancelled; warm and brief.",
            message=payload.message,
            history=payload.history,
            facts={"already_cancelled": True},
            fallback_reply=fb,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    if st == "invalid_state":
        reason = str(result.get("reason") or "")
        reply = engine.compose_flow_reply(
            instruction=(
                "Explain gently that this change couldn’t be applied (use reason). "
                "Suggest they email or start a new connect flow if needed."
            ),
            message=payload.message,
            history=payload.history,
            facts={"invalid_state_reason": reason},
            fallback_reply=(
                "I wasn’t able to apply that change to the meeting just now. "
                "If you still need help, reply here with a bit more detail or use your email thread with Ada."
            ),
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="meeting_change",
                intent_confidence=intent_confidence,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}

    emails = result.get("emails") or {}
    er = bool(emails.get("requester"))
    ej = bool(emails.get("host"))
    pref = (new_availability or "").strip()
    note_txt = (note or "").strip()

    if action == "cancel":
        facts: Dict[str, Any] = {
            "action": "cancel",
            "requester_name": entry.get("name"),
            "email_to_requester_sent": er,
            "email_to_jayanth_sent": ej,
        }
        fb_ok = (
            "All set — I’ve cancelled that meeting and sent a kind, clear note to **you and Jayanth** by email "
            "so everyone’s on the same page."
            if (er or ej)
            else "I’ve recorded the cancellation, but email didn’t go out just yet (SMTP). "
            "Please reach out through your usual thread if you need to confirm."
        )
        reply = engine.compose_flow_reply(
            instruction=(
                "The meeting was cancelled. Tone: gentle, warm, polite, human. "
                "If email_to_requester_sent or email_to_jayanth_sent, say both sides were notified by email. "
                "If both false, mention email may not have sent and they can follow up."
            ),
            message=payload.message,
            history=payload.history,
            facts=facts,
            fallback_reply=fb_ok,
        )
    else:
        # Reschedule: two-way flow — only a request/preference; Jayanth still proposes and requester confirms by email (same loop as first scheduling).
        facts = {
            "action": "reschedule_request",
            "requester_name": entry.get("name"),
            "requester_availability_or_preference": pref or None,
            "optional_note": note_txt or None,
            "email_to_requester_sent": er,
            "email_to_jayanth_sent": ej,
            "jayanth_must_still_propose": True,
            "new_time_not_final_until_email_confirmation": True,
        }
        fb_ok = (
            "I’ve **notified Jayanth by email** about your reschedule request"
            + (f" — you mentioned **{pref}**" if pref else "")
            + ". He’ll review it and send a **proposed time by email** when he can — same flow as when you first scheduled. "
            "You’ll confirm from that email; nothing is locked in from this chat alone."
            if (er or ej)
            else "I’ve saved your reschedule request, but email didn’t go out just yet (SMTP). "
            "Please try again shortly or reach Jayanth directly so he can propose a new time."
        )
        reply = engine.compose_flow_reply(
            instruction=(
                "STRICT — reschedule request (two-way process, same as initial scheduling): "
                "The user’s preferred time is ONLY a request. Jayanth has been emailed to review and will send a formal **proposal** by email. "
                "The meeting is NOT yet rescheduled to that time until they confirm via the email thread (Yes / alternatives), like the first booking. "
                "Say clearly that you’ve notified Jayanth and he will follow up shortly by email to verify/propose — do NOT say the meeting was already "
                "or 'successfully' moved to a specific new time. "
                "Do not invent that the new time is confirmed. "
                "If email_to_jayanth_sent is true, say Jayanth was notified. If email_to_requester_sent is true, mention a short confirmation was emailed to them too. "
                "Tone: professional, warm, concise, 2 short paragraphs max."
            ),
            message=payload.message,
            history=payload.history,
            facts=facts,
            fallback_reply=fb_ok,
        )
    if chat_store:
        chat_store.log_message(
            session_id=session_id,
            role="assistant",
            message_text=reply,
            intent="meeting_change",
            intent_confidence=intent_confidence,
            sources=[],
        )
    return {"reply": reply, "sources": [], "session_id": session_id, "intent": "meeting_change"}


def _handle_pending_verification(
    session_id: str,
    message: str,
    chat_store: Optional[ChatStore],
    history: Optional[List[Dict[str, str]]],
) -> Optional[Dict[str, Any]]:
    if not connect_verification.get_pending(session_id):
        return None

    if chat_store:
        chat_store.log_message(
            session_id=session_id,
            role="user",
            message_text=message,
            intent="email_verification_pending",
            intent_confidence=None,
        )

    if looks_like_resend_request(message):
        ok, err = connect_verification.resend(session_id)
        smtp_fail = bool(err and "SMTP" in (err or ""))
        fb_ok = "I've sent a fresh code to your email. Paste the 6-digit code here when you have it."
        fb_bad = (
            "I couldn't resend the code — email may not be configured. Check SMTP settings or try the connect flow again shortly."
            if smtp_fail
            else "I couldn't resend a code right now. Try submitting your connect details again."
        )
        reply = engine.compose_flow_reply(
            instruction=(
                "They asked to resend the verification code. If resend_succeeded, confirm a new code was sent. "
                "If not, explain briefly (SMTP may be misconfigured) and suggest retrying the connect flow."
            ),
            message=message,
            history=history,
            facts={"resend_succeeded": ok, "smtp_error": err or ""},
            fallback_reply=fb_ok if ok else fb_bad,
        )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="email_verification_pending",
                intent_confidence=None,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "email_verification_pending"}

    code = extract_six_digit_code(message)
    if code:
        status, payload = connect_verification.verify(session_id, code)
        if status == "ok" and payload:
            details = payload.get("details") or {}
            source_message = payload.get("source_message") or message
            reply = _complete_connect_after_verification(
                details=details,
                source_message=source_message,
                message=message,
                history=history,
                chat_session_id=session_id,
            )
            if chat_store:
                chat_store.log_message(
                    session_id=session_id,
                    role="assistant",
                    message_text=reply,
                    intent="connect_request",
                    intent_confidence=None,
                    sources=[],
                )
            return {"reply": reply, "sources": [], "session_id": session_id, "intent": "connect_request"}
        fb_wrong = (
            "That code doesn't match. Double-check the email from Ada and try again — or say **resend code** for a new one."
        )
        fb_expired = "That code has expired. Share your connect details again and I'll send a new code."
        fb_locked = "Too many incorrect attempts. Please start the connect flow again with your details."
        fb_else = "Something went wrong verifying that code. Try **resend code** or start the connect flow again."
        if status == "wrong":
            reply = engine.compose_flow_reply(
                instruction="The verification code they entered was wrong. Be kind; suggest checking email or resend code.",
                message=message,
                history=history,
                facts={"verification_result": "wrong_code"},
                fallback_reply=fb_wrong,
            )
        elif status == "expired":
            reply = engine.compose_flow_reply(
                instruction="The verification code expired. Ask them to share connect details again for a new code.",
                message=message,
                history=history,
                facts={"verification_result": "expired"},
                fallback_reply=fb_expired,
            )
        elif status == "locked":
            reply = engine.compose_flow_reply(
                instruction="Too many wrong attempts. Ask them to restart the connect flow with their details.",
                message=message,
                history=history,
                facts={"verification_result": "locked_out"},
                fallback_reply=fb_locked,
            )
        else:
            reply = engine.compose_flow_reply(
                instruction="Verification failed for an unexpected reason. Suggest resend code or restarting connect.",
                message=message,
                history=history,
                facts={"verification_result": status},
                fallback_reply=fb_else,
            )
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent="email_verification_pending",
                intent_confidence=None,
                sources=[],
            )
        return {"reply": reply, "sources": [], "session_id": session_id, "intent": "email_verification_pending"}

    fb_wait = (
        "I'm still waiting for the 6-digit code from your email. Paste it here to confirm — then I'll notify Jayanth right away. "
        "Need a new code? Say **resend code**."
    )
    reply = engine.compose_flow_reply(
        instruction=(
            "They haven't pasted the verification code yet (or message wasn't a code). "
            "Remind them gently to paste the 6-digit code from email, or say resend code."
        ),
        message=message,
        history=history,
        facts={"waiting_for_email_verification_code": True},
        fallback_reply=fb_wait,
    )
    if chat_store:
        chat_store.log_message(
            session_id=session_id,
            role="assistant",
            message_text=reply,
            intent="email_verification_pending",
            intent_confidence=None,
            sources=[],
        )
    return {"reply": reply, "sources": [], "session_id": session_id, "intent": "email_verification_pending"}


def _validate_and_normalize_connect_email(details: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    """If an email is present, validate syntax + optional MX. Returns (error_reply, details)."""
    email = (details.get("email") or "").strip()
    if not email:
        return None, details
    ok, err, normalized = _validate_requester_email(email)
    if not ok:
        return err, details
    return None, {**details, "email": normalized}


def _format_missing_fields(missing: List[str]) -> str:
    labels = {
        "name": "your name",
        "profession": "what you do (your role/profession)",
        "email": "your email",
        "preferred_time": "a couple of convenient time slots in US Eastern (EST/ET)",
    }
    items = [labels[item] for item in missing if item in labels]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _resolve_session_id(provided: Optional[str]) -> str:
    value = (provided or "").strip()
    return value if value else str(uuid.uuid4())


def _verify_admin_credentials(username: str, password: str) -> bool:
    return (
        bool(chat_admin_username)
        and bool(chat_admin_password)
        and secrets.compare_digest(username, chat_admin_username)
        and secrets.compare_digest(password, chat_admin_password)
    )


def _cleanup_expired_admin_sessions(now_ts: float) -> None:
    expired_tokens = [token for token, expiry in chat_admin_sessions.items() if expiry <= now_ts]
    for token in expired_tokens:
        chat_admin_sessions.pop(token, None)


def _is_valid_admin_session_token(token: str) -> bool:
    now_ts = time.time()
    _cleanup_expired_admin_sessions(now_ts)
    expiry = chat_admin_sessions.get(token)
    if not expiry:
        return False
    if expiry <= now_ts:
        chat_admin_sessions.pop(token, None)
        return False
    return True


def _create_admin_session_token() -> str:
    token = secrets.token_urlsafe(32)
    chat_admin_sessions[token] = time.time() + chat_admin_session_seconds
    return token


def _is_request_admin_authenticated(request: Request) -> bool:
    token = (request.cookies.get(chat_admin_session_cookie_name) or "").strip()
    return bool(token) and _is_valid_admin_session_token(token)


def _require_chat_admin_auth(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(chat_admin_security),
) -> None:
    if not chat_admin_enabled:
        # Hide admin surfaces when not enabled.
        raise HTTPException(status_code=404, detail="Not found")
    if not chat_admin_username or not chat_admin_password:
        raise HTTPException(status_code=503, detail="Admin auth is enabled but credentials are not configured.")
    if _is_request_admin_authenticated(request):
        return
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )

    is_username_valid = secrets.compare_digest(credentials.username, chat_admin_username)
    is_password_valid = secrets.compare_digest(credentials.password, chat_admin_password)
    if not (is_username_valid and is_password_valid):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


def _set_admin_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=chat_admin_session_cookie_name,
        value=token,
        max_age=chat_admin_session_seconds,
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "index_ready": engine.is_ready,
        "chunk_count": len(engine.index_data.get("chunks", [])),
        "chat_persistence": "postgres_enabled" if chat_store else "disabled_no_database_url",
    }


@app.get("/api/resume/download/{token}")
def download_resume(token: str) -> FileResponse:
    path = _resume_path_for_token(token)
    if not path:
        raise HTTPException(status_code=404, detail="Download link expired or invalid.")
    dl = (os.getenv("RESUME_DOWNLOAD_FILENAME") or "resume_jayanthd.pdf").strip() or "resume_jayanthd.pdf"
    if path.suffix.lower() == ".docx":
        if dl.lower().endswith(".pdf"):
            dl = "resume_jayanthd.docx"
    elif path.suffix.lower() == ".pdf":
        if not dl.lower().endswith(".pdf"):
            dl = "resume_jayanthd.pdf"
    media = (
        "application/pdf"
        if path.suffix.lower() == ".pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(path, filename=dl, media_type=media)


@app.post("/api/reindex")
def reindex() -> Dict[str, Any]:
    try:
        stats = engine.build_index()
        engine.invalidate_resume_cache()
        return {"status": "ok", "stats": stats}
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    try:
        session_id = _resolve_session_id(payload.session_id)
        if _email_verification_enabled():
            pending_reply = _handle_pending_verification(
                session_id, payload.message, chat_store, payload.history
            )
            if pending_reply is not None:
                return pending_reply

        routing = engine.classify_intent(payload.message, payload.history)
        intent = routing.get("intent", "information_request")
        # LLM guardrail: if classifier marks out_of_scope but query is still profile-related,
        # route back to information_request so RAG can answer.
        if intent == "out_of_scope":
            profile_check = engine.is_profile_related(payload.message, payload.history)
            if profile_check.get("profile_related"):
                intent = "information_request"
        if intent == "information_request" and engine.should_continue_resume_file_flow(
            payload.message, payload.history
        ):
            intent = "resume_request"
        raw_confidence = routing.get("confidence")
        intent_confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="user",
                message_text=payload.message,
                intent=intent,
                intent_confidence=intent_confidence,
            )

        if intent == "abusive":
            reply = engine.intent_reply(intent, payload.message, payload.history)
            if chat_store:
                chat_store.log_message(
                    session_id=session_id,
                    role="assistant",
                    message_text=reply,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                    is_abusive_blocked=True,
                )
            return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}
        if intent == "greeting":
            reply = engine.intent_reply(intent, payload.message, payload.history)
            if chat_store:
                chat_store.log_message(
                    session_id=session_id,
                    role="assistant",
                    message_text=reply,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                )
            return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}
        if intent == "meeting_change":
            return _handle_meeting_change(session_id, payload, intent_confidence, chat_store)
        if intent == "resume_request":
            entries = engine.get_resume_entries()
            dl_name = (os.getenv("RESUME_DOWNLOAD_FILENAME") or "resume_jayanthd.pdf").strip() or "resume_jayanthd.pdf"
            if not entries:
                fb = (
                    "I couldn't find a resume file on disk yet — once resumes are available under the Resume/ folder, "
                    "I'll be able to share a download here.\n\n"
                    "Would you like Jayanth's LinkedIn or email in the meantime?"
                )
                reply = engine.compose_flow_reply(
                    instruction="No resume files are deployed yet. Offer LinkedIn or email in Ada's voice.",
                    message=payload.message,
                    history=payload.history,
                    facts={"resume_files_on_disk": False},
                    fallback_reply=fb,
                )
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}

            focus_gate = engine.assess_resume_focus_gate(payload.message, payload.history)
            if focus_gate.get("needs_clarification"):
                reply = engine.resume_focus_clarification_reply(payload.message, payload.history)
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}

            picked = engine.select_resume_entry(payload.message, payload.history)
            if not picked:
                fb = (
                    "I couldn't locate a matching resume file just now.\n\n"
                    "Would you like Jayanth's LinkedIn or email in the meantime?"
                )
                reply = engine.compose_flow_reply(
                    instruction="No resume matched their ask. Offer LinkedIn or email helpfully.",
                    message=payload.message,
                    history=payload.history,
                    facts={"resume_match_found": False},
                    fallback_reply=fb,
                )
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}

            selected_path, selected_entry = picked
            if not engine.verify_resume_entry_matches_need(
                payload.message, payload.history, selected_entry
            ):
                reply = engine.resume_no_fit_connect_reply(payload.message, payload.history)
                details = engine.extract_connect_details(payload.message, payload.history)
                email_err, details = _validate_and_normalize_connect_email(details)
                if email_err:
                    fb = (
                        f"{reply}\n\n{email_err}\n\n"
                        "Once you share a reachable address, I can pass your details to Jayanth for a slot."
                    )
                    reply = engine.compose_flow_reply(
                        instruction=(
                            "Combine the resume-fit message in prior_reply with a clear note that their email needs fixing. "
                            "Use validation_issue verbatim for the email problem. Offer to connect once fixed."
                        ),
                        message=payload.message,
                        history=payload.history,
                        facts={"prior_reply": reply, "validation_issue": email_err},
                        fallback_reply=fb,
                    )
                    if chat_store:
                        chat_store.log_message(
                            session_id=session_id,
                            role="assistant",
                            message_text=reply,
                            intent="resume_no_fit_connect",
                            intent_confidence=intent_confidence,
                            sources=[],
                        )
                    return {"reply": reply, "sources": [], "session_id": session_id, "intent": "resume_no_fit_connect"}
                missing = meeting_coordinator.missing_fields(details)
                if not missing:
                    if _email_verification_enabled():
                        ok, err = connect_verification.start(
                            session_id=session_id,
                            details={
                                "name": details["name"] or "Unknown",
                                "profession": details["profession"] or "Unknown",
                                "email": details["email"] or "",
                                "preferred_time": details["preferred_time"] or "Not provided",
                            },
                            source_message=payload.message,
                        )
                        em = (details.get("email") or "").strip()
                        if ok:
                            fb = (
                                f"{reply}\n\nI've sent a 6-digit verification code to **{em}**. "
                                "Paste it here when it arrives — once it checks out, I'll notify Jayanth about fit and your connect request."
                            )
                            reply = engine.compose_flow_reply(
                                instruction=(
                                    "Build on base_resume_reply: a verification code was sent to verification_email. "
                                    "Ask them to paste it in chat; then you'll notify Jayanth about fit and the connect request."
                                ),
                                message=payload.message,
                                history=payload.history,
                                facts={
                                    "base_resume_reply": reply,
                                    "verification_email": em,
                                    "verification_code_sent": True,
                                },
                                fallback_reply=fb,
                            )
                        else:
                            fb = (
                                f"{reply}\n\nI couldn't email a verification code ({err or 'error'}), "
                                "so I didn't submit the connect request yet. Please try again when email is working."
                            )
                            reply = engine.compose_flow_reply(
                                instruction=(
                                    "Build on base_resume_reply: verification email failed; do NOT say a code was sent. "
                                    "Mention error_detail briefly."
                                ),
                                message=payload.message,
                                history=payload.history,
                                facts={
                                    "base_resume_reply": reply,
                                    "verification_code_sent": False,
                                    "error_detail": err or "",
                                },
                                fallback_reply=fb,
                            )
                    else:
                        details_dict = {
                            "name": details["name"] or "Unknown",
                            "profession": details["profession"] or "Unknown",
                            "email": details["email"] or "",
                            "preferred_time": details["preferred_time"] or "Not provided",
                        }
                        outcome = meeting_coordinator.create_request_or_link_existing_session(
                            details_dict,
                            payload.message,
                            session_id,
                        )
                        if outcome.get("kind") == "linked":
                            reply = _reply_linked_existing_meeting(
                                outcome["entry"],
                                details,
                                payload.message,
                                payload.history,
                                context="resume_no_fit",
                                base_reply=reply,
                            )
                        else:
                            request = outcome["request"]
                            requester_ack_sent = meeting_coordinator.send_requester_ack(request)
                            sent = meeting_coordinator.notify_jayanth_for_slot_selection(request)
                            extra_ok = (
                                "\n\nThanks — I also logged a connect request so Jayanth can follow up about fit. ✅ "
                                "He will propose a time option by email."
                            )
                            extra_bad = "\n\nThanks — I captured your details so Jayanth can follow up about fit. ✅"
                            fb = f"{reply}{extra_ok if (sent and requester_ack_sent) else extra_bad}"
                            reply = engine.compose_flow_reply(
                                instruction=(
                                    "Merge base_resume_reply with follow-up: connect request was logged; "
                                    "confirmation email to requester and Jayanth notification per facts."
                                ),
                                message=payload.message,
                                history=payload.history,
                                facts={
                                    "base_resume_reply": reply,
                                    "confirmation_email_to_requester": bool(requester_ack_sent),
                                    "jayanth_notified": bool(sent),
                                },
                                fallback_reply=fb,
                            )
                elif len(missing) < 4:
                    needs = _format_missing_fields(missing)
                    fb = f"{reply}\n\nI still need: {needs}."
                    reply = engine.compose_flow_reply(
                        instruction=(
                            "After base_resume_reply, ask only for the items in missing_fields_readable "
                            "(US Eastern times for availability)."
                        ),
                        message=payload.message,
                        history=payload.history,
                        facts={"base_resume_reply": reply, "missing_fields_readable": needs},
                        fallback_reply=fb,
                    )
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent="resume_no_fit_connect",
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": "resume_no_fit_connect"}

            if selected_path.suffix.lower() == ".docx" and dl_name.lower().endswith(".pdf"):
                dl_name = "resume_jayanthd.docx"
            token = _issue_resume_download_token(selected_path)
            base = _api_public_base()
            attachment = {
                "download_url": f"{base}/api/resume/download/{token}",
                "filename": dl_name,
                "label": "Download resume",
            }
            skip_connect = engine.thread_has_completed_meeting_request(payload.history)
            reply = engine.resume_attachment_reply(
                payload.message,
                payload.history,
                skip_connect_offer=skip_connect,
            )
            if chat_store:
                try:
                    src_rel = str(selected_path.relative_to(REPO_ROOT))
                except ValueError:
                    src_rel = str(selected_path)
                chat_store.log_message(
                    session_id=session_id,
                    role="assistant",
                    message_text=reply,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[src_rel],
                )
            return {
                "reply": reply,
                "sources": [],
                "session_id": session_id,
                "intent": intent,
                "attachment": attachment,
            }
        if intent == "information_request":
            hiring_check = engine.assess_hiring_query(payload.message, payload.history)
            if hiring_check.get("is_hiring") and not hiring_check.get("has_specific_role"):
                reply = engine.intent_reply("hiring_role_clarification", payload.message, payload.history)
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}
        if intent in {"connect_request", "connect_confirmation"}:
            # LLM-driven detail extraction (no keyword/regex routing for this step).
            details = engine.extract_connect_details(payload.message, payload.history)
            email_err, details = _validate_and_normalize_connect_email(details)
            if email_err:
                fb = (
                    f"{email_err}\n\n"
                    "When you have a valid address, I can send the confirmation and loop Jayanth in for a time slot."
                )
                reply = engine.compose_flow_reply(
                    instruction=(
                        "Explain we need a working email to schedule with Jayanth. "
                        "Use validation_issue for the exact problem; do not contradict it."
                    ),
                    message=payload.message,
                    history=payload.history,
                    facts={"validation_issue": email_err},
                    fallback_reply=fb,
                )
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}
            missing = meeting_coordinator.missing_fields(details)
            if missing:
                # First step: confirm scheduling intent in a friendly way.
                if len(missing) == 4 and intent == "connect_request":
                    if engine.thread_has_completed_meeting_request(payload.history):
                        fb = (
                            "You already have a connect request with Jayanth from this chat — he's on it! 🙌 "
                            "Want me to add a note or tweak anything?"
                        )
                        reply = engine.compose_flow_reply(
                            instruction=(
                                "They already submitted a connect/meeting request in this thread. "
                                "Acknowledge warmly; do NOT ask if they want to schedule a new call. "
                                "Offer to add a note or tweak details."
                            ),
                            message=payload.message,
                            history=payload.history,
                            facts={"stage": "connect_already_submitted"},
                            fallback_reply=fb,
                        )
                    else:
                        fb = (
                            "Sure — I can help set up a meeting with Jayanth. "
                            "Would you like me to schedule a quick call? 😊"
                        )
                        reply = engine.compose_flow_reply(
                            instruction=(
                                "They want to connect with Jayanth. Ask warmly if they'd like you to schedule a quick call."
                            ),
                            message=payload.message,
                            history=payload.history,
                            facts={"stage": "connect_scheduling_prompt"},
                            fallback_reply=fb,
                        )
                    if chat_store:
                        chat_store.log_message(
                            session_id=session_id,
                            role="assistant",
                            message_text=reply,
                            intent=intent,
                            intent_confidence=intent_confidence,
                            sources=[],
                        )
                    return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}
                name = (details.get("name") or "").strip()
                needs = _format_missing_fields(missing)
                if name:
                    fb = f"Thanks, {name}. Could you share {needs} so I can connect you with Jayanth? 😊"
                else:
                    fb = f"Could you share {needs} so I can connect you with Jayanth? 😊"
                reply = engine.compose_flow_reply(
                    instruction=(
                        "Ask for missing_fields to connect them with Jayanth; greet with visitor_name if provided. "
                        "Times should be US Eastern (EST/ET)."
                    ),
                    message=payload.message,
                    history=payload.history,
                    facts={"missing_fields_readable": needs, "visitor_name": name},
                    fallback_reply=fb,
                )
                if chat_store:
                    chat_store.log_message(
                        session_id=session_id,
                        role="assistant",
                        message_text=reply,
                        intent=intent,
                        intent_confidence=intent_confidence,
                        sources=[],
                    )
                return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}

            if _email_verification_enabled():
                ok, err = connect_verification.start(
                    session_id=session_id,
                    details={
                        "name": details["name"] or "Unknown",
                        "profession": details["profession"] or "Unknown",
                        "email": details["email"] or "",
                        "preferred_time": details["preferred_time"] or "Not provided",
                    },
                    source_message=payload.message,
                )
                em = (details.get("email") or "").strip()
                if ok:
                    fb_ok = (
                        f"I've sent a 6-digit verification code to **{em}**. "
                        "Paste it here when it arrives — once it checks out, I'll notify Jayanth and send your confirmation email."
                    )
                    reply = engine.compose_flow_reply(
                        instruction=(
                            "Verification code was sent to verification_email. Ask them to paste it in chat; "
                            "then you'll notify Jayanth and send confirmation per your usual flow."
                        ),
                        message=payload.message,
                        history=payload.history,
                        facts={"verification_email": em, "verification_code_sent": True},
                        fallback_reply=fb_ok,
                    )
                else:
                    fb_bad = (
                        f"I couldn't email a verification code ({err or 'error'}). "
                        "Without confirming your inbox, I won't notify Jayanth yet — fix SMTP or try again shortly."
                    )
                    reply = engine.compose_flow_reply(
                        instruction=(
                            "Could not send verification email; do not claim a code went out. "
                            "Reference error_detail; suggest fixing SMTP or retrying."
                        ),
                        message=payload.message,
                        history=payload.history,
                        facts={"verification_code_sent": False, "error_detail": err or ""},
                        fallback_reply=fb_bad,
                    )
            else:
                details_dict = {
                    "name": details["name"] or "Unknown",
                    "profession": details["profession"] or "Unknown",
                    "email": details["email"] or "",
                    "preferred_time": details["preferred_time"] or "Not provided",
                }
                outcome = meeting_coordinator.create_request_or_link_existing_session(
                    details_dict,
                    payload.message,
                    session_id,
                )
                if outcome.get("kind") == "linked":
                    reply = _reply_linked_existing_meeting(
                        outcome["entry"],
                        details,
                        payload.message,
                        payload.history,
                        context="connect_direct",
                    )
                else:
                    request = outcome["request"]
                    requester_ack_sent = meeting_coordinator.send_requester_ack(request)
                    sent = meeting_coordinator.notify_jayanth_for_slot_selection(request)
                    raw_cn = (details.get("name") or "").strip()
                    visitor_name = raw_cn if raw_cn and raw_cn.lower() not in {"unknown", "there"} else ""
                    preferred_time_display = ensure_est_in_slot_text(
                        (details.get("preferred_time") or "").strip() or "the time you shared"
                    )
                    cn_bit = f"{visitor_name}, " if visitor_name else ""
                    fb_connect = (
                        (
                            f"Yes — {cn_bit}this is happening! 🎉✨ Your request is in, Jayanth knows about your "
                            f"**{preferred_time_display}** slot, and your confirmation email is on the way. "
                            "Want me to pass any extra note to him?"
                        )
                        if (sent and requester_ack_sent)
                        else (
                            f"Yes — {cn_bit}I'm so glad we got this logged! 🎉 Jayanth's been notified about your "
                            f"**{preferred_time_display}** request. Want me to pass any extra note to him?"
                        )
                    )
                    reply = engine.compose_flow_reply(
                        instruction=(
                            "Connect request submitted. Use a very excited, happy tone—celebrate with them. "
                            "Use visitor_name and preferred_time_display from facts. "
                            "Confirm using facts only: confirmation email to requester, Jayanth notified. "
                            "Times are US Eastern when implied. Ask if they want an extra note passed to Jayanth."
                        ),
                        message=payload.message,
                        history=payload.history,
                        facts={
                            "visitor_name": visitor_name or None,
                            "preferred_time_display": preferred_time_display,
                            "confirmation_email_to_requester": bool(requester_ack_sent),
                            "jayanth_notified": bool(sent),
                        },
                        fallback_reply=fb_connect,
                    )
            if chat_store:
                chat_store.log_message(
                    session_id=session_id,
                    role="assistant",
                    message_text=reply,
                    intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                )
            return {"reply": reply, "sources": [], "session_id": session_id, "intent": intent}

        # Default profile flow: always attempt RAG for information and possible out_of_scope.
        meeting_done = engine.thread_has_completed_meeting_request(payload.history)
        result = engine.chat(
            payload.message,
            payload.history,
            allow_low_relevance=True,
            skip_meeting_scheduling_prompt=meeting_done,
        )
        reply = result.get("reply", "")
        sources = result.get("sources", [])
        stored_intent = intent
        if sources:
            stored_intent = "information_request"
        elif intent not in {
            "greeting",
            "abusive",
            "connect_request",
            "connect_confirmation",
            "meeting_change",
            "resume_request",
            "resume_no_fit_connect",
        }:
            stored_intent = "out_of_scope"
        if chat_store:
            chat_store.log_message(
                session_id=session_id,
                role="assistant",
                message_text=reply,
                intent=stored_intent,
                intent_confidence=intent_confidence,
                sources=sources,
                is_fallback=(not sources),
            )
        return {"reply": reply, "sources": sources, "session_id": session_id, "intent": stored_intent}
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/meeting/approve/{token}")
def approve_meeting(token: str) -> Dict[str, Any]:
    # Backward-compatible route: redirect to proposal form.
    return RedirectResponse(url=f"/api/meeting/propose/{token}", status_code=status.HTTP_302_FOUND)


@app.get("/api/meeting/propose/{token}")
def meeting_proposal_form(token: str) -> HTMLResponse:
    page = f"""
    <html>
      <head>
        <title>Propose Meeting Slot</title>
        <style>
          body {{ font-family: Arial, sans-serif; background:#0f1522; color:#eef1f8; padding:24px; }}
          .card {{ max-width:560px; margin:0 auto; background:#171f30; border:1px solid #2c3d61; border-radius:12px; padding:18px; }}
          label {{ display:block; margin-top:12px; margin-bottom:6px; }}
          input, textarea {{ width:100%; padding:10px; border-radius:8px; border:1px solid #42567f; background:#0f1522; color:#eef1f8; }}
          button {{ margin-top:14px; padding:10px 14px; border-radius:8px; border:1px solid #4c6fff; background:#22389f; color:#fff; cursor:pointer; }}
          .hint {{ font-size:13px; color:#94a3b8; margin-top:6px; }}
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Propose a slot for this request</h2>
          <p class="hint">All times must be <strong>US Eastern (EST/ET)</strong> — Jayanth is in NJ. Main time is what they see first; optional alternatives appear if they need another slot.</p>
          <form method="post" action="/api/meeting/propose/{token}">
            <label>Primary proposed day/time</label>
            <input name="proposed_time" placeholder="e.g., Tuesday, Apr 2 at 4:30 PM EST" required />
            <label>Optional note</label>
            <textarea name="note" rows="3" placeholder="Add any message for the requester"></textarea>
            <label>Alternative slot 1 (optional)</label>
            <input name="alt_slot_1" placeholder="e.g., Wed 3pm EST" />
            <label>Alternative slot 2 (optional)</label>
            <input name="alt_slot_2" placeholder="e.g., Thu 10am EST" />
            <label>Alternative slot 3 (optional)</label>
            <input name="alt_slot_3" placeholder="e.g., Fri 1pm EST" />
            <button type="submit">Send proposal email</button>
          </form>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=page)


@app.post("/api/meeting/propose/{token}")
def propose_meeting_slot(
    token: str,
    proposed_time: str = Form(...),
    note: str = Form(""),
    alt_slot_1: str = Form(""),
    alt_slot_2: str = Form(""),
    alt_slot_3: str = Form(""),
) -> HTMLResponse:
    try:
        alternatives = [alt_slot_1, alt_slot_2, alt_slot_3]
        result = meeting_coordinator.propose_slot(
            token=token,
            proposed_time=proposed_time.strip(),
            note=note.strip(),
            alternative_slots=alternatives,
        )
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Meeting request not found.")
        if result["status"] == "invalid_state":
            reason = html.escape(str(result.get("reason") or "Invalid state"))
            return HTMLResponse(
                content=(
                    f"<html><body style='font-family:Arial;padding:24px;background:#0f1522;color:#eef1f8;'>"
                    f"<h2>Cannot send proposal</h2><p>{reason}</p></body></html>"
                ),
                status_code=400,
            )
        sent = bool(result.get("proposal_email_sent"))
        body = (
            "<html><body style='font-family:Arial;padding:24px;background:#0f1522;color:#eef1f8;'>"
            "<h2>Proposal sent ✅</h2>"
        )
        if sent:
            body += "<p>The requester was emailed with <strong>Yes, this works</strong> and "
            "<strong>Need another slot</strong> (they can pick your alternatives or type their own).</p>"
        else:
            body += "<p>Could not send email (check SMTP). Proposal was still saved.</p>"
        body += "</body></html>"
        return HTMLResponse(content=body)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/meeting/counter/{response_token}")
def meeting_counter_form(response_token: str) -> HTMLResponse:
    entry = meeting_coordinator.get_entry_for_counter_page(response_token)
    if not entry:
        raise HTTPException(status_code=404, detail="Link not found or expired.")
    st = (entry.get("status") or "").strip()
    if st == "confirmed":
        return HTMLResponse(
            "<html><body style='font-family:Arial;padding:24px;'><h2>Already confirmed ✅</h2>"
            "<p>This meeting was already confirmed. Check your email for the Meet link.</p></body></html>"
        )
    if st == "awaiting_host_proposal":
        return HTMLResponse(
            "<html><body style='font-family:Arial;padding:24px;'><h2>Thanks — we got your availability</h2>"
            "<p>Jayanth will send a new proposed time by email. Watch your inbox.</p></body></html>"
        )
    if st != "proposed":
        return HTMLResponse(
            content=(
                "<html><body style='font-family:Arial;padding:24px;'><h2>This link is not active</h2>"
                "<p>Use the latest email from Ada for the current options.</p></body></html>"
            ),
            status_code=400,
        )

    name = html.escape(entry.get("name") or "there")
    primary = html.escape(entry.get("proposed_time") or "")
    note = entry.get("proposed_note")
    note_html = f"<p><strong>Note from Jayanth:</strong> {html.escape(note)}</p>" if note else ""
    alts = entry.get("alternative_slots") or []

    alt_blocks: List[str] = []
    for i, slot in enumerate(alts):
        safe = html.escape(slot)
        req = " required" if i == 0 else ""
        alt_blocks.append(
            f'<label style="display:block;margin:10px 0;cursor:pointer;">'
            f'<input type="radio" name="choice" value="alt_{i}"{req} /> '
            f"<strong>Alternative {i + 1}:</strong> {safe}</label>"
        )
    alts_section = "\n".join(alt_blocks) if alt_blocks else ""

    if alts:
        choice_section = (
            "<p style='font-size:13px;color:#94a3b8;margin:0 0 10px;'>Times are <strong>US Eastern (EST/ET)</strong> (Jayanth / NJ).</p>"
            f"<p>Jayanth proposed <strong>{primary}</strong> first. If that does not work, choose one of these "
            f"or pick &quot;I&apos;ll type my own&quot; below.</p>"
            f"{note_html}"
            f"<div style='margin:16px 0;'>{alts_section}"
            f'<label style="display:block;margin:10px 0;cursor:pointer;">'
            f'<input type="radio" name="choice" value="custom" /> '
            f"I&apos;ll type my preferred time / availability</label></div>"
        )
    else:
        choice_section = (
            "<p style='font-size:13px;color:#94a3b8;margin:0 0 10px;'>Jayanth schedules in <strong>US Eastern (EST/ET)</strong>.</p>"
            f"<p>Jayanth proposed: <strong>{primary}</strong></p>{note_html}"
            f"<p>No extra alternatives were listed—share when you&apos;re free (prefer Eastern / EST) and Jayanth will propose again.</p>"
            '<input type="hidden" name="choice" value="custom" />'
        )

    custom_box = (
        '<label style="display:block;margin-top:14px;">Your availability '
        '(required if you chose &quot;I&apos;ll type my own&quot; above)</label>'
        '<textarea name="custom_text" rows="4" style="width:100%;max-width:520px;padding:10px;" '
        'placeholder="e.g. Wed 4pm EST, or any weekday after 5pm Eastern"'
        f"{' required' if not alts else ''}></textarea>"
    )

    page = f"""
    <html>
      <head>
        <title>Need another time?</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
          body {{ font-family: Arial, sans-serif; background:#0f1522; color:#eef1f8; padding:24px; line-height:1.5; }}
          .card {{ max-width:560px; margin:0 auto; background:#171f30; border:1px solid #2c3d61; border-radius:12px; padding:18px; }}
          button {{ margin-top:16px; padding:10px 18px; border-radius:8px; border:1px solid #4c6fff; background:#22389f; color:#fff; cursor:pointer; }}
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Hi {name} — need another slot?</h2>
          <form method="post" action="/api/meeting/counter/{response_token}">
            {choice_section}
            {custom_box}
            <button type="submit">Send to Jayanth</button>
          </form>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=page)


@app.post("/api/meeting/counter/{response_token}")
def meeting_counter_submit(
    response_token: str,
    choice: str = Form(""),
    custom_text: str = Form(""),
) -> HTMLResponse:
    result = meeting_coordinator.submit_requester_counter(
        response_token=response_token,
        choice=choice.strip(),
        custom_text=custom_text.strip(),
    )
    status = result.get("status")
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Not found.")
    if status == "invalid_state":
        return HTMLResponse(
            content="<html><body style='padding:24px;font-family:Arial;'><h2>Not available</h2>"
            "<p>This link may be outdated. Use the latest email from Ada.</p></body></html>",
            status_code=400,
        )
    if status == "invalid_choice":
        return HTMLResponse(
            content="<html><body style='padding:24px;font-family:Arial;'><h2>Pick an option</h2>"
            "<p>Choose one of the alternatives or select &quot;I&apos;ll type my own&quot; and fill the box.</p>"
            f"<p><a href=\"/api/meeting/counter/{html.escape(response_token, quote=True)}\">Go back</a></p></body></html>",
            status_code=400,
        )
    if status == "invalid_custom":
        return HTMLResponse(
            content="<html><body style='padding:24px;font-family:Arial;'><h2>Add a bit more detail</h2>"
            "<p>Please type at least a short preferred time or window.</p>"
            f"<p><a href=\"/api/meeting/counter/{html.escape(response_token, quote=True)}\">Go back</a></p></body></html>",
            status_code=400,
        )
    if status != "ok":
        raise HTTPException(status_code=400, detail=str(result))

    notified = result.get("jayanth_notified")
    body = (
        "<html><body style='padding:24px;font-family:Arial;background:#0f1522;color:#eef1f8;'>"
        "<h2>Thanks — Jayanth has been notified ✅</h2>"
        "<p>He&apos;ll review your availability and email a new proposed time. "
        "You can repeat this if needed until you confirm.</p>"
        "</body></html>"
    )
    if not notified:
        body = (
            "<html><body style='padding:24px;font-family:Arial;'>"
            "<h2>Saved your preference</h2>"
            "<p>We could not email Jayanth automatically (SMTP). Please reach out if you don&apos;t hear back.</p>"
            "</body></html>"
        )
    return HTMLResponse(content=body)


@app.get("/api/meeting/respond/{response_token}")
def respond_meeting_slot(response_token: str, decision: str) -> HTMLResponse:
    try:
        if (decision or "").strip().lower() == "reject":
            return RedirectResponse(
                url=f"/api/meeting/counter/{response_token}",
                status_code=status.HTTP_302_FOUND,
            )
        result = meeting_coordinator.respond_to_proposal(response_token=response_token, decision=decision)
        state = result.get("status")
        if state == "not_found":
            raise HTTPException(status_code=404, detail="Meeting response token not found.")
        if state == "invalid_decision":
            raise HTTPException(status_code=400, detail="Invalid decision.")
        if state == "invalid_state":
            return HTMLResponse(
                content=(
                    "<html><body style='padding:24px;font-family:Arial;'><h2>This link is no longer active</h2>"
                    "<p>A newer proposal may have been sent—check your latest email from Ada for updated Yes / Need another slot links.</p>"
                    "</body></html>"
                ),
                status_code=400,
            )
        if state == "confirmed":
            return HTMLResponse(
                "<h2>Thanks! Your meeting is confirmed ✅</h2>"
                "<p>We have emailed your confirmation details, including the Google Meet link.</p>"
            )
        return HTMLResponse("<h2>Response received.</h2>")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/chats/sessions")
def list_chat_sessions(limit: int = 50, _: None = Depends(_require_chat_admin_auth)) -> Dict[str, Any]:
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat persistence is disabled. Set DATABASE_URL.")
    return {"sessions": chat_store.list_sessions(limit=limit)}


@app.get("/api/chats/{session_id}")
def get_chat_session_messages(session_id: str, _: None = Depends(_require_chat_admin_auth)) -> Dict[str, Any]:
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat persistence is disabled. Set DATABASE_URL.")
    return {"session_id": session_id, "messages": chat_store.get_session_messages(session_id)}


@app.post("/api/admin/postgres/test")
def test_postgres_connection(
    payload: PostgresTestRequest,
    _: None = Depends(_require_chat_admin_auth),
) -> Dict[str, Any]:
    try:
        with psycopg.connect(
            host=payload.host.strip(),
            port=payload.port,
            dbname=payload.database.strip(),
            user=payload.username.strip(),
            password=payload.password,
            sslmode=payload.sslmode.strip() or "disable",
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user, current_database(), version()")
                current_user, current_database, version = cur.fetchone()
                cur.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                      AND table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name
                    """
                )
                table_rows = cur.fetchall()
        return {
            "ok": True,
            "current_user": current_user,
            "current_database": current_database,
            "postgres_version": version,
            "tables": [{"schema": row[0], "name": row[1]} for row in table_rows],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Postgres connection failed: {exc}") from exc


@app.post("/api/admin/login")
def admin_login(payload: AdminLoginRequest, request: Request) -> Response:
    if not chat_admin_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not _verify_admin_credentials(payload.username.strip(), payload.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = _create_admin_session_token()
    response = Response(content='{"ok":true}', media_type="application/json")
    _set_admin_session_cookie(response, request, token)
    return response


@app.post("/api/admin/logout")
def admin_logout(request: Request, _: None = Depends(_require_chat_admin_auth)) -> Response:
    token = (request.cookies.get(chat_admin_session_cookie_name) or "").strip()
    if token:
        chat_admin_sessions.pop(token, None)
    response = Response(content='{"ok":true}', media_type="application/json")
    response.delete_cookie(chat_admin_session_cookie_name)
    return response


@app.get("/api/admin/me")
def admin_me(request: Request, _: None = Depends(_require_chat_admin_auth)) -> Dict[str, Any]:
    return {"ok": True, "authenticated": _is_request_admin_authenticated(request)}


@app.get("/admin/login")
def admin_login_page(request: Request) -> FileResponse:
    if not chat_admin_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if _is_request_admin_authenticated(request):
        return RedirectResponse(url="/admin/chats", status_code=status.HTTP_302_FOUND)
    return FileResponse(REPO_ROOT / "backend" / "chat_login.html")


@app.get("/admin/chats")
def chats_admin_page(request: Request) -> FileResponse:
    if not chat_admin_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not _is_request_admin_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)
    return FileResponse(REPO_ROOT / "backend" / "chat_admin.html")


# Serve the existing static portfolio so chatbot can call same-origin /api routes.
app.mount("/assets", StaticFiles(directory=str(REPO_ROOT / "assets")), name="assets")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(REPO_ROOT / "index.html")
