from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from openai import OpenAI
from pypdf import PdfReader

from .resume_catalog import load_resume_entries


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ResumeRagEngine:
    def __init__(self, repo_root: Path, data_dir: Path) -> None:
        self.repo_root = repo_root
        self.data_dir = data_dir
        self.index_path = data_dir / "vector_index.json"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.intent_model = os.getenv("OPENAI_INTENT_MODEL", self.chat_model)
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.chunk_size = _safe_int(os.getenv("RAG_CHUNK_SIZE"), 900)
        self.chunk_overlap = _safe_int(os.getenv("RAG_CHUNK_OVERLAP"), 150)
        self.top_k = _safe_int(os.getenv("RAG_TOP_K"), 5)
        # 0 = index all supported files under Resume/ (including nested folders).
        self.resume_limit = _safe_int(os.getenv("RAG_RESUME_LIMIT"), 0)
        self.min_relevance_score = _safe_float(os.getenv("RAG_MIN_RELEVANCE_SCORE"), 0.20)
        self.index_data: Dict[str, Any] = {"chunks": []}
        self._emoji_cursor = 0
        self._last_emoji: Optional[str] = None
        self._resume_entries_cache: Optional[List[Dict[str, Any]]] = None
        self.load_index()

    def invalidate_resume_cache(self) -> None:
        self._resume_entries_cache = None

    @staticmethod
    def _finalize_reply(text: str) -> str:
        """Keep response complete and avoid trailing cut-off fragments."""
        reply = (text or "").strip()
        if not reply:
            return reply
        if reply[-1] in ".!?":
            return reply
        # Cut to the last complete sentence if output was truncated.
        matches = list(re.finditer(r"[.!?](?:\"|')?\s", reply))
        if matches:
            cut_idx = matches[-1].end()
            return reply[:cut_idx].strip()
        # Fallback: avoid raw cut-off word endings.
        return reply.rstrip(" ,;:-")

    def _ensure_emoji(self, reply: str, intent: str = "general") -> str:
        emoji_re = re.compile(
            "[\U0001F300-\U0001FAFF\u2600-\u27BF]",
            flags=re.UNICODE,
        )
        if emoji_re.search(reply or ""):
            return reply

        pools = {
            "greeting": ["👋", "✨", "😄", "🤝"],
            "abusive": ["🙏", "🙂", "🤝", "🧭"],
            "out_of_scope": ["🙂", "💬", "🤝", "✨"],
            "hiring_role_clarification": ["🎯", "🚀", "💼", "✨"],
            "resume_attachment": ["📄", "✨", "🚀", "💼"],
            "resume_attachment_post_meeting": ["📄", "✨", "🎉", "🚀"],
            "resume_focus_clarification": ["💡", "✨", "🎯", "🙌"],
            "resume_no_fit_connect": ["🤝", "✨", "📩", "💬"],
            "information_request": ["🚀", "💡", "✨", "🙌", "😄"],
            "general": ["✨", "🚀", "💡", "🙌", "😄"],
        }
        pool = pools.get(intent, pools["general"])
        idx = self._emoji_cursor % len(pool)
        emoji = pool[idx]
        self._emoji_cursor += 1
        if emoji == self._last_emoji:
            emoji = pool[(idx + 1) % len(pool)]
        self._last_emoji = emoji
        return (reply.rstrip() + f" {emoji}").strip()

    @staticmethod
    def _reply_has_trailing_question(text: str) -> bool:
        """True if the message already ends with a question (ignores trailing emoji after ?)."""
        t = (text or "").strip()
        if not t:
            return False
        # Strip trailing whitespace/emoji so "...? 🤝" counts as already having a question.
        emoji_tail = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\s]+$")
        stripped = emoji_tail.sub("", t).rstrip()
        return stripped.endswith("?")

    def _intent_followup_constraint(self, intent: str, *, skip_meeting_prompt: bool = False) -> str:
        """Natural-language rules for the follow-up synthesizer (not a fixed question string)."""
        no_meet = (
            " Do NOT ask about scheduling calls, meetings, booking time, or setting up a call."
            if skip_meeting_prompt
            else ""
        )
        mapping = {
            "information_request": (
                "Ground the question in what the user asked and what Ada just said. "
                "Avoid generic closings; vary wording each turn."
                + no_meet
            ),
            "general": "One relevant conversational question continuing this thread." + no_meet,
            "greeting": (
                "Invite a natural next step about Jayanth (experience, projects, skills, research). "
                "Warm and specific; do not repeat identical wording across turns."
            ),
            "abusive": (
                "Politely ask the user to stay respectful and redirect toward Jayanth's profile "
                "(experience, skills, work)—one short question only."
            ),
            "out_of_scope": (
                "Suggest a sensible next step (e.g. something about Jayanth's profile or how to reach him). "
                "Stay friendly."
                + no_meet
            ),
            "hiring_role_clarification": (
                "Ask one sharp clarifying question about the role, seniority, stack, or what 'good fit' means for them."
            ),
            "resume_attachment": (
                "Ask something tied to their resume request or what they are evaluating next (role, stack, focus)."
            ),
            "resume_attachment_post_meeting": (
                "Ask what part of Jayanth's background, skills, or impact to explore next."
                + no_meet
                + (" They already have a connect request in this chat." if skip_meeting_prompt else "")
            ),
            "resume_focus_clarification": (
                "Ask which role, seniority, stack, or domain they need the resume optimized for."
            ),
            "resume_no_fit_connect": (
                "Offer one clear next question about fit, role needs, or talking with Jayanth—natural, not checklist-like."
            ),
        }
        return mapping.get(
            intent,
            "Ask one short, interrogative follow-up that fits Ada's reply and the user's last message."
            + no_meet,
        )

    def _synthesize_followup_question(
        self,
        user_message: str,
        history: List[Dict[str, str]] | None,
        assistant_reply: str,
        intent: str,
        *,
        skip_meeting_prompt: bool = False,
    ) -> str:
        """Ask a lightweight LLM for one closing question tailored to this exchange and intent."""
        prior = history or []
        recent: List[str] = []
        for item in prior[-6:]:
            role = item.get("role", "")
            content = (item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                content = content.replace("\n", " ").strip()
                if len(content) > 420:
                    content = content[:417].rstrip() + "..."
                recent.append(f"{role}: {content}")
        transcript = "\n".join(recent)[-2000:]
        body = (assistant_reply or "").strip()
        body = body.replace("\n", " ").strip()
        if len(body) > 900:
            body = body[:897].rstrip() + "..."

        constraint = self._intent_followup_constraint(intent, skip_meeting_prompt=skip_meeting_prompt)
        system_prompt = (
            "You write exactly ONE short follow-up question for Ada, Jayanth's portfolio chat assistant. "
            f"Routing intent (for tone only): {intent}. "
            f"Constraint: {constraint} "
            "The question is spoken BY Ada TO the visitor: Ada has the context; the visitor is learning about Jayanth. "
            "NEVER ask the visitor to provide, supply, explain, or recount Jayanth's jobs, metrics, or achievements "
            "(reject patterns like 'Can you provide more details about Jayanth…', 'Could you tell me about his role at…' where the burden is on the visitor). "
            "USE invitations for Ada to continue: e.g. 'Want me to go deeper on his Harvest internship?', "
            "'Should I break down how he approached that analysis?', 'What aspect should I expand on next—tools, scale, or outcomes?'. "
            "The question must reflect the user's latest message and Ada's latest reply—never a generic filler. "
            "Vary phrasing; never reuse the same closing question across different turns unless the user explicitly repeated the same ask. "
            "No greeting lead-in, no bullet points, no preface. Output only the question, ending with ? Maximum 22 words."
        )
        user_block = (
            f"Latest user message:\n{(user_message or '').strip()}\n\n"
            f"Ada's latest reply (may end without a question):\n{body or '(empty)'}\n\n"
            f"Recent transcript:\n{transcript or '(none)'}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.55,
                max_tokens=70,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_block},
                ],
            )
            q = (completion.choices[0].message.content or "").strip()
            q = q.split("\n")[0].strip().strip("\"'“”")
            if not q:
                raise ValueError("empty follow-up")
            if not q.endswith("?"):
                q = q.rstrip(".!… ") + "?"
            low = q.lower()
            if re.search(r"\b(can|could)\s+you\s+(provide|share|give|spell|tell)\b", low) and (
                "jayanth" in low
                or "his role" in low
                or "intern" in low
                or "work at" in low
                or "achievement" in low
            ):
                q = "Want me to go a layer deeper on that part of his experience?"
            return q
        except Exception:
            return "What would you like to ask next about Jayanth?"

    def _ensure_trailing_question(
        self,
        reply: str,
        intent: str = "general",
        *,
        user_message: str = "",
        history: List[Dict[str, str]] | None = None,
        skip_meeting_prompt: bool = False,
    ) -> str:
        text = (reply or "").strip()
        if not text:
            return text
        if self._reply_has_trailing_question(text):
            return text
        follow = self._synthesize_followup_question(
            user_message,
            history,
            text,
            intent,
            skip_meeting_prompt=skip_meeting_prompt,
        )
        suffix = "" if text[-1] in ".!?" else "."
        return f"{text}{suffix} {follow}".strip()

    @staticmethod
    def _repoint_inverted_profile_closing_question(reply: str) -> str:
        """If the closing asks the visitor to supply Jayanth's facts, flip it to an Ada-led offer."""
        t = (reply or "").strip()
        if not t or "?" not in t:
            return t

        def _is_bad_closing(closing: str) -> bool:
            low = closing.lower()
            if not re.search(r"\b(can|could)\s+you\s+(provide|share|give|spell|tell)\b", low):
                return False
            return bool(
                re.search(
                    r"jayanth|his role|his work|internship|intern at|achievement|achievements at|experience at",
                    low,
                )
            )

        parts = t.rsplit("\n\n", 1)
        if len(parts) == 2:
            body, closing = parts[0].strip(), parts[1].strip()
            if _is_bad_closing(closing):
                return f"{body}\n\nWant me to unpack that part of his background a bit more?".strip()

        qpos = t.rfind("?")
        before = t[:qpos]
        for sep in ("\n\n", ". ", "! "):
            idx = before.rfind(sep)
            if idx != -1:
                closing = t[idx + len(sep) : qpos + 1].strip()
                if _is_bad_closing(closing):
                    prefix = t[: idx + len(sep)].rstrip()
                    return f"{prefix} Want me to go a layer deeper on that stretch of his work?".strip()
        if _is_bad_closing(t):
            return "Want me to unpack that part of his background a bit more?"
        return t

    def _ensure_information_reply_ends_with_question(
        self,
        reply: str,
        user_message: str,
        history: List[Dict[str, str]] | None,
        *,
        skip_meeting_scheduling_prompt: bool = False,
    ) -> str:
        """Prefer model-authored questions; if missing, add one LLM-synthesized follow-up (not a fixed string)."""
        text = (reply or "").strip()
        if not text:
            return text
        if self._reply_has_trailing_question(text):
            return text
        follow = self._synthesize_followup_question(
            user_message,
            history,
            text,
            "information_request",
            skip_meeting_prompt=skip_meeting_scheduling_prompt,
        )
        suffix = "" if text[-1] in ".!?" else "."
        return f"{text}{suffix} {follow}".strip()

    @staticmethod
    def _ensure_two_paragraphs(reply: str) -> str:
        """Normalize output to exactly two readable paragraphs."""
        text = (reply or "").strip()
        if not text:
            return text

        normalized = text.replace("\r\n", "\n")
        parts = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
        if len(parts) >= 2:
            return f"{parts[0]}\n\n{' '.join(parts[1:]).strip()}".strip()

        one_block = parts[0] if parts else normalized
        sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", one_block) if seg.strip()]
        if len(sentences) >= 3:
            split_idx = max(1, len(sentences) // 2)
            first = " ".join(sentences[:split_idx]).strip()
            second = " ".join(sentences[split_idx:]).strip()
            if first and second:
                return f"{first}\n\n{second}".strip()

        words = one_block.split()
        if len(words) >= 16:
            mid = len(words) // 2
            return f"{' '.join(words[:mid]).strip()}\n\n{' '.join(words[mid:]).strip()}".strip()

        return f"{one_block}\n\nWould you like me to share more specific highlights?".strip()

    @staticmethod
    def _default_resume_connect_offer_paragraph() -> str:
        return (
            "If you'd like to go deeper, I can help set up a relaxed chat with Jayanth about how he fits your role and what you need — "
            "want me to line that up?"
        )

    @staticmethod
    def _ensure_three_paragraphs(reply: str) -> str:
        """Normalize resume attachment replies to exactly three paragraphs (last = connect offer)."""
        text = (reply or "").strip()
        if not text:
            return text
        default_third = ResumeRagEngine._default_resume_connect_offer_paragraph()
        normalized = text.replace("\r\n", "\n")
        parts = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
        if len(parts) >= 3:
            merged_tail = " ".join(parts[2:]).strip()
            return f"{parts[0]}\n\n{parts[1]}\n\n{merged_tail}".strip()
        if len(parts) == 2:
            return f"{parts[0]}\n\n{parts[1]}\n\n{default_third}".strip()
        one = parts[0] if parts else normalized
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", one) if s.strip()]
        if len(sentences) >= 4:
            n = len(sentences)
            a = max(1, n // 3)
            b = max(1, (n - a) // 2)
            p1 = " ".join(sentences[:a])
            p2 = " ".join(sentences[a : a + b])
            p3 = " ".join(sentences[a + b :])
            return f"{p1}\n\n{p2}\n\n{p3}".strip()
        if len(sentences) == 3:
            return f"{sentences[0]}\n\n{sentences[1]}\n\n{sentences[2]}".strip()
        if len(sentences) == 2:
            return f"{sentences[0]}\n\n{sentences[1]}\n\n{default_third}".strip()
        return f"{one}\n\n{default_third}".strip()

    @staticmethod
    def thread_has_completed_meeting_request(history: Optional[List[Dict[str, str]]]) -> bool:
        """True if this thread already has a completed connect / meeting handoff (verified or no-verify path)."""
        if not history:
            return False
        pending_only = (
            "6-digit",
            "paste it here",
            "paste the code",
            "waiting for the",
            "once it checks out",
            "when it arrives",
        )
        completion_phrases = (
            "you're verified",
            "email has been verified",
            "verified successfully",
            "perfect — you're verified",
            "pass any extra note",
            "extra note to him",
            "your request is in",
            "request is captured",
            "notified jayanth",
            "jayanth has been notified",
            "logged a connect request",
            "already have a connect request",
            "existing meeting",
            "linked this chat",
            "reconnected this chat",
            "linked to your existing",
        )
        for item in history:
            if item.get("role") != "assistant":
                continue
            text = (item.get("content") or "").lower()
            if any(marker in text for marker in pending_only) and not any(
                done in text for done in ("you're verified", "email has been verified", "verified successfully")
            ):
                continue
            if any(phrase in text for phrase in completion_phrases):
                return True
        return False

    def intent_reply(
        self,
        intent: str,
        message: str,
        history: List[Dict[str, str]] | None = None,
        *,
        resume_skip_connect_offer: bool = False,
        out_of_scope_skip_schedule_offer: bool = False,
    ) -> str:
        """LLM-generated reply for non-RAG intents (greeting, abusive, out_of_scope)."""
        prior = history or []
        has_prior_assistant_turn = any(
            item.get("role") == "assistant" and str(item.get("content", "")).strip()
            for item in prior
        )
        base_system = (
            "You are Ada, Jayanth's AI assistant. "
            "Tone: friendly, conversational, and charismatic—sound like a warm, sharp human who listens and responds, "
            "never stiff, robotic, scripted, or like a support ticket. "
            "STRICT: Read the full conversation above. Your reply must directly address the user's latest message—what they said, "
            "asked, or implied in this turn—and build on prior turns when relevant. "
            "STRICT: Do not repeat yourself. Never reuse the same sentences, catchphrases, or whole paragraphs you already wrote "
            "earlier in this chat. If you already asked a question, do not ask it again verbatim—acknowledge what they answered "
            "and move forward. Vary wording every time. "
            "STRICT: Do not paste canned blocks, boilerplate disclaimers, or identical closings turn after turn. "
            "Keep responses concise. "
            "Do not speak as if you are Jayanth personally; talk about him in third person. "
            "Add emojis naturally (at least 1, at most 3). "
            "Speak highly of Jayanth with confident, positive language grounded in provided context. "
            "Write in paragraph style for easy reading, not bullet points. "
            "Keep replies crisp and high-impact without feeling abrupt. "
            "Always finish with a complete sentence; never end mid-sentence. "
            "Use blank lines between paragraphs when there are multiple paragraphs. "
            "Do not sound like customer support or use corporate filler. "
            "Your scope is only Jayanth's profile: experience, skills, projects, research, education, and contact. "
            "Never imply you can answer anything outside that scope. "
            "Avoid repetitive opener words like 'Hey', 'Hey there', or 'Hi' on every message. "
            "Use a greeting opener only for the first assistant message in a conversation, "
            "or when the user explicitly greets you."
        )
        if intent == "resume_attachment":
            if resume_skip_connect_offer:
                para_rule = (
                    "Use exactly 2 short paragraphs; each should be 1-3 short sentences. "
                    "Do NOT offer to schedule a call, book a session, or arrange a meeting with Jayanth — "
                    "they already submitted a connect/meeting request in this chat. "
                    "End with a question about Jayanth's work, skills, or what to explore next (not scheduling)."
                )
            else:
                para_rule = (
                    "Use exactly 3 short paragraphs; each should be 1-3 short sentences. "
                    "The third and final paragraph must end with a natural yes/no question (connect-session offer). "
                    "Paragraphs 1–2 may be statements; do not put the main question only in paragraph 1 or 2."
                )
        else:
            para_rule = (
                "Use exactly 2 short paragraphs; each paragraph should be 1-3 short sentences. "
                "Prefer conversational language and end every reply with one natural question."
            )

        if intent in {"greeting", "abusive"}:
            return self._greeting_or_abusive_via_compose(
                intent=intent,
                message=message,
                history=history,
                base_system=base_system,
                para_rule=para_rule,
                has_prior_assistant_turn=has_prior_assistant_turn,
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": base_system + " " + para_rule}]
        for item in prior[-8:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        intent_instruction = {
            "greeting": (
                "Respond warmly. If this is the first assistant reply in the chat, include a short intro that says "
                "you are Ada, Jayanth's AI assistant and invite profile questions. "
                "If this is not the first assistant reply, answer naturally without repeating the same greeting opener. "
                "Do not say things like 'ask me anything else'. End with one conversational question. "
                "Keep it in exactly 2 short paragraphs."
            ),
            "abusive": (
                "Set a respectful boundary politely. Ask the user to keep it respectful and redirect them to "
                "questions about Jayanth's profile. Keep this response short, in exactly 2 small paragraphs, "
                "and end with one redirecting question."
            ),
            "out_of_scope": (
                "Say you do not have that detail in your current profile context, but keep the tone very friendly. "
                "Offer to help connect with Jayanth or schedule a call. Include BOTH contact lines exactly once:\n"
                "Email: jayanthdasamantharao@gmail.com\n"
                "LinkedIn: https://www.linkedin.com/in/djayanth/\n"
                "Keep it short, in exactly 2 paragraphs, and end with one conversational question."
            ),
            "hiring_role_clarification": (
                "The user appears to be hiring but has not shared a specific role yet. "
                "Ask a clear clarification question first to get role, level/seniority, and core skills expected. "
                "Keep tone warm and human. Keep it short, in exactly 2 concise paragraphs, and end with one direct question."
            ),
            "resume_attachment": (
                (
                    "The user asked for Jayanth's resume/CV as a file. Use exactly 2 short paragraphs separated by blank lines. "
                    "Confirm they can download the resume from the button or link in this chat (do not paste long URLs); "
                    "briefly highlight relevant strengths. "
                    "Do NOT ask whether they want to schedule a call or arrange a session — they already completed a "
                    "connect/meeting request earlier in this chat. "
                    "Do not mention company names, client names, or internal folder or file paths."
                )
                if resume_skip_connect_offer
                else (
                    "The user asked for Jayanth's resume/CV as a file. Use exactly 3 short paragraphs separated by blank lines. "
                    "Paragraphs 1 and 2: confirm they can download the resume from the button or link in this chat (do not paste long URLs); "
                    "keep tone warm and brief. "
                    "Paragraph 3 (must be the LAST paragraph): offer to help connect them with Jayanth for a detailed session "
                    "about how he fits their role and their requirements — this paragraph must end with a clear yes/no question "
                    "(e.g. whether they would like you to arrange that). "
                    "Do not mention company names, client names, or internal folder or file paths."
                )
            ),
            "resume_focus_clarification": (
                "The user asked for a resume/CV but has not said what role, stack, or focus they care about yet. "
                "Ask what they are looking for (role, seniority, stack, or domain). "
                "Briefly mention Jayanth is diversified across software development, machine learning, AI, computer vision, "
                "and other technical areas — so the right file depends on what they need. "
                "Do not mention company or client names. Exactly 2 short paragraphs; end with a question."
            ),
            "resume_no_fit_connect": (
                "There is no resume variant in the library that clearly matches what they asked for. "
                "Say so in a warm, conversational way—charismatic but honest—without exposing internal file or folder names. "
                "In the second paragraph, briefly offer to help coordinate a conversation with Jayanth about fit for their role — "
                "do NOT list or repeat required fields (name, email, role, time slots) as a checklist; keep one clear call-to-action. "
                "Exactly 2 short paragraphs; end with a single conversational question (only one question in the whole reply)."
            ),
        }.get(intent, "Reply briefly and helpfully.")
        if intent == "out_of_scope" and out_of_scope_skip_schedule_offer:
            intent_instruction = (
                "Say you do not have that detail in your current profile context, but keep the tone very friendly. "
                "Do NOT offer to schedule a call or meeting — they already have a connect request in this thread. "
                "Include BOTH contact lines exactly once:\n"
                "Email: jayanthdasamantharao@gmail.com\n"
                "LinkedIn: https://www.linkedin.com/in/djayanth/\n"
                "Keep it short, in exactly 2 paragraphs, and end with one conversational question that is not about scheduling."
            )

        _word_cap_resume_attach = "130"
        if intent == "resume_attachment":
            if resume_skip_connect_offer:
                user_extra = (
                    f"Instruction: {intent_instruction}\n"
                    "Return exactly 2 short paragraphs separated by one blank line. "
                    f"Keep this concise and complete. Do not exceed {_word_cap_resume_attach} words. "
                    "Do NOT offer to schedule calls or meetings. End with a question mark. "
                    "Respond to this specific exchange; do not repeat wording from earlier assistant turns in the chat."
                )
            else:
                user_extra = (
                    f"Instruction: {intent_instruction}\n"
                    "Return exactly 3 short paragraphs separated by one blank line. "
                    f"Keep this concise and complete. Do not exceed {_word_cap_resume_attach} words. "
                    "The last paragraph must be the connect-session offer and must end with a question mark. "
                    "Respond to this specific exchange; do not repeat wording from earlier assistant turns in the chat."
                )
        else:
            user_extra = (
                f"Instruction: {intent_instruction}\n"
                "Return exactly 2 short paragraphs separated by one blank line. "
                f"Keep this concise and complete. Do not exceed "
                f"{'100' if intent in {'resume_focus_clarification', 'resume_no_fit_connect'} else '75'} words. "
                "The final line must end with a question mark. "
                "Follow the thread: respond to what the user just said; do not ignore new details they provided. "
                "Do not mirror or repeat your own previous assistant messages from this chat."
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Intent: {intent}\n"
                    f"User message: {message}\n"
                    f"Has prior assistant turn: {has_prior_assistant_turn}\n"
                    f"{user_extra}"
                ),
            }
        )
        try:
            _wide = intent in {"resume_focus_clarification", "resume_no_fit_connect", "resume_attachment"}
            completion = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=_safe_float(os.getenv("RAG_TEMPERATURE"), 0.2),
                max_tokens=(
                    220
                    if (intent == "resume_attachment" and resume_skip_connect_offer)
                    else (280 if intent == "resume_attachment" else (200 if _wide else 120))
                ),
                messages=messages,
            )
            reply = completion.choices[0].message.content or ""
            reply = self._finalize_reply(reply)
            if intent == "resume_attachment":
                reply = self._ensure_emoji(
                    reply,
                    intent="resume_attachment_post_meeting" if resume_skip_connect_offer else intent,
                )
                if resume_skip_connect_offer:
                    reply = reply or (
                        "You can download Jayanth's resume using the link in this chat.\n\n"
                        "You already have a connect request in with Jayanth — tell me which part of his background you want next."
                    )
                    reply = self._ensure_trailing_question(
                        reply,
                        intent="resume_attachment_post_meeting",
                        user_message=message,
                        history=history,
                        skip_meeting_prompt=True,
                    )
                    return self._ensure_two_paragraphs(reply)
                reply = reply or (
                    "You can download Jayanth's resume using the link in this chat.\n\n"
                    "If anything looks like a strong match, tell me what role you're hiring for next.\n\n"
                    + self._default_resume_connect_offer_paragraph()
                )
                return self._ensure_three_paragraphs(reply)
            reply = self._ensure_trailing_question(
                reply,
                intent=intent,
                user_message=message,
                history=history,
                skip_meeting_prompt=bool(
                    intent == "out_of_scope" and out_of_scope_skip_schedule_offer
                ),
            )
            reply = self._ensure_emoji(reply, intent=intent)
            reply = reply or "I'm Ada, Jayanth's AI assistant ✨ Ask me anything about Jayanth's work and experience."
            return self._ensure_two_paragraphs(reply)
        except Exception:
            # Runtime-safe fallback only if LLM call fails.
            if intent == "abusive":
                return self._ensure_two_paragraphs(
                    self._ensure_trailing_question(
                        "Let's keep it respectful 🙏 Ask me anything about Jayanth's experience, skills, projects, or research.",
                        intent="abusive",
                        user_message=message,
                        history=history,
                    )
                )
            if intent == "out_of_scope":
                return self._ensure_two_paragraphs(
                    self._ensure_trailing_question(
                        "I might not have that detail in my current profile context yet 🙂 "
                        "You can reach Jayanth at jayanthdasamantharao@gmail.com or "
                        "https://www.linkedin.com/in/djayanth/.",
                        intent="out_of_scope",
                        user_message=message,
                        history=history,
                        skip_meeting_prompt=out_of_scope_skip_schedule_offer,
                    )
                )
            return self._ensure_two_paragraphs(
                self._ensure_trailing_question(
                    "Hey! I'm Ada, Jayanth's AI assistant 👋 Ask me anything about Jayanth's experience, skills, projects, or research.",
                    intent="greeting",
                    user_message=message,
                    history=history,
                )
            )

    def compose_flow_reply(
        self,
        *,
        instruction: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        facts: Optional[Dict[str, Any]] = None,
        fallback_reply: str,
        max_tokens: int = 340,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """LLM-generated reply (same chat model as the rest of Ada). Facts in JSON are authoritative for app-layer flows."""
        facts_clean: Dict[str, Any] = {}
        for key, value in (facts or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                facts_clean[key] = value
            elif isinstance(value, dict):
                facts_clean[key] = value
            else:
                facts_clean[key] = str(value)

        default_system = (
            "You are Ada, Jayanth's AI assistant. Write ONE chat reply in Ada's warm, natural voice. "
            "Follow INSTRUCTION and FACTS_JSON exactly: never invent verification codes, email addresses, or outcomes not stated in facts. "
            "If facts include booleans (e.g. jayanth_notified), reflect them accurately. "
            "Respond to the latest user message and thread; do not sound like a canned system message. "
            "Use 1–3 short paragraphs. Use emojis sparingly (0–2). End with a question when appropriate."
        )
        system = (system_prompt or default_system).strip()
        payload = json.dumps({"instruction": instruction, "facts": facts_clean}, ensure_ascii=False)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for item in (history or [])[-10:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"FACTS_AND_INSTRUCTION:\n{payload}\n\nLatest user message:\n{message.strip()}",
            }
        )
        try:
            self._ensure_api_key()
            temp = (
                float(temperature)
                if temperature is not None
                else _safe_float(os.getenv("FLOW_REPLY_TEMPERATURE", "0.4"), 0.4)
            )
            completion = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=temp,
                max_tokens=max_tokens,
                messages=messages,
            )
            reply = self._finalize_reply(completion.choices[0].message.content or "")
            return reply.strip() if reply.strip() else fallback_reply
        except Exception:
            return fallback_reply

    def _greeting_or_abusive_via_compose(
        self,
        *,
        intent: str,
        message: str,
        history: Optional[List[Dict[str, str]]],
        base_system: str,
        para_rule: str,
        has_prior_assistant_turn: bool,
    ) -> str:
        """greeting / abusive: same LLM path as compose_flow_reply, then intent_reply post-processing."""
        if intent == "greeting":
            task = (
                "Respond warmly. If this is the first assistant reply in the chat, include a short intro that says "
                "you are Ada, Jayanth's AI assistant and invite profile questions. "
                "If this is not the first assistant reply, answer naturally without repeating the same greeting opener. "
                "Do not say things like 'ask me anything else'. End with one conversational question. "
                "Keep it in exactly 2 short paragraphs."
            )
            fb = (
                "Hey! I'm Ada, Jayanth's AI assistant 👋 Ask me anything about Jayanth's experience, skills, projects, or research."
            )
        else:
            task = (
                "Set a respectful boundary politely. Ask the user to keep it respectful and redirect them to "
                "questions about Jayanth's profile. Keep this response short, in exactly 2 small paragraphs, "
                "and end with one redirecting question."
            )
            fb = "Let's keep it respectful 🙏 Ask me anything about Jayanth's experience, skills, projects, or research."

        instruction = (
            f"{task} "
            "Use has_prior_assistant_turn in facts: if false, you may open with a short Ada intro (greeting intent only); "
            "if true, do not repeat the same greeting opener. "
            "Keep under 75 words. The final line must end with a question mark."
        )
        system_prompt = (
            base_system
            + "\n\n"
            + para_rule
            + "\n\n"
            "You will receive JSON with instruction and facts in the user message—follow it. "
            "Do not invent verification codes or meeting outcomes unless stated in facts."
        )
        reply = self.compose_flow_reply(
            instruction=instruction,
            message=message,
            history=history,
            facts={"has_prior_assistant_turn": has_prior_assistant_turn},
            fallback_reply=fb,
            max_tokens=200,
            system_prompt=system_prompt,
            temperature=_safe_float(os.getenv("RAG_TEMPERATURE"), 0.2),
        )
        reply = self._finalize_reply(reply)
        reply = self._ensure_trailing_question(
            reply, intent=intent, user_message=message, history=history
        )
        reply = self._ensure_emoji(reply, intent=intent)
        reply = reply or fb
        return self._ensure_two_paragraphs(reply)

    def assess_hiring_query(self, message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """LLM check for hiring intent and whether a specific role is provided."""
        prior = history or []
        recent_lines = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2400]
        if not conversation:
            return {"is_hiring": False, "has_specific_role": False, "role": None}

        system_prompt = (
            "You detect hiring/recruiting context for a personal portfolio assistant. "
            "Return strict JSON only with keys: "
            "{\"is_hiring\": boolean, \"has_specific_role\": boolean, \"role\": string|null}. "
            "Rules: "
            "is_hiring=true if the user implies recruiting, hiring, evaluating fit, or role suitability. "
            "has_specific_role=true only when an explicit role/title is present (example: 'ML Engineer', 'Agentic AI Engineer'). "
            "If no explicit role is provided, set has_specific_role=false and role=null. "
            "Do not hallucinate missing role titles."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.2,
                max_tokens=90,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            is_hiring = bool(parsed.get("is_hiring", False))
            has_specific_role = bool(parsed.get("has_specific_role", False))
            role_value = parsed.get("role")
            role_text = str(role_value).strip() if role_value is not None else None
            return {
                "is_hiring": is_hiring,
                "has_specific_role": has_specific_role,
                "role": role_text if role_text else None,
            }
        except Exception:
            return {"is_hiring": False, "has_specific_role": False, "role": None}

    def assess_resume_focus_gate(self, message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """Decide if we need role/stack/domain before sending a resume file."""
        prior = history or []
        recent_lines: List[str] = []
        for item in prior[-10:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2800]
        if not conversation.strip():
            return {"needs_clarification": True}

        system_prompt = (
            "The user is interacting about Jayanth's resume/CV file download. "
            "Return strict JSON: {\"needs_clarification\": boolean, \"confidence\": number}. "
            "Set needs_clarification=true when the user wants a resume but has NOT indicated what they need it for "
            "(e.g. role, domain, stack: ML vs AI engineering vs software development vs computer vision vs data, "
            "or seniority/team context). "
            "Set needs_clarification=false when they already state a clear focus in the current or prior messages "
            "(examples: 'MLE resume', 'AI engineer CV', 'computer vision', 'full-stack', 'general SWE', "
            "'RAG/LLM role', 'the quant role'). "
            "If the assistant already asked what kind of resume they need and the user is now answering with any "
            "substantive focus (even short), set needs_clarification=false. "
            "If they only say 'resume', 'CV', 'send pdf' with no domain, set needs_clarification=true."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.1,
                max_tokens=80,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return {
                "needs_clarification": bool(parsed.get("needs_clarification", True)),
                "confidence": float(parsed.get("confidence", 0.0)),
            }
        except Exception:
            return {"needs_clarification": True, "confidence": 0.0}

    def is_general_profile_question_not_resume_file(self, message: str) -> bool:
        """True when the user is asking for profile/explanation content, not to obtain a résumé file."""
        msg = (message or "").strip().lower()
        if not msg:
            return False
        resume_lex = ("resume", "cv", "pdf", "download", "curriculum", "attachment")
        if any(w in msg for w in resume_lex):
            return False
        pivot_phrases = (
            "work experience",
            "experience",
            "explain",
            "tell me about",
            "what did he",
            "what does he",
            "describe",
            "background",
            "career",
            "projects",
            "research",
            "skill",
            "currently",
            "where does he",
            "his work",
            "his job",
            "his role",
            "his role at",
            "day to day",
            "what is he",
            "what he's",
            "elaborate",
            "more about him",
            "about his",
        )
        return any(p in msg for p in pivot_phrases)

    def should_continue_resume_file_flow(
        self, message: str, history: List[Dict[str, str]] | None = None
    ) -> bool:
        """True when intent was information_request but user is clearly continuing the resume-download thread."""
        prior = history or []
        if len(prior) < 2:
            return False
        if self.is_general_profile_question_not_resume_file(message):
            return False

        last_bot = ""
        for item in reversed(prior):
            if item.get("role") == "assistant":
                last_bot = str(item.get("content", "")).lower()
                break
        # Last turn must actually be about delivering or scoping a résumé/CV (not generic ML/experience copy).
        if not (
            re.search(r"\b(resume|curriculum)\b", last_bot)
            or re.search(r"\bcv\b", last_bot)
            or "download" in last_bot
            or "link in this chat" in last_bot
        ):
            return False

        system_prompt = (
            "Return strict JSON: {\"continue_resume\": boolean}. "
            "The previous assistant message was about Jayanth's resume/CV file (download, link, or which resume variant to send). "
            "continue_resume=true ONLY if the user's latest message still belongs to THAT thread: "
            "clarifying which résumé file, role label for the CV, format, resending the link, or a short tag like MLE/AI engineer/CV for X. "
            "continue_resume=false if the user moved on to general questions about Jayanth's work history, experience, projects, skills, "
            "current job, day-to-day responsibilities, research, or anything they want explained in chat (not about obtaining the file). "
            "continue_resume=false for small talk unrelated to the file. When unsure, prefer false."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Last assistant message (truncated): {last_bot[:1200]}\n\nUser: {message.strip()}"},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return bool(parsed.get("continue_resume", False))
        except Exception:
            return False

    def verify_resume_entry_matches_need(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
        entry: Dict[str, Any],
    ) -> bool:
        """True if the chosen resume entry can reasonably match what the user asked for."""
        prior = history or []
        recent_lines: List[str] = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2400]

        tags = entry.get("tags") or []
        file_name = Path(entry.get("relative_path", "")).name
        payload = {"tags": tags, "file_name": file_name}

        system_prompt = (
            "Decide if the available resume option can reasonably match the user's stated need. "
            "Return strict JSON: {\"adequate\": boolean}. "
            "adequate=true when the tags/domain align with what the user asked for (e.g. MLE tags for an ML role). "
            "adequate=false when the user asks for a specialization or niche that these tags do not plausibly cover "
            "(e.g. they ask for pure blockchain, quantum, or hardware roles and tags are only ML/AI/SWE without overlap). "
            "When unsure, prefer adequate=true if there is partial overlap."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.0,
                max_tokens=60,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Conversation:\n{conversation}\n\nResume option (JSON):\n{json.dumps(payload)}",
                    },
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            return bool(parsed.get("adequate", True))
        except Exception:
            return True

    def get_resume_entries(self) -> List[Dict[str, Any]]:
        if self._resume_entries_cache is None:
            self._resume_entries_cache = load_resume_entries(self.repo_root)
        return self._resume_entries_cache

    def select_resume_entry(
        self, message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """Pick the best on-disk resume entry for this conversation (manifest tags + LLM choice)."""
        entries = self.get_resume_entries()
        if not entries:
            return None
        if len(entries) == 1:
            path = self.repo_root / entries[0]["relative_path"]
            if path.is_file():
                return path.resolve(), entries[0]
            return None

        hiring = self.assess_hiring_query(message, history)
        role_hint = ""
        if hiring.get("role"):
            role_hint = f"Stated or implied role/title: {hiring['role']}\n"

        prior = history or []
        recent_lines: List[str] = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2400]

        catalog = [
            {
                "id": e["id"],
                "tags": e.get("tags", []),
                "file_name": Path(e["relative_path"]).name,
            }
            for e in entries
        ]

        system_prompt = (
            "You select which resume variant best fits the user's request. "
            "Each option has an id, tags (topics/roles), and file_name only. "
            "Match the conversation to the best option (e.g., MLE vs AI Engineer vs general). "
            "If unclear, choose the strongest general-purpose option (usually the first entry). "
            "Return strict JSON only: {\"choice_id\": string, \"confidence\": number}."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.1,
                max_tokens=120,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{role_hint}"
                            f"Conversation:\n{conversation}\n\n"
                            f"Resume options (JSON):\n{json.dumps(catalog)}"
                        ),
                    },
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            choice = str(parsed.get("choice_id", "")).strip()
            for entry in entries:
                if entry["id"] == choice:
                    path = self.repo_root / entry["relative_path"]
                    if path.is_file():
                        return path.resolve(), entry
        except Exception:
            pass

        fallback = self.repo_root / entries[0]["relative_path"]
        if fallback.is_file():
            return fallback.resolve(), entries[0]
        return None

    def select_resume_path(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[Path]:
        """Pick the best on-disk resume path (convenience wrapper)."""
        result = self.select_resume_entry(message, history)
        return result[0] if result else None

    def resume_focus_clarification_reply(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        return self.intent_reply("resume_focus_clarification", message, history)

    def resume_no_fit_connect_reply(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        return self.intent_reply("resume_no_fit_connect", message, history)

    def resume_attachment_reply(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        *,
        skip_connect_offer: bool = False,
    ) -> str:
        """Short reply when serving a resume download."""
        return self.intent_reply(
            "resume_attachment",
            message,
            history,
            resume_skip_connect_offer=skip_connect_offer,
        )

    def extract_connect_details(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Optional[str]]:
        """LLM-only extraction for meeting/connect details."""
        prior = history or []
        recent_lines = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            # Use only user lines to avoid assistant/Jayanth mentions polluting requester identity.
            if role == "user" and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:3200]

        system_prompt = (
            "You extract meeting scheduling details from a conversation. "
            "Return strict JSON only with exactly these keys: "
            "{\"name\": string|null, \"profession\": string|null, \"email\": string|null, \"preferred_time\": string|null}. "
            "Rules: "
            "1) Do not hallucinate values. If unknown, return null. "
            "2) Name must be a real person name for the requester, not job phrases (e.g. 'hiring for an ML Engineer' is not a name). "
            "2b) The portfolio owner is Jayanth — do not set name from phrases like 'connect me with Jayanth' alone. "
            "If the visitor explicitly introduces themselves (e.g. 'I'm Jayanth', 'name is Jayanth'), use that as name even if it matches the owner. "
            "3) Profession should describe the requester role only when clearly stated. "
            "4) Email must be explicit. "
            "5) preferred_time should capture explicit availability only when present; use US Eastern (EST/ET) when giving times, "
            "since Jayanth schedules from New Jersey (Eastern)."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0,
                max_tokens=220,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            result = {
                "name": parsed.get("name"),
                "profession": parsed.get("profession"),
                "email": parsed.get("email"),
                "preferred_time": parsed.get("preferred_time"),
            }
            normalized: Dict[str, Optional[str]] = {}
            for key, value in result.items():
                if value is None:
                    normalized[key] = None
                    continue
                text = str(value).strip()
                normalized[key] = text if text else None
            return normalized
        except Exception:
            # Fail-safe: ask user for details explicitly.
            return {"name": None, "profession": None, "email": None, "preferred_time": None}

    def extract_meeting_change_action(
        self, message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Optional[str]]:
        """Parse cancel vs reschedule and optional note / availability."""
        prior = history or []
        recent_lines = []
        for item in prior[-10:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:3600]

        system_prompt = (
            "The user may want to cancel or reschedule a meeting with Jayanth that was set up through this chat. "
            "Return strict JSON only with keys: "
            "{\"action\": \"cancel\"|\"reschedule\"|\"unclear\", "
            "\"note\": string|null, \"new_availability\": string|null}. "
            "Rules: "
            "action=cancel if they want to call off, withdraw, cancel, or cannot attend. "
            "action=reschedule if they want a different time, move the meeting, or need another slot (still want to meet). "
            "action=unclear if you cannot tell. "
            "note = short message to pass to Jayanth (apology, reason) when present. "
            "new_availability = preferred windows or times for reschedule when stated; else null. "
            "Do not invent details."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            action = str(parsed.get("action") or "unclear").strip().lower()
            if action not in ("cancel", "reschedule", "unclear"):
                action = "unclear"
            note = parsed.get("note")
            navail = parsed.get("new_availability")
            return {
                "action": action,
                "note": str(note).strip() if note else None,
                "new_availability": str(navail).strip() if navail else None,
            }
        except Exception:
            return {"action": "unclear", "note": None, "new_availability": None}

    @staticmethod
    def _looks_like_insufficient_context(reply: str) -> bool:
        text = (reply or "").strip().lower()
        markers = (
            "do not have enough verified information",
            "don't have enough verified information",
            "not enough verified information",
            "not enough information in the provided context",
            "provided context",
            "cannot answer from the context",
            "insufficient context",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _last_assistant_content(history: List[Dict[str, str]] | None) -> str:
        if not history:
            return ""
        for item in reversed(history):
            if item.get("role") == "assistant":
                return (item.get("content") or "").strip()
        return ""

    def _is_short_affirmation(self, message: str) -> bool:
        raw = (message or "").strip().lower()
        if not raw or len(raw) > 48:
            return False
        normalized = re.sub(r"[\s,;:'\"]+", " ", raw)
        normalized = re.sub(r"^[.!?]+|[.!?]+$", "", normalized).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return False
        allow = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "please",
            "pls",
            "absolutely",
            "definitely",
            "indeed",
            "of",
            "course",
            "sounds",
            "good",
            "go",
            "ahead",
            "do",
            "tell",
            "me",
            "more",
            "kinda",
            "kind",
            "right",
        }
        phrases = {
            "yes please",
            "yes pls",
            "sounds good",
            "go ahead",
            "please do",
            "of course",
            "sure thing",
            "ok sure",
            "okay sure",
            "yes sure",
            "tell me more",
        }
        if normalized in phrases:
            return True
        words = normalized.split()
        if words and all(w in allow for w in words):
            return True
        return False

    def _last_substantive_user_question(self, history: List[Dict[str, str]] | None) -> str:
        if not history:
            return ""
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            content = (item.get("content") or "").strip()
            if not content or self._is_short_affirmation(content):
                continue
            return content
        return ""

    def _assistant_offered_scheduling_or_connect(self, assistant_text: str) -> bool:
        t = (assistant_text or "").lower()
        if not t:
            return False
        markers = (
            "schedule",
            "scheduling",
            "quick call",
            "set up a call",
            "set up a meeting",
            "book a",
            "meeting with",
            "call with jayanth",
            "connect you with jayanth",
            "connect you to jayanth",
            "time slot",
            "time works best",
            "arrange a",
            "would you like me to schedule",
            "help you connect",
            "help schedule",
            "loop jayanth",
            "set up a quick",
        )
        return any(m in t for m in markers)

    def should_treat_as_information_followup(
        self, message: str, history: List[Dict[str, str]] | None = None
    ) -> bool:
        """Route short affirmations back to RAG when the bot did not offer scheduling/contact for a call."""
        if not self._is_short_affirmation(message):
            return False
        last_a = self._last_assistant_content(history)
        if not last_a:
            return False
        if self._assistant_offered_scheduling_or_connect(last_a):
            return False
        return True

    def _retrieval_query(self, message: str, history: List[Dict[str, str]] | None) -> str:
        m = (message or "").strip()
        if not m:
            return m
        if self._is_short_affirmation(m) and history:
            parts: List[str] = [m]
            prev_u = self._last_substantive_user_question(history)
            if prev_u:
                p = prev_u.replace("\n", " ").strip()
                parts.append(f"Earlier user question: {p[:280]}")
            la = self._last_assistant_content(history)
            if la:
                snippet = la.replace("\n", " ").strip()
                if len(snippet) > 320:
                    snippet = snippet[:317] + "..."
                parts.append(f"Prior assistant: {snippet}")
            return "\n".join(parts)
        return m

    def _history_for_classifier(self, message: str, history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
        prior = list(history or [])
        msg = (message or "").strip()
        if prior and prior[-1].get("role") == "user" and (prior[-1].get("content") or "").strip() == msg:
            return prior[:-1]
        return prior

    def classify_intent(self, message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """LLM-based intent classifier for routing chat behavior."""
        prior = self._history_for_classifier(message, history)
        recent_lines = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2600]
        if not conversation:
            return {"intent": "information_request", "confidence": 0.0}

        system_prompt = (
            "You are an intent classifier for a personal portfolio chatbot. "
            "Classify the user intent into exactly one label from this set: "
            "[greeting, information_request, resume_request, connect_request, connect_confirmation, meeting_change, out_of_scope, abusive]. "
            "Definitions: "
            "greeting = hello/hi OR bot-identity/capability questions like 'who are you', 'what are you', "
            "'what can you do', 'what can I ask you' ONLY when there is no additional substantive request. "
            "If a message includes both a greeting and a real request (for example hiring/recruiting, profile, skills, or experience ask), "
            "do NOT label as greeting; label by the main request intent. "
            "resume_request = user wants Jayanth's resume/CV as a file (PDF/DOCX download, send me your resume, attach CV, can I get a copy). "
            "If they only want a summary of experience without asking for a file, use information_request instead. "
            "information_request = asks about Jayanth's profile only (experience, skills, projects, research, education, contact); "
            "hiring/recruiting role-fit questions (e.g., 'I'm hiring for X role, is he a fit?') should be information_request. "
            "connect_request = asks to connect/talk/schedule with Jayanth, OR asks for information beyond what the bot knows from profile docs "
            "in a way that implies they want direct interaction with Jayanth (for example: 'I want to know more than what you know about Jayanth', "
            "'I want to know Jayanth personally', 'can I speak to Jayanth directly', 'how can I connect with him', "
            "'interesting, tell me more about him' after a profile summary answer); "
            "connect_confirmation = ONLY when the immediately prior assistant message explicitly offered to schedule a call/meeting "
            "or asked yes/no about connecting for a call (e.g. proposed times, meeting setup). "
            "Short replies like 'yes/sure/ok' after purely informational follow-ups (examples, skills, projects, work detail, "
            "'want more on his experience') MUST be information_request — not connect_confirmation. "
            "Bare 'yes' that continues a profile Q&A thread is information_request unless the prior assistant clearly proposed scheduling. "
            "meeting_change = user wants to cancel, reschedule, move, or withdraw from a meeting/call they scheduled with Jayanth, "
            "or add a note about cancellation/changing plans (e.g. 'cancel my call', 'I need a different time', 'sorry I can't make it'); "
            "out_of_scope = asks anything not grounded in Jayanth's profile docs, including general knowledge, politics/news, opinions on external events, "
            "or personal/private topics not explicitly in profile context; "
            "abusive = cuss words, hate, harassment, indecent or explicit abusive language. "
            "Return strict JSON only: {\"intent\":\"...\", \"confidence\":0..1}."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.2,
                max_tokens=70,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "")).strip().lower()
            confidence = float(parsed.get("confidence", 0))
            allowed = {
                "greeting",
                "information_request",
                "resume_request",
                "connect_request",
                "connect_confirmation",
                "meeting_change",
                "out_of_scope",
                "abusive",
            }
            if intent not in allowed:
                intent = "information_request"
            return {"intent": intent, "confidence": confidence}
        except Exception:
            # Fail-safe: continue with normal RAG response path.
            return {"intent": "information_request", "confidence": 0.0}

    def is_profile_related(self, message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """LLM check: is user asking about Jayanth profile/expertise context?"""
        prior = self._history_for_classifier(message, history)
        recent_lines = []
        for item in prior[-8:]:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                recent_lines.append(f"{role}: {content}")
        recent_lines.append(f"user: {message.strip()}")
        conversation = "\n".join(recent_lines)[:2600]
        if not conversation:
            return {"profile_related": False, "confidence": 0.0}

        system_prompt = (
            "Decide if the user query is about Jayanth's profile domain. "
            "Return strict JSON only: {\"profile_related\": true|false, \"confidence\": 0..1}. "
            "Treat as profile_related when the user asks about Jayanth's experience, skills, AI/ML background, "
            "projects, research, education, work exposure, tools, strengths, fit for roles, or contact/connecting. "
            "Also treat follow-up references like 'tell me more about him' as profile_related if prior context is about Jayanth. "
            "Treat as not profile_related for politics/news/general world topics not about Jayanth profile."
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.intent_model,
                temperature=0.0,
                max_tokens=60,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            related = bool(parsed.get("profile_related", False))
            confidence = float(parsed.get("confidence", 0.0))
            return {"profile_related": related, "confidence": confidence}
        except Exception:
            return {"profile_related": False, "confidence": 0.0}

    @property
    def is_ready(self) -> bool:
        return bool(self.index_data.get("chunks"))

    def _ensure_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it before using /api/reindex or /api/chat.")

    def _discover_documents(self) -> List[Path]:
        docs: List[Path] = []
        all_resumes: List[Path] = []
        resume_dir = self.repo_root / "Resume"
        if resume_dir.exists():
            for p in resume_dir.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if p.name.lower() == "resume_manifest.json":
                    continue
                all_resumes.append(p)

        all_resumes.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        limit = self.resume_limit
        if limit <= 0:
            selected_resumes = all_resumes
        else:
            selected_resumes = all_resumes[: max(1, limit)]
        docs.extend(selected_resumes)

        # Root-level PDFs/DOCX/etc. (resume, CV, research paper). Previously only filenames
        # containing "paper" were indexed, so production missed e.g. jayanth__resume.pdf while
        # still picking up Research_paper.pdf — skewing "current role" answers toward old research.
        root_docs: List[Path] = []
        for p in self.repo_root.iterdir():
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            root_docs.append(p)

        docs.extend(sorted(root_docs, key=lambda path: path.stat().st_mtime, reverse=True))

        # Deduplicate while keeping stable order
        unique_docs: List[Path] = []
        seen = set()
        for doc in docs:
            key = str(doc.resolve())
            if key not in seen:
                unique_docs.append(doc)
                seen.add(key)
        return unique_docs

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            return "\n\n".join(p for p in pages if p)
        if suffix == ".docx":
            doc = Document(str(path))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _normalize(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        filtered = [line for line in lines if line]
        return "\n".join(filtered).strip()

    def _recursive_split(self, text: str, separators: List[str], size: int) -> List[str]:
        if len(text) <= size:
            return [text]
        if not separators:
            return [text[i : i + size] for i in range(0, len(text), size)]

        separator = separators[0]
        if separator and separator in text:
            parts = text.split(separator)
            merged: List[str] = []
            current = ""
            for part in parts:
                candidate = part if not current else f"{current}{separator}{part}"
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        merged.append(current)
                    current = part
            if current:
                merged.append(current)
        else:
            merged = [text]

        results: List[str] = []
        for part in merged:
            if len(part) <= size:
                results.append(part)
            else:
                results.extend(self._recursive_split(part, separators[1:], size))
        return results

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        if not chunks or overlap <= 0:
            return chunks
        overlapped: List[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev_tail = chunks[idx - 1][-overlap:]
            merged = f"{prev_tail}\n{chunk}".strip()
            overlapped.append(merged)
        return overlapped

    def _chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        base_chunks = self._recursive_split(normalized, DEFAULT_SEPARATORS, self.chunk_size)
        final_chunks = self._apply_overlap(base_chunks, self.chunk_overlap)
        return [Chunk(text=c, metadata={**metadata, "chunk_index": i}) for i, c in enumerate(final_chunks) if c]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.embed_model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def build_index(self) -> Dict[str, Any]:
        self._ensure_api_key()
        documents = self._discover_documents()
        chunk_objects: List[Chunk] = []

        for path in documents:
            text = self._extract_text(path)
            doc_type = "research" if "paper" in path.name.lower() else "resume"
            rel = str(path.relative_to(self.repo_root))
            chunk_objects.extend(
                self._chunk_document(
                    text,
                    {
                        "source_file": rel,
                        "doc_type": doc_type,
                        "modified_at": int(path.stat().st_mtime),
                    },
                )
            )

        if not chunk_objects:
            raise RuntimeError("No supported documents found to index.")

        texts = [c.text for c in chunk_objects]
        vectors = self._embed_texts(texts)
        records = []
        for i, (chunk, embedding) in enumerate(zip(chunk_objects, vectors)):
            records.append(
                {
                    "id": f"chunk-{i}",
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

        payload = {
            "config": {
                "embed_model": self.embed_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "stats": {"documents": len(documents), "chunks": len(records)},
            "chunks": records,
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload), encoding="utf-8")
        self.index_data = payload
        return payload["stats"]

    def load_index(self) -> None:
        if self.index_path.exists():
            self.index_data = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index_data = {"chunks": []}

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        is_current_experience_query = any(
            token in query_lower for token in ("current", "currently", "work experience", "working on", "present role")
        )
        is_experience_overview_query = any(
            token in query_lower
            for token in (
                "work experience",
                "experience",
                "professional experience",
                "career",
                "background",
            )
        )
        query_vec = self.client.embeddings.create(model=self.embed_model, input=[query]).data[0].embedding
        scored = []
        for chunk in self.index_data.get("chunks", []):
            sim = self._cosine_similarity(query_vec, chunk["embedding"])
            bonus = 0.0
            meta = chunk.get("metadata") or {}
            doc_type = (meta.get("doc_type") or "").lower()
            if is_current_experience_query:
                # Prefer CV/resume PDFs over academic research for "current role" questions.
                if doc_type == "resume":
                    bonus += 0.14
                elif doc_type == "research":
                    bonus -= 0.18
                text = chunk["text"].lower()
                if any(
                    token in text
                    for token in (
                        "present",
                        "current",
                        "currently",
                        "aug 2024",
                        "fedway",
                        "eliteus",
                        "elite us",
                        "piscataway",
                        "ai engineer",
                        "applied ai",
                    )
                ):
                    bonus += 0.08
            if is_experience_overview_query:
                text = chunk["text"].lower()
                if any(
                    token in text
                    for token in (
                        "eliteus",
                        "fedway",
                        "harvest software",
                        "accenture",
                        "research assistant",
                        "self driving",
                        "andhra university",
                    )
                ):
                    bonus += 0.12
            scored.append({**chunk, "score": sim + bonus})
        scored.sort(key=lambda item: item["score"], reverse=True)
        top_n = self.top_k + 3 if is_experience_overview_query else self.top_k
        return scored[: top_n]

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]] | None = None,
        allow_low_relevance: bool = False,
        *,
        skip_meeting_scheduling_prompt: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_api_key()
        if not self.is_ready:
            raise RuntimeError("Index is empty. Run /api/reindex first.")

        retrieve_query = self._retrieval_query(message, history)
        retrieved = self._retrieve(retrieve_query)
        if not retrieved:
            return {
                "reply": self.intent_reply(
                    "out_of_scope",
                    message,
                    history,
                    out_of_scope_skip_schedule_offer=skip_meeting_scheduling_prompt,
                ),
                "sources": [],
            }
        if (not allow_low_relevance) and retrieved[0].get("score", 0.0) < self.min_relevance_score:
            return {
                "reply": self.intent_reply(
                    "out_of_scope",
                    message,
                    history,
                    out_of_scope_skip_schedule_offer=skip_meeting_scheduling_prompt,
                ),
                "sources": [],
            }

        context_blocks = []
        sources = []
        for item in retrieved:
            src = item["metadata"].get("source_file", "unknown")
            sources.append(src)
            context_blocks.append(f"[Source: {src}]\n{item['text']}")
        context_text = "\n\n---\n\n".join(context_blocks)
        is_experience_overview_query = any(
            token in (message or "").lower()
            for token in ("work experience", "professional experience", "career", "background")
        )

        system_prompt = (
            "You are Ada, Jayanth's AI assistant. Speak about Jayanth in third person. "
            "Tone: friendly, conversational, and charismatic—warm, human, and engaging; never stiff, robotic, or scripted. "
            "STRICT: Use the conversation history. Answer the user's latest message in context—reference what they asked, "
            "correct misunderstandings, and build on what they already said. Do not give a generic answer that ignores the thread. "
            "STRICT: Never repeat the same introduction, same sentences, or same closing question you used in a previous turn. "
            "Each reply must feel freshly written for this moment. Vary phrasing; do not loop canned paragraphs. "
            "Use emojis naturally (at least 1, at most 5). "
            "Speak highly of Jayanth with confident, positive language grounded in context. "
            "Only use the provided context. If the answer is not in context, say you do not have enough verified information. "
            "When asked about current work experience, prioritize roles explicitly marked as current/present and avoid presenting older internships as current. "
            "When discussing work experience, impact, or achievements, include concrete numbers when the context supports them "
            "(e.g. years, percentages, data scale, team size, throughput, users, revenue ranges if stated). "
            "Prefer figures taken directly from the retrieved context; do not invent precise statistics that are not grounded there. "
            "You may describe impact vividly when the resume implies scale, but stay faithful to what is written. "
            "Do not mention exact date phrases like 'since Aug 2024' unless the user explicitly asks for dates or timeline. "
            "Write in readable paragraph format with exactly 2 paragraphs, not bullet points or fragmented lines. "
            "Each paragraph should be 1-3 short lines/sentences. "
            "Keep answers crisp and conversational, roughly 45-90 words, with high-impact phrasing. "
            "Always finish with complete sentences; never leave the response cut off. "
            "Use blank lines between paragraphs. "
            "End every response with one natural follow-up question that fits this specific exchange—not a stock question. "
            "STRICT: The last sentence of your reply must be that question and it must end with ? (not a period). "
            "STRICT: The closing question is FROM Ada TO the visitor. Ada holds Jayanth's profile—you are not interviewing the visitor for facts. "
            "Never ask the visitor to supply, provide, explain, or spell out Jayanth's roles, achievements, or résumé details "
            "(e.g. forbid: 'Can you provide more details about Jayanth's role at…?', 'Could you tell/share more about his work at…?'). "
            "Instead invite what they want to hear next from Ada, e.g. 'Want me to unpack his Harvest analytics work a bit more?', "
            "'Curious how that scaled in production?', 'Should we zoom in on his stack or team impact there?'. "
            "Do not trail off with only factual statements; weave the question as the true closing line. "
            "Never default to the same closing prompt every turn (avoid repeating generic 'want an example' wording unless the user clearly asked for examples). "
            "Do not sound like customer support; avoid corporate filler and generic lines like 'How may I assist you today?'. "
            "If the user appears to be hiring/recruiting (explicitly or implicitly), provide a persuasive, confident pitch that "
            "positions Jayanth as the strongest smart choice for the role using concrete evidence from context. "
            "If hiring/recruiting is implied but the role is not explicitly provided, ask a clarifying role question first before giving the pitch. "
            "For hiring-fit answers, prefer this structure: short impact opener paragraph, one proof paragraph, then one final question line."
        )
        if is_experience_overview_query:
            system_prompt += (
                " STRICT FORMAT FOR EXPERIENCE OVERVIEW: produce exactly 8 concise lines in this order: "
                "lines 1-3 about current Fedway/EliteUS work (technical and quantified), "
                "lines 4-5 about Harvest (quantified impact), "
                "lines 6-7 about Accenture (quantified impact), "
                "line 8 about research assistant/research paper work (quantified where available). "
                "After those 8 lines, add one final line that starts exactly with 'Key skills:' and list skills drawn from experience context only. "
                "Use concrete numbers/percentages/counts/ranges whenever context provides them. "
                "Do not invent exact metrics not present in context; if a metric is implied but not explicit, use a cautious range phrase."
            )
        if skip_meeting_scheduling_prompt:
            system_prompt += (
                " STRICT: The user already submitted a connect or meeting request earlier in this conversation. "
                "Do NOT ask whether they want to schedule a call, book a session, or arrange a meeting with Jayanth. "
                "Do not invite them to set up another call. End with a different follow-up question about skills, projects, or content."
            )
        intro_prompt = (
            "This is the first message in the thread. Begin with one short natural line introducing yourself as Ada, "
            "Jayanth's AI assistant (paraphrase freely—do not use a fixed script), then answer using the context. "
            "Do not copy a long boilerplate intro; keep it tight and specific to their question."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        prior = history or []
        for item in prior[-8:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"User question: {message}\n"
            "Answer using only the context."
        )
        if not prior:
            user_prompt = f"{intro_prompt}\n\n{user_prompt}"

        messages.append({"role": "user", "content": user_prompt})

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=_safe_float(os.getenv("RAG_TEMPERATURE"), 0.2),
            max_tokens=_safe_int(os.getenv("RAG_MAX_TOKENS"), 150),
            messages=messages,
        )
        reply = completion.choices[0].message.content or "I do not have enough verified information to answer that."
        reply = self._finalize_reply(reply)
        reply = self._ensure_two_paragraphs(reply)
        reply = self._ensure_information_reply_ends_with_question(
            reply,
            message,
            history,
            skip_meeting_scheduling_prompt=skip_meeting_scheduling_prompt,
        )
        reply = self._repoint_inverted_profile_closing_question(reply)
        reply = self._ensure_emoji(reply, intent="information_request")
        if self._looks_like_insufficient_context(reply):
            return {
                "reply": self.intent_reply(
                    "out_of_scope",
                    message,
                    history,
                    out_of_scope_skip_schedule_offer=skip_meeting_scheduling_prompt,
                ),
                "sources": [],
            }
        return {"reply": reply, "sources": sorted(set(sources))}
