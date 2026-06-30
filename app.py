"""
Workspace AI — Streamlit frontend
Wired to:
  POST /auth/register
  POST /auth/login    → access_token, refresh_token
  GET  /auth/me       → id, email, username, full_name, department, role
  POST /chat/query    → answer, sources, latency_ms, was_blocked, quality
  GET  /chat/history  → total, queries[]
"""

import os
import html

import requests
import streamlit as st


API_BASE = os.getenv("ROLE_RAG_API_BASE", "http://localhost:8001").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Workspace AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
# Refined, friendlier blue palette — softer surfaces, slightly larger radii,
# gentle shadows, clearer hierarchy. Still clean & minimal, just warmer.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ws-blue:        #1a73e8;
    --ws-blue-dark:   #1558b3;
    --ws-blue-light:  #eaf2fe;
    --ws-blue-lighter:#f4f8ff;
    --ws-border:      #e3e6ec;
    --ws-bg:          #fafbfd;
    --ws-text:        #14181f;
    --ws-text-sub:    #6b7280;
    --ws-text-faint:  #a3a9b4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.75rem; padding-bottom: 2rem; max-width: 580px; }
body { background: var(--ws-bg); }

/* ── Brand ── */
.ws-brand {
    display: flex; align-items: center; gap: 10px; margin-bottom: 1.75rem;
}
.ws-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--ws-blue), var(--ws-blue-dark));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
    box-shadow: 0 2px 6px rgba(26,115,232,0.25);
}
.ws-brand-name { font-size: 16.5px; font-weight: 700; color: var(--ws-text); letter-spacing: -0.2px; }

/* ── Typography ── */
.ws-h1 { font-size: 22px; font-weight: 700; color: var(--ws-text); margin-bottom: 4px; letter-spacing: -0.3px; }
.ws-sub { font-size: 14px; color: var(--ws-text-sub); margin-bottom: 1.5rem; }

/* ── Trust row ── */
.ws-trust {
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; color: var(--ws-text-sub); margin-bottom: 9px;
}
.ws-trust-dot {
    width: 30px; height: 30px; background: var(--ws-blue-light); border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}

/* ── Greeting ── */
.ws-greeting {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font-size: 15.5px; font-weight: 600; color: var(--ws-text); margin-bottom: 1.25rem;
}
.ws-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--ws-blue-light); color: var(--ws-blue-dark);
    font-size: 12px; font-weight: 600; padding: 4px 11px;
    border-radius: 20px;
}

/* ── Chat bubbles ── */
.ws-msg-wrap-user { display: flex; justify-content: flex-end; margin-bottom: 4px; }
.ws-msg-wrap-bot  { display: flex; justify-content: flex-start; margin-bottom: 4px; }
.ws-bubble-user {
    background: linear-gradient(135deg, var(--ws-blue), var(--ws-blue-dark));
    color: #fff;
    border-radius: 16px 16px 3px 16px;
    padding: 11px 15px; font-size: 14px; line-height: 1.6; max-width: 86%;
    box-shadow: 0 2px 8px rgba(26,115,232,0.18);
}
.ws-bubble-bot {
    background: #ffffff; color: var(--ws-text);
    border: 1px solid var(--ws-border);
    border-radius: 16px 16px 16px 3px;
    padding: 11px 15px; font-size: 14px; line-height: 1.6; max-width: 90%;
    box-shadow: 0 1px 3px rgba(20,24,31,0.04);
}
.ws-bubble-blocked {
    background: #fff8e6; color: #7a4f00;
    border: 1px solid rgba(200,150,0,0.18);
    border-radius: 16px; padding: 11px 15px;
    font-size: 13.5px; line-height: 1.6; max-width: 90%;
}
.ws-lbl { font-size: 11.5px; color: var(--ws-text-faint); margin-bottom: 4px; font-weight: 500; }
.ws-lbl-r { font-size: 11.5px; color: var(--ws-text-faint); margin-bottom: 4px; text-align: right; font-weight: 500; }
.ws-source {
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid var(--ws-border);
    font-size: 11.5px; color: var(--ws-text-sub);
}
.ws-gap { height: 14px; }
.ws-divider { border: none; border-top: 1px solid var(--ws-border); margin: 1rem 0; }

/* ── Score box ── */
.ws-scorebox {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--ws-border);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px 14px;
    font-size: 11.5px;
    color: var(--ws-text-sub);
}
.ws-scorebox b { color: var(--ws-text); font-weight: 600; }
.ws-score-pill {
    display: inline-flex; align-items: center;
    background: var(--ws-blue-lighter); color: var(--ws-blue-dark);
    border-radius: 6px; padding: 1px 6px; font-weight: 600; margin-left: 4px;
}

/* ── Suggestion chips ── */
.ws-chips { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 1rem; }
.ws-chip {
    padding: 6px 12px; background: #fff;
    border: 1px solid var(--ws-border); border-radius: 20px;
    font-size: 12.5px; color: var(--ws-text-sub); font-weight: 500;
}

/* ── History card ── */
.ws-hist-card {
    background: #fff; border: 1px solid var(--ws-border);
    border-radius: 14px; padding: 15px 17px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(20,24,31,0.03);
}
.ws-hist-q { font-size: 14px; font-weight: 600; color: var(--ws-text); margin-bottom: 5px; }
.ws-hist-a { font-size: 13.5px; color: #444; line-height: 1.55; margin-bottom: 7px; }
.ws-hist-meta { font-size: 11.5px; color: var(--ws-text-faint); }
.ws-badge-ok  { display:inline-flex;padding:2px 9px;background:#e6f4ea;color:#137333;font-size:11px;font-weight:600;border-radius:20px; }
.ws-badge-blk { display:inline-flex;padding:2px 9px;background:#fff8e6;color:#7a4f00;font-size:11px;font-weight:600;border-radius:20px; }

/* ── Input & button overrides ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border: 1px solid var(--ws-border) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 11px 15px !important;
    background: #fff !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--ws-blue) !important;
    box-shadow: 0 0 0 3px var(--ws-blue-light) !important;
}
div[data-testid="stButton"] > button {
    width: 100%;
    background: #fff !important; color: var(--ws-text) !important;
    border: 1px solid var(--ws-border) !important;
    border-radius: 12px !important;
    font-size: 14px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 16px !important; transition: all 0.13s !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--ws-blue-lighter) !important;
    border-color: var(--ws-blue) !important;
    color: var(--ws-blue-dark) !important;
}
/* Primary "Send" button gets a solid blue treatment */
div[data-testid="stButton"][key="send"] > button,
div[data-testid="column"]:has(button:contains("Send")) button {
    background: var(--ws-blue) !important;
    color: #fff !important;
    border-color: var(--ws-blue) !important;
}
div[data-testid="stAlert"] {
    border-radius: 12px !important; font-size: 13.5px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "access_token":  None,
        "refresh_token": None,
        "user":          None,   # dict from /auth/me
        "chat_history":  [],     # list of {question, answer, sources, quality, latency_ms, was_blocked}
        "page":          "chat", # chat | history
        "auth_tab":      "login",# login | register
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── API helpers ───────────────────────────────────────────────────────────────
def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.access_token}"}

def api_register(email, username, password, full_name, department):
    try:
        r = requests.post(f"{API_BASE}/auth/register", json={
            "email": email, "username": username,
            "password": password, "full_name": full_name,
            "department": department, "role": "viewer",
        }, timeout=10)
        return r.status_code == 201, r.json()
    except Exception as e:
        return False, {"detail": str(e)}

def api_login(email, password):
    try:
        r = requests.post(f"{API_BASE}/auth/login",
                          json={"email": email, "password": password}, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json()
    except Exception as e:
        return False, {"detail": str(e)}

def api_me(token):
    try:
        r = requests.get(f"{API_BASE}/auth/me",
                         headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, {}
    except Exception as e:
        return False, {}

def api_query(question, top_k=5):
    try:
        r = requests.post(f"{API_BASE}/chat/query",
                          headers=auth_headers(),
                          json={"question": question, "top_k": top_k},
                          timeout=45)
        if r.status_code == 200:
            return True, r.json()
        return False, {"answer": "Something went wrong. Please try again.", "sources": [], "quality": None, "was_blocked": False, "latency_ms": 0}
    except Exception as e:
        return False, {"answer": "Could not reach the server. Please check your connection.", "sources": [], "quality": None, "was_blocked": False, "latency_ms": 0}

def api_history():
    try:
        r = requests.get(f"{API_BASE}/chat/history",
                         headers=auth_headers(), timeout=15)
        if r.status_code == 200:
            return r.json().get("queries", [])
        return []
    except Exception:
        return []


# ── Score formatting helpers ───────────────────────────────────────────────────
def format_score(value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        # RAGAS-style scores are 0–1, show as a percentage.
        if 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        return f"{value:.3f}"
    except Exception:
        return "N/A"


def build_score_html(entry):
    """Builds the quality/retrieval score grid shown under a bot answer."""
    sources = entry.get("sources", []) or []
    quality = entry.get("quality") or {}

    source_scores = [
        float(s.get("score", 0))
        for s in sources
        if s.get("score") is not None
    ]
    retrieval_score = (sum(source_scores) / len(source_scores)) if source_scores else None

    overall            = quality.get("overall")
    faithfulness       = quality.get("faithfulness")
    correctness        = quality.get("answer_relevancy")
    context_precision  = quality.get("context_precision")

    # If we truly have nothing to show, skip the box entirely.
    if not sources and not quality:
        return ""

    return f"""
    <div class="ws-scorebox">
        <div><b>Retrieval</b><span class="ws-score-pill">{format_score(retrieval_score)}</span></div>
        <div><b>Overall</b><span class="ws-score-pill">{format_score(overall)}</span></div>
        <div><b>Faithfulness</b><span class="ws-score-pill">{format_score(faithfulness)}</span></div>
        <div><b>Correctness</b><span class="ws-score-pill">{format_score(correctness)}</span></div>
        <div><b>Context precision</b><span class="ws-score-pill">{format_score(context_precision)}</span></div>
    </div>
    """


# ── Shared brand ──────────────────────────────────────────────────────────────
def brand():
    st.markdown('<div class="ws-brand"><div class="ws-brand-icon">🧠</div><span class="ws-brand-name">Workspace AI</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_auth():
    brand()

    # Tab switcher
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Sign in", use_container_width=True, key="tab_login"):
            st.session_state.auth_tab = "login"
    with col_b:
        if st.button("New account", use_container_width=True, key="tab_reg"):
            st.session_state.auth_tab = "register"

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── LOGIN ──
    if st.session_state.auth_tab == "login":
        st.markdown('<div class="ws-h1">Welcome back</div><div class="ws-sub">Sign in to access your workspace</div>', unsafe_allow_html=True)

        email    = st.text_input("Email", placeholder="you@company.com", key="li_email")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("Sign in", use_container_width=True, key="do_login"):
            if not email or not password:
                st.warning("Please enter your email and password.")
            else:
                with st.spinner("Signing in…"):
                    ok, data = api_login(email, password)
                if ok:
                    st.session_state.access_token  = data["access_token"]
                    st.session_state.refresh_token = data.get("refresh_token")
                    _, me = api_me(data["access_token"])
                    st.session_state.user = me
                    st.session_state.chat_history = []
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again.")

    # ── REGISTER ──
    else:
        st.markdown('<div class="ws-h1">Create an account</div><div class="ws-sub">Join your team on Workspace AI</div>', unsafe_allow_html=True)

        full_name  = st.text_input("Full name",  placeholder="Jane Smith",        key="rg_name")
        email      = st.text_input("Email",      placeholder="you@company.com",   key="rg_email")
        username   = st.text_input("Username",   placeholder="janesmith",         key="rg_user")
        department = st.selectbox("Your team", ["finance","hr","marketing","engineering","general"], key="rg_dept")
        password   = st.text_input("Password",   type="password", placeholder="••••••••", key="rg_pass")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("Create account", use_container_width=True, key="do_register"):
            if not all([full_name, email, username, password]):
                st.warning("Please fill in all fields.")
            else:
                with st.spinner("Creating your account…"):
                    ok, data = api_register(email, username, password, full_name, department)
                if ok:
                    st.success("Account created! Please sign in.")
                    st.session_state.auth_tab = "login"
                    st.rerun()
                else:
                    detail = data.get("detail", "Registration failed. Please try again.")
                    st.error(html.escape(str(detail)))

    st.markdown("<hr class='ws-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ws-trust"><div class="ws-trust-dot">🔒</div>You only see documents relevant to your team</div>
    <div class="ws-trust"><div class="ws-trust-dot">📄</div>Answers come from official company documents</div>
    <div class="ws-trust"><div class="ws-trust-dot">💬</div>Ask in your own words — no special commands needed</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP HEADER (shown on chat + history)
# ══════════════════════════════════════════════════════════════════════════════
def render_app_header():
    user = st.session_state.user or {}
    dept = html.escape((user.get("department") or "").replace("_", " ").title())
    name = html.escape(user.get("full_name") or user.get("username", ""))

    col1, col2 = st.columns([4, 1])
    with col1:
        brand()
    with col2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sign out", key="signout"):
            for k in ["access_token","refresh_token","user","chat_history"]:
                st.session_state[k] = None if "token" in k or k == "user" else []
            st.rerun()

    st.markdown(f"""
    <div class="ws-greeting">
        Hi, {name}!
        <span class="ws-badge">🏢 {dept} team</span>
    </div>
    """, unsafe_allow_html=True)

    # Nav pills
    col_c, col_h = st.columns(2)
    with col_c:
        if st.button("💬  Ask a question", key="nav_chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
    with col_h:
        if st.button("🕘  My history", key="nav_hist", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

    st.markdown("<hr class='ws-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_chat():
    render_app_header()

    # ── History bubbles ──
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            question_html = html.escape(entry.get("question", ""))

            # User bubble
            st.markdown(f"""
            <div class="ws-lbl-r">You</div>
            <div class="ws-msg-wrap-user">
                <div class="ws-bubble-user">{question_html}</div>
            </div>
            <div class="ws-gap"></div>
            """, unsafe_allow_html=True)

            # Bot bubble
            if entry.get("was_blocked"):
                blocked_answer = html.escape(entry.get("answer", ""))
                st.markdown(f"""
                <div class="ws-lbl">Workspace AI</div>
                <div class="ws-msg-wrap-bot">
                    <div class="ws-bubble-blocked">⚠️ {blocked_answer}</div>
                </div>
                <div class="ws-gap"></div>
                """, unsafe_allow_html=True)
            else:
                answer_html = html.escape(entry.get("answer", "")).replace("\n", "<br>")
                sources = entry.get("sources", [])
                score_html = build_score_html(entry)

                src_html = ""
                if sources:
                    src_lines = "".join([
                        "<div>📄 "
                        + html.escape(s.get("filename") or "—")
                        + (f" · p{html.escape(str(s.get('page')))}" if s.get("page") else "")
                        + f" <span style='color:#bbb'>({html.escape(s.get('department') or '')})</span></div>"
                        for s in sources[:3]
                    ])
                    src_html = f'<div class="ws-source">{src_lines}</div>'

                latency = entry.get("latency_ms", 0)
                lat_html = f'<div style="font-size:11px;color:#bbb;margin-top:6px">{latency}ms</div>' if latency else ""

                st.markdown(f"""
                <div class="ws-lbl">Workspace AI</div>
                <div class="ws-msg-wrap-bot">
                    <div class="ws-bubble-bot">{answer_html}{score_html}{src_html}{lat_html}</div>
                </div>
                <div class="ws-gap"></div>
                """, unsafe_allow_html=True)

        st.markdown("<hr class='ws-divider'>", unsafe_allow_html=True)

    else:
        # Empty state + suggestion chips
        st.markdown("""
        <div style='text-align:center;padding:1.25rem 0 0.75rem;color:#aaa;font-size:14px'>
            Ask me anything about your team's documents.
        </div>
        <div class="ws-chips">
            <div class="ws-chip">📋 What is the leave policy?</div>
            <div class="ws-chip">💸 How do I submit expenses?</div>
            <div class="ws-chip">🏠 Remote work guidelines</div>
            <div class="ws-chip">📚 Training budget</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input ──
    question = st.text_area(
        "Question",
        placeholder="Ask a question about your team's documents…",
        label_visibility="collapsed",
        height=88,
        key="q_input",
    )

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send = st.button("Send", use_container_width=True, key="send")
    with col_clear:
        clear = st.button("Clear chat", use_container_width=True, key="clear")

    if send:
        if not question.strip():
            st.warning("Please type a question before sending.")
        else:
            with st.spinner("Finding your answer…"):
                ok, data = api_query(question.strip())
            if not ok:
                st.error("We couldn't process that question. Please try again.")
            st.session_state.chat_history.append({
                "question":    question.strip(),
                "answer":      data.get("answer", ""),
                "sources":     data.get("sources", []),
                "quality":     data.get("quality"),
                "latency_ms":  data.get("latency_ms", 0),
                "was_blocked": data.get("was_blocked", False),
            })
            st.rerun()

    if clear:
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_history():
    render_app_header()

    with st.spinner("Loading your history…"):
        logs = api_history()

    if not logs:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0;color:#aaa;font-size:14px'>
            No questions yet. Go ask something!
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"<div style='font-size:13px;color:#888;margin-bottom:12px'>{len(logs)} recent questions</div>", unsafe_allow_html=True)

    for log in logs:
        status_badge = (
            '<span class="ws-badge-blk">⚠️ Blocked</span>'
            if log.get("was_blocked")
            else '<span class="ws-badge-ok">✓ Answered</span>'
        )

        raw_answer = log.get("answer") or ""
        answer_preview = html.escape(raw_answer[:180])
        if len(raw_answer) > 180:
            answer_preview += "…"

        question_html = html.escape(log.get("question", ""))
        dept  = html.escape((log.get("department") or "").replace("_", " ").title())
        lat   = log.get("latency_ms", 0)
        ts    = html.escape((log.get("created_at") or "")[:16].replace("T", " "))

        sources = log.get("sources", [])
        src_str = ""
        if sources:
            names = [html.escape(s.get("filename", "—")) for s in sources[:2]]
            src_str = f" · 📄 {', '.join(names)}"

        st.markdown(f"""
        <div class="ws-hist-card">
            <div class="ws-hist-q">{question_html}</div>
            <div class="ws-hist-a">{answer_preview}</div>
            <div class="ws-hist-meta">
                {status_badge}
                &nbsp;·&nbsp; {dept} team
                &nbsp;·&nbsp; {lat}ms
                {src_str}
                &nbsp;·&nbsp; {ts}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.access_token is None:
    render_auth()
elif st.session_state.page == "history":
    render_history()
else:
    render_chat()