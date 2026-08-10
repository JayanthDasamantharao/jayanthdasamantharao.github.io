'use strict';



// element toggle function
const elementToggleFunc = function (elem) { elem.classList.toggle("active"); }



// sidebar variables
const sidebar = document.querySelector("[data-sidebar]");
const sidebarBtn = document.querySelector("[data-sidebar-btn]");

// sidebar toggle functionality for mobile
sidebarBtn.addEventListener("click", function () { elementToggleFunc(sidebar); });

// contact form variables
const form = document.querySelector("[data-form]");
const formInputs = document.querySelectorAll("[data-form-input]");
const formBtn = document.querySelector("[data-form-btn]");

// add event to all form input field
for (let i = 0; i < formInputs.length; i++) {
  formInputs[i].addEventListener("input", function () {

    // check form validation
    if (form.checkValidity()) {
      formBtn.removeAttribute("disabled");
    } else {
      formBtn.setAttribute("disabled", "");
    }

  });
}



// page navigation variables
const navigationLinks = document.querySelectorAll("[data-nav-link]");
const pages = document.querySelectorAll("[data-page]");

// add event to all nav link
for (let i = 0; i < navigationLinks.length; i++) {
  navigationLinks[i].addEventListener("click", function () {

    for (let i = 0; i < pages.length; i++) {
      if (this.innerHTML.toLowerCase() === pages[i].dataset.page) {
        pages[i].classList.add("active");
        navigationLinks[i].classList.add("active");
        window.scrollTo(0, 0);
      } else {
        pages[i].classList.remove("active");
        navigationLinks[i].classList.remove("active");
      }
    }

  });
}

// chatbot widget
const chatbotToggle = document.querySelector("[data-chatbot-toggle]");
const chatbotPanel = document.querySelector("[data-chatbot-panel]");
const chatbotClose = document.querySelector("[data-chatbot-close]");
const chatbotMinimize = document.querySelector("[data-chatbot-minimize]");
const chatbotMessages = document.querySelector("[data-chatbot-messages]");
const chatbotForm = document.querySelector("[data-chatbot-form]");
const chatbotInput = document.querySelector("[data-chatbot-input]");

if (chatbotToggle && chatbotPanel && chatbotClose && chatbotMinimize && chatbotMessages && chatbotForm && chatbotInput) {
  const chatHistory = [];
  const apiEndpoint = "https://jayanth-portfolio-api.onrender.com/api/chat";

  /** Resolve relative /api/... paths against the API host (chat on GitHub Pages cannot fetch same-origin /api). */
  const resolveApiUrl = function (pathOrUrl) {
    if (!pathOrUrl) return pathOrUrl;
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
    try {
      const base = new URL(apiEndpoint);
      const path = pathOrUrl.startsWith("/") ? pathOrUrl : "/" + pathOrUrl;
      return base.origin + path;
    } catch (e) {
      return pathOrUrl;
    }
  };
  const chatbotSendButton = chatbotForm.querySelector(".chatbot-send");
  const chatbotSuggestions = chatbotMessages.querySelector("[data-chatbot-suggestions]");
  const chatbotSuggestionButtons = chatbotMessages.querySelectorAll("[data-chatbot-suggestion]");
  const abusiveLockThreshold = 5;
  let abusiveMessageCount = 0;
  let chatLockedForAbuse = false;
  const chatSessionStorageKey = "jayanth_chat_session_id";
  let chatSessionId = window.localStorage.getItem(chatSessionStorageKey);
  if (!chatSessionId) {
    chatSessionId = (window.crypto && window.crypto.randomUUID)
      ? window.crypto.randomUUID()
      : `sess_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    window.localStorage.setItem(chatSessionStorageKey, chatSessionId);
  }
  const chatbotMinimizeIcon = chatbotMinimize.querySelector("ion-icon");

  const escapeHtml = function (value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  };

  const formatRichText = function (value) {
    let safe = escapeHtml(value);
    safe = safe.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    safe = safe.replace(
      /\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b/gi,
      '<a href="mailto:$1">$1</a>'
    );
    const blocks = safe
      .split(/\n{2,}/)
      .map(function (block) { return block.trim(); })
      .filter(Boolean)
      .map(function (block) {
        return `<p class="chatbot-paragraph">${block.replace(/\n/g, "<br>")}</p>`;
      });
    return blocks.length ? blocks.join("") : safe.replace(/\n/g, "<br>");
  };

  const createMessageBubble = function (sender) {
    const bubble = document.createElement("div");
    bubble.className = `chatbot-message ${sender}`;
    const content = document.createElement("div");
    bubble.appendChild(content);
    chatbotMessages.appendChild(bubble);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return { bubble, content };
  };

  const sleep = function (ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  };

  const setMinimizedState = function (isMinimized) {
    chatbotPanel.classList.toggle("minimized", isMinimized);
    chatbotMinimize.setAttribute("aria-label", isMinimized ? "Expand chat assistant" : "Minimize chat assistant");
    if (chatbotMinimizeIcon) {
      chatbotMinimizeIcon.setAttribute("name", isMinimized ? "expand-outline" : "chevron-down-outline");
    }
  };

  const toggleChatbot = function (shouldOpen) {
    const wasOpen = chatbotPanel.classList.contains("active");
    const isOpen = typeof shouldOpen === "boolean" ? shouldOpen : !chatbotPanel.classList.contains("active");
    chatbotPanel.classList.toggle("active", isOpen);
    chatbotPanel.setAttribute("aria-hidden", String(!isOpen));

    if (isOpen) {
      if (!wasOpen) {
        setMinimizedState(false);
      }
      chatbotInput.focus();
    }
  };

  const addMessage = function (message, sender) {
    const node = createMessageBubble(sender);
    if (sender === "bot") {
      node.content.innerHTML = formatRichText(message);
    } else {
      node.content.textContent = message;
    }
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return node.bubble;
  };

  const appendResumeAttachment = function (bubble, attachment) {
    if (!attachment || !attachment.download_url) {
      return;
    }
    const wrap = document.createElement("div");
    wrap.className = "chatbot-attachment";
    const a = document.createElement("a");
    a.className = "chatbot-attachment-link";
    a.href = resolveApiUrl(attachment.download_url);
    a.textContent = attachment.label || "Download resume";
    if (attachment.filename) {
      a.setAttribute("download", attachment.filename);
    }
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    wrap.appendChild(a);
    bubble.appendChild(wrap);
  };

  const addMessageTyped = async function (message, sender, attachment) {
    const node = createMessageBubble(sender);
    const full = String(message);
    if (sender !== "bot") {
      node.content.textContent = full;
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
      return node.bubble;
    }

    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      node.content.innerHTML = formatRichText(full);
      if (attachment) {
        appendResumeAttachment(node.bubble, attachment);
      }
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
      return node.bubble;
    }

    // Reveal in multi-character steps (not word-by-word) and throttle scroll updates
    // so the pane does not jitter on every token.
    const charStep = 18;
    const tickMs = 32;
    const scrollEveryMs = 110;
    let lastScrollAt = 0;

    node.content.classList.add("chatbot-reply-streaming");

    for (let pos = 0; pos < full.length; ) {
      pos = Math.min(full.length, pos + charStep);
      node.content.textContent = full.slice(0, pos);
      const now = Date.now();
      if (now - lastScrollAt >= scrollEveryMs) {
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        lastScrollAt = now;
      }
      if (pos < full.length) {
        await sleep(tickMs);
      }
    }

    node.content.classList.remove("chatbot-reply-streaming");
    node.content.innerHTML = formatRichText(full);
    if (attachment) {
      appendResumeAttachment(node.bubble, attachment);
    }
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return node.bubble;
  };

  const addTypingIndicator = function () {
    const bubble = document.createElement("div");
    bubble.className = "chatbot-message bot typing";

    const label = document.createElement("span");
    label.className = "typing-label";
    label.textContent = "Ada is typing";

    const dots = document.createElement("span");
    dots.className = "typing-dots";
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("span");
      dots.appendChild(dot);
    }

    bubble.appendChild(label);
    bubble.appendChild(dots);
    chatbotMessages.appendChild(bubble);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return bubble;
  };

  const hideStarterSuggestions = function () {
    if (chatbotSuggestions) {
      chatbotSuggestions.style.display = "none";
    }
  };

  const showStarterSuggestions = function () {
    if (!chatbotSuggestions || chatHistory.length > 0) {
      return;
    }
    chatbotSuggestions.style.display = "flex";
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
  };

  const getFallbackReply = function (text, includeIntro) {
    const input = text.toLowerCase();
    const intro = "Hey! I'm Ada, Jayanth's AI 👋 ";
    let reply = "";

    if (input.includes("experience") || input.includes("work")) {
      reply = "I currently work as an Applied AI/ML Engineer and focus on production AI systems, RAG pipelines, and scalable data workflows.";
    } else if (input.includes("project") || input.includes("research")) {
      reply = "Please check the Projects and Research tabs for details on my NLP, machine learning, and computer vision work.";
    } else if (input.includes("skill") || input.includes("tech")) {
      reply = "My core stack includes Python, SQL, FastAPI, AWS/Azure, LLM applications, and machine learning evaluation.";
    } else if (input.includes("contact") || input.includes("email")) {
      reply = "You can reach me at jayanthdasamantharao@gmail.com or through the Contact section on this page.";
    } else {
      reply = (
        "I might be missing that detail right now, but I can check with Jayanth for you 🙂\n"
        + "You can also reach him directly at:\n"
        + "jayanthdasamantharao@gmail.com\n"
        + "https://www.linkedin.com/in/djayanth/"
      );
    }

    return includeIntro ? intro + reply : reply;
  };

  const getBotReply = async function (text) {
    const response = await fetch(apiEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: chatHistory,
        session_id: chatSessionId,
      }),
    });

    if (!response.ok) {
      let detail = "Chat API unavailable";
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch (error) {
        // No-op: keep default detail
      }
      throw new Error(detail);
    }

    const data = await response.json();
    if (data.session_id && data.session_id !== chatSessionId) {
      chatSessionId = data.session_id;
      window.localStorage.setItem(chatSessionStorageKey, chatSessionId);
    }
    return {
      reply: data.reply || "I do not have enough verified information to answer that.",
      intent: data.intent || "",
      attachment: data.attachment || null,
    };
  };

  const setChatEnabled = function (enabled) {
    chatbotInput.disabled = !enabled;
    if (chatbotSendButton) {
      chatbotSendButton.disabled = !enabled;
    }
  };

  const lockChatForAbuse = function () {
    chatLockedForAbuse = true;
    setChatEnabled(false);
    const notice = "This chat is temporarily disabled due to repeated abusive language. Please refresh the page to continue.";
    addMessage(notice, "bot");
    chatHistory.push({ role: "assistant", content: notice });
  };

  const processUserMessage = async function (text) {
    if (!text || chatLockedForAbuse) {
      return;
    }

    hideStarterSuggestions();
    addMessage(text, "user");
    chatHistory.push({ role: "user", content: text });
    chatbotInput.value = "";
    setChatEnabled(false);

    const typingBubble = addTypingIndicator();

    try {
      const result = await getBotReply(text);
      typingBubble.remove();
      await addMessageTyped(result.reply, "bot", result.attachment);
      chatHistory.push({ role: "assistant", content: result.reply });
      if (result.intent === "abusive") {
        abusiveMessageCount += 1;
        if (abusiveMessageCount >= abusiveLockThreshold) {
          lockChatForAbuse();
        }
      }
    } catch (error) {
      const fallback = getFallbackReply(text, chatHistory.length <= 1);
      typingBubble.remove();
      await addMessageTyped(fallback, "bot");
      chatHistory.push({ role: "assistant", content: fallback });
    } finally {
      if (!chatLockedForAbuse) {
        setChatEnabled(true);
        chatbotInput.focus();
      }
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }
  };

  chatbotToggle.addEventListener("click", function () {
    toggleChatbot();
    showStarterSuggestions();
  });

  chatbotClose.addEventListener("click", function () {
    toggleChatbot(false);
  });

  chatbotMinimize.addEventListener("click", function () {
    const currentlyMinimized = chatbotPanel.classList.contains("minimized");
    setMinimizedState(!currentlyMinimized);
    if (!currentlyMinimized) {
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    } else {
      chatbotInput.focus();
    }
  });

  chatbotForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const text = chatbotInput.value.trim();
    await processUserMessage(text);
  });

  chatbotSuggestionButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      if (chatLockedForAbuse) {
        return;
      }
      const prompt = (button.textContent || "").trim();
      if (!prompt) {
        return;
      }
      void processUserMessage(prompt);
    });
  });

  showStarterSuggestions();
}