"use client";

import { useState, FormEvent, useRef, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = {
  source: string;
  score: number | null;
  preview: string;
};

type Stats = {
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  stats?: Stats;
  error?: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 4 }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`API ${res.status}: ${errText}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          stats: data.stats,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "",
          error: err instanceof Error ? err.message : String(err),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 text-slate-900">
      {/* ── header ───────────────────────────────────────────────────── */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-baseline justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Pareto</h1>
            <p className="text-sm text-slate-500">
              Cost-optimized RAG. Ask anything about your corpus.
            </p>
          </div>
          <a
            href="https://github.com/Engineer-coding/pareto"
            target="_blank"
            rel="noopener"
            className="text-sm text-slate-500 hover:text-slate-900 transition"
          >
            GitHub →
          </a>
        </div>
      </header>

      {/* ── messages ─────────────────────────────────────────────────── */}
      <div className="max-w-3xl mx-auto px-6 py-8 pb-40">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <p className="text-slate-400 text-lg mb-6">
              Ask a question to get started
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {[
                "What is GDPR?",
                "Summarize Basel III",
                "What are hypertension symptoms?",
              ].map((sample) => (
                <button
                  key={sample}
                  onClick={() => setInput(sample)}
                  className="px-4 py-2 text-sm bg-white border border-slate-200 rounded-full hover:border-slate-400 transition"
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((msg, idx) => (
            <MessageBubble key={idx} message={msg} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-slate-500 italic">
              <span className="inline-block w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
              <span
                className="inline-block w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="inline-block w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
              <span className="ml-2 text-sm">Pareto is thinking…</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ── input ────────────────────────────────────────────────────── */}
      <form
        onSubmit={handleSubmit}
        className="fixed bottom-0 inset-x-0 border-t border-slate-200 bg-white/95 backdrop-blur-sm"
      >
        <div className="max-w-3xl mx-auto px-6 py-4 flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Pareto a question…"
            disabled={loading}
            className="flex-1 px-4 py-3 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 disabled:bg-slate-100 disabled:text-slate-400"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-5 py-3 bg-slate-900 text-white rounded-xl font-medium hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition"
          >
            {loading ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </main>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Message bubble
// ──────────────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-slate-900 text-white px-5 py-3 rounded-2xl rounded-br-md">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant
  if (message.error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-800 px-5 py-4 rounded-xl">
        <p className="font-medium mb-1">Error</p>
        <p className="text-sm font-mono">{message.error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* answer */}
      <div className="bg-white border border-slate-200 px-5 py-4 rounded-2xl rounded-bl-md whitespace-pre-wrap leading-relaxed">
        {message.content}
      </div>

      {/* citations */}
      {message.citations && message.citations.length > 0 && (
        <div className="px-1">
          <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">
            Sources
          </p>
          <div className="flex flex-wrap gap-2">
            {message.citations.map((c, i) => (
              <CitationChip key={i} citation={c} index={i + 1} />
            ))}
          </div>
        </div>
      )}

      {/* stats */}
      {message.stats && <StatsRow stats={message.stats} />}
    </div>
  );
}

function CitationChip({
  citation,
  index,
}: {
  citation: Citation;
  index: number;
}) {
  return (
    <div className="group relative">
      <div className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 transition text-xs rounded-lg cursor-pointer">
        <span className="text-slate-400 mr-1">[{index}]</span>
        <span className="font-medium">{citation.source}</span>
        {citation.score !== null && (
          <span className="text-slate-400 ml-2">
            {citation.score.toFixed(3)}
          </span>
        )}
      </div>
      {/* Tooltip with preview */}
      <div className="absolute bottom-full left-0 mb-2 w-80 p-3 bg-slate-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition z-20 shadow-lg">
        {citation.preview}…
      </div>
    </div>
  );
}

function StatsRow({ stats }: { stats: Stats }) {
  return (
    <div className="px-1 text-xs text-slate-400 font-mono flex flex-wrap gap-x-4 gap-y-1">
      <span>{stats.model}</span>
      <span>
        {stats.total_tokens} tok (p{stats.prompt_tokens} / c
        {stats.completion_tokens})
      </span>
      <span>retr {stats.retrieval_latency_ms}ms</span>
      <span>gen {stats.generation_latency_ms}ms</span>
      <span>
        {stats.cost_usd === 0
          ? "$0.00 (local)"
          : `$${stats.cost_usd.toFixed(5)}`}
      </span>
    </div>
  );
}