"use client";

import { useState, useRef, useEffect } from "react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  isThinking?: boolean;
}

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: ChatMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    // Add assistant thinking placeholder
    let assistantMessage: ChatMessage = {
      role: "assistant",
      content: "thinking...",
      isThinking: true,
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const res = await fetch("http://localhost:8000/rag-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });

      if (!res.body) throw new Error("No response body from RAG API");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      assistantMessage.content = ""; // Reset placeholder
      assistantMessage.isThinking = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        assistantMessage.content += decoder.decode(value);
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = assistantMessage;
          return copy;
        });
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "⚠️ AI Assistant: Error contacting RAG API",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-[#F9F7F3] p-6">
      <div className="flex flex-col max-w-3xl w-full mx-auto h-full">
        {/* Messages container */}
        <div className="flex-1 overflow-y-auto mb-4 space-y-4 p-4 bg-white rounded-2xl shadow-inner">
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex items-start ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.role === "assistant" && (
                <div className="flex-shrink-0 mr-2">
                  <div className="w-10 h-10 rounded-full bg-[#FDB750] flex items-center justify-center text-white font-bold">
                    AI
                  </div>
                </div>
              )}

              <div
                className={`max-w-[70%] px-4 py-2 rounded-xl shadow ${
                  message.role === "user"
                    ? "bg-[#FDB750] text-white rounded-br-none"
                    : "bg-white border border-[#FDB750] text-black rounded-bl-none"
                }`}
              >
                <span
                  className={`${
                    message.isThinking ? "animate-pulse italic" : ""
                  }`}
                >
                  {message.content}
                </span>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input box */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            value={input}
            placeholder="Type a message..."
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            className="flex-1 px-4 py-2 rounded-2xl border border-[#E0E0E0] bg-white text-black focus:outline-none focus:ring-2 focus:ring-[#FDB750]"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-[#FDB750] text-white px-6 py-2 rounded-2xl hover:bg-[#b48f20] disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
