import { streamText } from "ai";

// Helper to convert UIMessage[] to ModelMessage[]
function convertToModelMessages(messages: any[]) {
  return messages.map((m) => ({
    role: m.role as "system" | "user" | "assistant",
    content:
      typeof m.content === "string"
        ? m.content
        : m.parts?.map((p: any) => p.text).join("") || "",
  }));
}

export const runtime = "edge";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    if (!messages || !Array.isArray(messages)) {
      return new Response("Missing messages array", { status: 400 });
    }

    const systemMessage = {
      role: "system",
      content: "You are a helpful assistant.",
    };

    const modelMessages = [systemMessage, ...convertToModelMessages(messages)];

    // Combine all user messages to a single string
    const userQuery = modelMessages
      .filter((m) => m.role === "user")
      .map((m) => m.content)
      .join("\n");

    // Stream wrapper
    const result = streamText({
      async generator() {
        // Call your Python RAG API
        const res = await fetch("http://localhost:8000/rag", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: userQuery }),
        });

        if (!res.body) throw new Error("No response from RAG API");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let done = false;

        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            yield decoder.decode(value, { stream: true });
          }
        }
      },
    });

    return result.toUIMessageStreamResponse();
  } catch (err: any) {
    console.error("POST /api/chat error:", err);
    return new Response(err.message || "Internal Server Error", {
      status: 500,
    });
  }
}
