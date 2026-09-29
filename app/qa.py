import ollama


class VideoQA:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model

    def answer(self, question, retrieved_chunks):

        context_parts = []

        for chunk in retrieved_chunks:
            start = format_time(chunk["start"])
            end = format_time(chunk["end"])

            context_parts.append(
                f"[{start} - {end}] {chunk['text']}"
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are an AI assistant answering questions about a video.

Answer the user's question ONLY using the provided video transcript.

If the transcript does not contain enough information,
say that the answer could not be found in the video.

Do not invent information.

Video transcript excerpts:

{context}

User question:
{question}

Give a concise answer.
Also mention the most relevant timestamp.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"