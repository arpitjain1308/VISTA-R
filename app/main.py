import sys

from transcriber import VideoTranscriber
from retriever import SemanticRetriever
from qa import VideoQA


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python app/main.py <video_path>")
        return

    video_path = sys.argv[1]

    # -----------------------------
    # 1. TRANSCRIPTION
    # -----------------------------

    transcriber = VideoTranscriber(
        model_size="base"
    )

    transcript = transcriber.transcribe(
        video_path
    )

    print(
        f"\nTranscription complete: "
        f"{len(transcript)} segments\n"
    )

    # -----------------------------
    # 2. SEMANTIC INDEX
    # -----------------------------

    retriever = SemanticRetriever()

    retriever.index(
        transcript
    )

    print("Video indexed successfully.")

    # -----------------------------
    # 3. QUESTION ANSWERING
    # -----------------------------

    qa = VideoQA()

    print("\nVISTA-R is ready.")
    print("Ask questions about the video.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        results = retriever.search(
            question,
            top_k=5
        )

        answer = qa.answer(
            question,
            results
        )

        print("\nVISTA-R:")
        print(answer)

        print("\nRetrieved evidence:")

        for result in results:

            start = format_time(
                result["start"]
            )

            end = format_time(
                result["end"]
            )

            print(
                f"[{start} - {end}] "
                f"{result['score']:.3f} "
                f"{result['text']}"
            )

        print()


def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    main()