from faster_whisper import WhisperModel


class VideoTranscriber:
    def __init__(self, model_size="base"):
        print("Loading Whisper model...")

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, video_path):
        print("Transcribing video...")

        segments, info = self.model.transcribe(
            video_path,
            beam_size=5
        )

        results = []

        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        return results
