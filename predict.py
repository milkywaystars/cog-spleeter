import subprocess
import tempfile

from cog import BaseModel, BasePredictor, Input, Path
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator


class ModelOutput(BaseModel):
    # Only return the compressed vocals file. The Cloudflare pipeline does not
    # need Replicate to upload and retain the accompaniment stem.
    vocals: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the Spleeter two-stem model once when the container starts."""
        self.separator = Separator("spleeter:2stems")
        self.audio_loader = AudioAdapter.default()

    def predict(
        self,
        audio: Path = Input(description="MP4, MP3, M4A or other FFmpeg-readable media"),
    ) -> ModelOutput:
        """Separate vocals and return mono 16 kHz, 128 kbps MP3 audio."""

        # The existing model loads the complete source before separation.
        waveform, sample_rate = self.audio_loader.load(str(audio))
        prediction = self.separator.separate(waveform)

        output_directory = Path(tempfile.mkdtemp())
        vocals_wav = output_directory / "vocals.wav"
        vocals_mp3 = output_directory / "vocals.mp3"

        # Spleeter generates the vocals stem as uncompressed WAV first.
        self.audio_loader.save(
            str(vocals_wav),
            prediction["vocals"],
            sample_rate,
        )

        # Compress it before Replicate uploads the result.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(vocals_wav),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(vocals_mp3),
            ],
            check=True,
        )

        return ModelOutput(vocals=vocals_mp3)
