from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scipy.signal import resample

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "pt"
DEFAULT_OUTPUT_SUFFIX = ".transcricao_pt.txt"
TARGET_SAMPLE_RATE = 16000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Converte um arquivo de audio para WAV mono 16 kHz e gera uma "
            "transcricao em portugues usando faster-whisper."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Caminho do arquivo de audio da reuniao (ex.: .mp3, .wav, .m4a).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Caminho do arquivo .txt de saida. Padrao: <input>.transcricao_pt.txt",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Modelo faster-whisper a usar. Padrao: medium. "
            "Para maior qualidade, experimente large-v2."
        ),
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Idioma fixo da transcricao. Padrao: pt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Dispositivo do WhisperModel. Padrao: "auto".',
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help='Compute type do WhisperModel. Padrao: "default".',
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size da transcricao final. Padrao: 5",
    )
    parser.add_argument(
        "--sample-minutes",
        type=float,
        help=(
            "Limita a conversao aos primeiros N minutos para validacao rapida. "
            "Use 10 ou 15 antes de processar a reuniao inteira."
        ),
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Ativa o filtro de VAD do faster-whisper durante a transcricao.",
    )
    return parser


def ensure_input_file(path: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {resolved_path}")
    return resolved_path


def ensure_ffmpeg_available() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg nao foi encontrado no PATH. Instale o ffmpeg antes de rodar este teste."
        )
    return ffmpeg_path


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return input_path.with_suffix(f"{input_path.suffix}{DEFAULT_OUTPUT_SUFFIX}")


def convert_audio_to_wav(
    ffmpeg_path: str,
    input_path: Path,
    wav_path: Path,
    sample_minutes: float | None,
) -> None:
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
    ]

    if sample_minutes is not None:
        if sample_minutes <= 0:
            raise ValueError("--sample-minutes deve ser maior que zero.")
        command.extend(["-t", str(sample_minutes * 60)])

    command.extend(
        [
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
    )

    subprocess.run(command, check=True)


def normalize_audio_array(audio_array, sample_rate: int):
    if sf is None:
        raise RuntimeError(
            "A dependencia 'soundfile' nao esta instalada. Rode 'pip install -r requirements.txt'."
        )

    if getattr(audio_array, "ndim", 1) == 2:
        audio_array = audio_array.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        target_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / sample_rate)
        audio_array = resample(audio_array, target_samples).astype("float32")
        sample_rate = TARGET_SAMPLE_RATE

    return audio_array, sample_rate


def load_audio_array(audio_path: Path):
    if sf is None:
        raise RuntimeError(
            "A dependencia 'soundfile' nao esta instalada. Rode 'pip install -r requirements.txt'."
        )

    audio_array, sample_rate = sf.read(audio_path, dtype="float32")
    return normalize_audio_array(audio_array, sample_rate)


def trim_audio_minutes(audio_array, sample_rate: int, sample_minutes: float | None):
    if sample_minutes is None:
        return audio_array
    if sample_minutes <= 0:
        raise ValueError("--sample-minutes deve ser maior que zero.")

    max_samples = int(sample_minutes * 60 * sample_rate)
    return audio_array[:max_samples]


def prepare_audio(input_path: Path, sample_minutes: float | None):
    try:
        audio_array, sample_rate = load_audio_array(input_path)
        audio_array = trim_audio_minutes(audio_array, sample_rate, sample_minutes)
        return audio_array, sample_rate
    except RuntimeError:
        raise
    except Exception:
        pass

    ffmpeg_path = ensure_ffmpeg_available()
    with tempfile.TemporaryDirectory(prefix="realtimestt_transcribe_") as temp_dir:
        wav_path = Path(temp_dir) / "meeting.wav"
        convert_audio_to_wav(
            ffmpeg_path=ffmpeg_path,
            input_path=input_path,
            wav_path=wav_path,
            sample_minutes=sample_minutes,
        )
        return load_audio_array(wav_path)


def transcribe_audio(args: argparse.Namespace, audio_array) -> tuple[str, str | None]:
    if WhisperModel is None:
        raise RuntimeError(
            "A dependencia 'faster-whisper' nao esta instalada. Rode 'pip install -r requirements.txt'."
        )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments, info = model.transcribe(
        audio_array,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )

    segment_texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            segment_texts.append(text)

    transcript = " ".join(segment_texts).strip()
    detected_language = getattr(info, "language", None)
    return transcript, detected_language


def write_transcript(output_path: Path, transcript: str) -> None:
    output_path.write_text(transcript + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path = ensure_input_file(args.input_file)
        output_path = build_output_path(input_path, args.output)
        audio_array, sample_rate = prepare_audio(
            input_path=input_path,
            sample_minutes=args.sample_minutes,
        )

        print(f"Arquivo de entrada: {input_path}")
        print(f"Modelo: {args.model}")
        print(f"Idioma fixo: {args.language}")
        print(f"Sample rate preparado para transcricao: {sample_rate} Hz")
        if args.sample_minutes is None:
            print(
                "Dica: para validar rapidamente, rode primeiro com --sample-minutes 15."
            )

        transcript, detected_language = transcribe_audio(args, audio_array)
        write_transcript(output_path, transcript)

        if detected_language:
            print(f"Idioma detectado pelo modelo: {detected_language}")
        print(f"Transcricao salva em: {output_path}")
        print(f"Tamanho da transcricao: {len(transcript)} caracteres")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Falha ao converter audio com ffmpeg: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
