import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import yaml


def _gm_program_to_slakh_class(program_num: int) -> str:
    """
    Map a General MIDI program number (0-127) to one of the Slakh class names
    supported by `contrib.preprocessor._SLAKH_CLASS_PROGRAMS`.
    """
    # GM program families are 0-indexed, 8 programs per family.
    fam = int(program_num) // 8

    # Pianos
    if fam == 0:
        return "Acoustic Piano"
    if fam == 1:
        return "Electric Piano"
    # Chromatic percussion
    if fam == 2:
        return "Chromatic Percussion"
    # Organ
    if fam == 3:
        return "Organ"
    # Guitars (acoustic/clean/distorted)
    if fam == 4:
        # 24-31
        if program_num <= 25:
            return "Acoustic Guitar"
        if program_num <= 28:
            return "Clean Electric Guitar"
        return "Distorted Electric Guitar"
    # Bass
    if fam == 5:
        # 32-39
        return "Electric Bass" if program_num >= 33 else "Acoustic Bass"
    # Strings (violin/viola/cello/contrabass)
    if fam == 6:
        # 48-55
        if program_num <= 49:
            return "String Ensemble"
        if program_num <= 51:
            return "Synth Strings"
        if program_num <= 55:
            return "Choir and Voice" if program_num <= 54 else "Orchestral Hit"
        return "String Ensemble"
    # Brass
    if fam == 7:
        # 56-63
        if program_num <= 58:
            return ["Trumpet", "Trombone", "Tuba"][program_num - 56]
        if program_num <= 60:
            return "French Horn"
        return "Brass Section"
    # Reeds / pipe
    if fam == 8:
        # 64-71
        if program_num <= 65:
            return "Soprano/Alto Sax"
        if program_num == 66:
            return "Tenor Sax"
        if program_num == 67:
            return "Baritone Sax"
        if program_num == 68:
            return "Oboe"
        if program_num == 69:
            return "English Horn"
        if program_num == 70:
            return "Bassoon"
        return "Clarinet"
    if fam == 9:
        # 72-79
        return "Pipe"
    # Synth lead / pad
    if fam == 10:
        # 80-87
        return "Synth Lead"
    if fam == 11:
        # 88-95
        return "Synth Pad"

    # Default fallbacks: prefer pads for FX/unknown.
    return "Synth Pad"


def _read_metadata_inst_map(metadata_path: Path) -> dict[str, str]:
    meta = yaml.safe_load(metadata_path.read_text())
    stems = meta.get("stems", {}) or {}
    out: dict[str, str] = {}
    for stem_id, stem_meta in stems.items():
        # Expect stem_id like "S00"
        is_drum = bool(stem_meta.get("is_drum", False))
        if is_drum:
            out[stem_id] = "Drums"
            continue

        program_num = stem_meta.get("program_num")
        if program_num is None:
            # Fall back to provided class name if present.
            inst_class = stem_meta.get("inst_class")
            if inst_class:
                out[stem_id] = str(inst_class)
            continue

        out[stem_id] = _gm_program_to_slakh_class(int(program_num))
    return out


def _write_inst_names_json(track_dir: Path) -> None:
    metadata_path = track_dir / "metadata.yaml"
    midi_dir = track_dir / "MIDI"
    if not metadata_path.exists() or not midi_dir.exists():
        return

    inst_map = _read_metadata_inst_map(metadata_path)
    midi_stems = [p for p in midi_dir.glob("S*.mid")]

    # Dataset code expects a mapping: { "<stem_file_basename>": "<inst_class>" }
    # and later loads "<midi_dir>/<stem>.mid"
    out = {}
    for midi_path in midi_stems:
        stem = midi_path.stem  # "S00"
        if stem in inst_map:
            out[stem] = inst_map[stem]

    (track_dir / "inst_names.json").write_text(json.dumps(out, indent=2, sort_keys=True))


def _resample_to_16k(in_wav: Path, out_wav: Path) -> None:
    audio, sr = sf.read(str(in_wav), always_2d=False)
    if sr == 16000:
        sf.write(str(out_wav), audio, 16000, subtype="PCM_24")
        return

    # librosa resample via soundfile->numpy is simplest, but keep deps minimal:
    # use scipy if available, else fall back to librosa.
    try:
        import scipy.signal  # type: ignore

        # polyphase resampling
        import math

        g = math.gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        audio_rs = scipy.signal.resample_poly(audio, up=up, down=down, axis=0)
    except Exception:
        import librosa  # type: ignore

        audio_rs = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    sf.write(str(out_wav), audio_rs, 16000, subtype="PCM_24")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to babySlakh root (contains Track00001, ...)")
    ap.add_argument(
        "--dst",
        required=True,
        help="Path to output root (will create train/validation/test subfolders)",
    )
    ap.add_argument("--split", choices=["test", "train", "validation"], default="test")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_split = dst_root / args.split
    dst_split.mkdir(parents=True, exist_ok=True)

    tracks = sorted([p for p in src.iterdir() if p.is_dir() and p.name.lower().startswith("track")])
    if not tracks:
        raise SystemExit(f"No Track* directories found under {src}")

    for track in tracks:
        out_track = dst_split / track.name
        out_track.mkdir(parents=True, exist_ok=True)

        # Symlink MIDI/ (and other metadata) to avoid copying large files.
        for name in ["MIDI", "metadata.yaml", "all_src.mid"]:
            src_path = track / name
            if not src_path.exists():
                continue
            dst_path = out_track / name
            if dst_path.exists():
                continue
            os.symlink(src_path, dst_path)

        # Slakh evaluation expects all_src_v2.mid.
        # babySlakh provides all_src.mid, so create a compatible alias.
        gt_v2 = out_track / "all_src_v2.mid"
        if not gt_v2.exists():
            # Prefer linking to keep things lightweight.
            src_all = out_track / "all_src.mid"
            if src_all.exists():
                os.symlink(src_all, gt_v2)

        # Create inst_names.json in-place (in out_track) so dataset can read it.
        _write_inst_names_json(out_track)

        # Create mix_16k.wav expected by configs.
        src_mix = track / "mix.wav"
        if src_mix.exists():
            out_mix = out_track / "mix_16k.wav"
            if not out_mix.exists():
                _resample_to_16k(src_mix, out_mix)

    print(f"Prepared {len(tracks)} tracks into {dst_split}")


if __name__ == "__main__":
    main()

