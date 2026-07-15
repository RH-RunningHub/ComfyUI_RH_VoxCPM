import logging
import os
import re
import tempfile
import inspect

import numpy as np
import torch
import comfy.utils

from .generate import (
    _denoise_audio,
    _generate_with_quality_guard,
    _safe_save_wav,
    _sanitize_control,
)

logger = logging.getLogger("RunningHub.VoxCPM")

NUM_SPEAKERS = 5
DEFAULT_REFERENCE_AUDIO_INPUTS = 2
TARGET_RMS = 0.08
RMS_FLOOR = 1e-6
_AUDIO_INPUT_PATTERN = re.compile(r"^audio_(\d+)$")


def _normalize_rms(wav, target_rms=TARGET_RMS):
    """Normalize a 1-D numpy waveform to a target RMS level."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < RMS_FLOOR:
        return wav
    return wav * (target_rms / rms)


def _save_audio_to_temp(audio_dict):
    """Save ComfyUI AUDIO dict to a temp wav file, return path."""
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    _safe_save_wav(tmp.name, waveform.cpu(), sr)
    return tmp.name


def _parse_script(text, max_speakers=None):
    """Parse multi-speaker script into ordered segments.

    Format: [spk1]Hello world[spk2]How are you?[spk1]I'm fine.
    Returns: [(1, "Hello world"), (2, "How are you?"), (1, "I'm fine.")]
    """
    pattern = re.compile(r"\[spk(\d+)\]", re.IGNORECASE)
    parts = pattern.split(text)

    # parts = ['prefix_before_any_tag', '1', 'text1', '2', 'text2', ...]
    # If text starts with [spk1], parts[0] is empty string
    segments = []

    # Handle text before any tag — treat as spk1
    if parts[0].strip():
        segments.append((1, parts[0].strip()))

    for i in range(1, len(parts), 2):
        spk_idx = int(parts[i])
        if spk_idx < 1:
            raise ValueError(f"Invalid speaker index: [spk{spk_idx}], must be >= 1")
        if max_speakers is not None and spk_idx > max_speakers:
            raise ValueError(
                f"Invalid speaker index: [spk{spk_idx}], must be 1-{max_speakers}"
            )
        segment_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if segment_text:
            segments.append((spk_idx, segment_text))

    if not segments:
        raise ValueError(
            "No valid segments found. Use [spk1]...[spk2]... format. "
            "Example: [spk1]Hello[spk2]Hi there"
        )
    return segments


def _parse_speaker_controls(text, max_speakers=None):
    """Parse tagged control text into per-speaker control instructions."""
    control_text = (text or "").strip()
    if not control_text:
        return {}

    controls = {}
    for spk_idx, segment_text in _parse_script(control_text, max_speakers=max_speakers):
        controls.setdefault(spk_idx, []).append(segment_text)

    cleaned = {}
    for spk_idx, parts in controls.items():
        joined = "\n".join(parts).strip()
        sanitized = _sanitize_control(joined)
        if sanitized:
            cleaned[spk_idx] = sanitized
    return cleaned


def _parse_reference_texts(text, max_speakers=None):
    """Parse exact, user-supplied reference transcripts by speaker tag."""

    tagged_text = (text or "").strip()
    if not tagged_text:
        return {}

    transcripts = {}
    for spk_idx, segment_text in _parse_script(
        tagged_text, max_speakers=max_speakers
    ):
        transcripts.setdefault(spk_idx, []).append(segment_text)
    return {
        spk_idx: "\n".join(parts).strip()
        for spk_idx, parts in transcripts.items()
        if "\n".join(parts).strip()
    }


class _DynamicAudioOptionalInputs(dict):
    def __contains__(self, item):
        if super().__contains__(item):
            return True
        if not isinstance(item, str):
            return False
        match = _AUDIO_INPUT_PATTERN.match(item)
        if not match:
            return False
        index = int(match.group(1))
        return index >= 1

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key not in self:
            raise KeyError(key)
        return ("AUDIO",)


class RunningHubVoxCPMMultiSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "script": ("STRING", {
                    "default": "[spk5]今日AI新闻速报\n[spk1]T8那个瓜娃子，又更新什么了？\n[spk2]管他干啥呀，带派不就行了。\n[spk1]天天只知道做视频，耳都不耳人一哈。\n[spk2]别笑哈，你试你也过不了第二关。",
                    "multiline": True,
                }),
                "cfg_value": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "inference_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": {},
        }

        control_defaults = {
            1: "四川话",
            2: "成年女性，东北话",
            3: "",
            4: "",
            5: "旁白音，成熟男性",
        }
        for i in range(1, NUM_SPEAKERS + 1):
            inputs["optional"][f"audio_{i}"] = ("AUDIO",)
            inputs["optional"][f"control_{i}"] = ("STRING", {
                "default": control_defaults.get(i, ""),
            })

        inputs["optional"]["normalize_text"] = ("BOOLEAN", {"default": False})
        inputs["optional"]["denoise_reference"] = ("BOOLEAN", {"default": False})
        inputs["optional"]["max_len"] = ("INT", {
            "default": 4096,
            "min": 64,
            "max": 8192,
            "step": 64,
        })
        inputs["optional"]["retry_badcase"] = ("BOOLEAN", {"default": True})
        # Append new widgets after all legacy widgets so existing serialized
        # widget_values keep their positional meaning when old workflows load.
        for i in range(1, NUM_SPEAKERS + 1):
            inputs["optional"][f"reference_text_{i}"] = ("STRING", {
                "default": "",
                "multiline": True,
            })

        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "RunningHub/VoxCPM"

    def generate(self, model, script, cfg_value, inference_steps, seed, **kwargs):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        voxcpm_model = model["model"]
        sample_rate = model["sample_rate"]
        is_v2 = model["architecture"] == "voxcpm2"

        normalize_text = kwargs.get("normalize_text", False)
        denoise_reference = kwargs.get("denoise_reference", False)
        max_len = kwargs.get("max_len", 4096)
        retry_badcase = kwargs.get("retry_badcase", True)

        segments = _parse_script(script, max_speakers=NUM_SPEAKERS)
        logger.info("Parsed %d segments from script", len(segments))

        speaker_audios = {}
        speaker_controls = {}
        speaker_reference_texts = {}
        for i in range(1, NUM_SPEAKERS + 1):
            speaker_audios[i] = kwargs.get(f"audio_{i}")
            speaker_controls[i] = _sanitize_control(
                (kwargs.get(f"control_{i}") or "").strip()
            )
            speaker_reference_texts[i] = (
                kwargs.get(f"reference_text_{i}") or ""
            ).strip()

        total_steps = len(segments) * (int(inference_steps) + 2) + 1
        pbar = comfy.utils.ProgressBar(total_steps)
        step_counter = 0

        temp_files = []
        generated_segments = []

        try:
            spk_wav_cache = {}
            for spk_idx, audio in speaker_audios.items():
                if audio is not None:
                    wav_path = _save_audio_to_temp(audio)
                    temp_files.append(wav_path)
                    if denoise_reference:
                        denoised = _denoise_audio(wav_path)
                        temp_files.append(denoised)
                        wav_path = denoised
                    spk_wav_cache[spk_idx] = wav_path

            for segment_index, (spk_idx, segment_text) in enumerate(segments, 1):
                ref_wav_path = spk_wav_cache.get(spk_idx)
                control = speaker_controls.get(spk_idx, "")
                reference_text = speaker_reference_texts.get(spk_idx, "")
                has_control = bool(control)

                logger.info(
                    "Generating segment %d/%d for spk%d: %d chars, "
                    "has_ref=%s, has_control=%s, has_reference_text=%s",
                    segment_index,
                    len(segments),
                    spk_idx,
                    len(segment_text),
                    ref_wav_path is not None,
                    has_control,
                    bool(reference_text),
                )

                final_text = f"({control}){segment_text}" if has_control else segment_text
                generate_kwargs = {
                    "text": final_text,
                    "cfg_value": float(cfg_value),
                    "inference_timesteps": int(inference_steps),
                    "normalize": normalize_text,
                    "denoise": False,
                    "max_len": int(max_len),
                    "retry_badcase": retry_badcase,
                }
                if has_control:
                    if ref_wav_path is not None and is_v2:
                        generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None and reference_text:
                    generate_kwargs["prompt_wav_path"] = ref_wav_path
                    generate_kwargs["prompt_text"] = reference_text
                    if is_v2:
                        generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None and is_v2:
                    # Safe V2 fallback: condition on voice characteristics only.
                    # Do not invent prompt_text through automatic ASR.
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None:
                    raise ValueError(
                        f"spk{spk_idx} uses a VoxCPM v1 reference audio and "
                        f"requires reference_text_{spk_idx}. Automatic ASR is "
                        "intentionally disabled for generation."
                    )
                elif reference_text:
                    raise ValueError(
                        f"reference_text_{spk_idx} was provided without audio_{spk_idx}."
                    )

                step_counter += 1
                pbar.update_absolute(step_counter, total_steps)

                wav_np = _generate_with_quality_guard(
                    voxcpm_model,
                    generate_kwargs,
                    target_text=segment_text,
                    sample_rate=sample_rate,
                    retry_badcase=retry_badcase,
                )

                step_counter += int(inference_steps) + 1
                pbar.update_absolute(step_counter, total_steps)

                generated_segments.append(_normalize_rms(wav_np))

            combined = np.concatenate(generated_segments, axis=-1)
            peak = np.max(np.abs(combined))
            if peak > 0.99:
                combined = combined * (0.99 / peak)
            wav_tensor = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)

            pbar.update_absolute(total_steps, total_steps)
            return ({"waveform": wav_tensor, "sample_rate": sample_rate},)

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass


class RunningHubVoxCPMMultiSpeakerListReference:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"audio_{index}": ("AUDIO",)
            for index in range(1, DEFAULT_REFERENCE_AUDIO_INPUTS + 1)
        }
        optional_inputs.update({
            "normalize_text": ("BOOLEAN", {"default": False}),
            "denoise_reference": ("BOOLEAN", {"default": False}),
            "max_len": ("INT", {
                "default": 4096,
                "min": 64,
                "max": 8192,
                "step": 64,
            }),
            "retry_badcase": ("BOOLEAN", {"default": True}),
            # Keep this after legacy widgets for workflow compatibility.
            "reference_texts": ("STRING", {
                "default": "",
                "multiline": True,
            }),
        })

        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            optional_inputs = _DynamicAudioOptionalInputs(optional_inputs)

        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "script": ("STRING", {
                    "default": "[spk5]今日AI新闻速报\n[spk1]T8那个瓜娃子，又更新什么了？\n[spk2]管他干啥呀，带派不就行了。\n[spk1]天天只知道做视频，耳都不耳人一哈。\n[spk2]别笑哈，你试你也过不了第二关。",
                    "multiline": True,
                }),
                "speaker_controls": ("STRING", {
                    "default": "[spk1]四川话\n[spk2]成年女性，东北话\n[spk5]旁白音，成熟男性",
                    "multiline": True,
                }),
                "cfg_value": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "inference_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "RunningHub/VoxCPM"

    def generate(
        self,
        model,
        script,
        speaker_controls,
        cfg_value,
        inference_steps,
        seed,
        **kwargs,
    ):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        voxcpm_model = model["model"]
        sample_rate = model["sample_rate"]
        is_v2 = model["architecture"] == "voxcpm2"
        script = (script or "").strip()
        speaker_controls = speaker_controls or ""
        cfg_value = float(cfg_value)
        inference_steps = int(inference_steps)
        seed = int(seed)
        normalize_text = kwargs.get("normalize_text", False)
        denoise_reference = kwargs.get("denoise_reference", False)
        max_len = int(kwargs.get("max_len", 4096))
        retry_badcase = kwargs.get("retry_badcase", True)

        segments = _parse_script(script)
        control_map = _parse_speaker_controls(speaker_controls)
        reference_text_map = _parse_reference_texts(
            kwargs.get("reference_texts", "")
        )
        logger.info(
            "Parsed %d segments, %d speaker controls, and %d reference transcripts",
            len(segments),
            len(control_map),
            len(reference_text_map),
        )

        speaker_audios = {}
        for key, audio in kwargs.items():
            if audio is None:
                continue
            match = _AUDIO_INPUT_PATTERN.match(key)
            if not match:
                continue
            speaker_index = int(match.group(1))
            speaker_audios[speaker_index] = audio

        total_steps = len(segments) * (inference_steps + 2) + 1
        pbar = comfy.utils.ProgressBar(total_steps)
        step_counter = 0

        temp_files = []
        generated_segments = []

        try:
            spk_wav_cache = {}
            for spk_idx, audio in speaker_audios.items():
                wav_path = _save_audio_to_temp(audio)
                temp_files.append(wav_path)
                if denoise_reference:
                    denoised = _denoise_audio(wav_path)
                    temp_files.append(denoised)
                    wav_path = denoised
                spk_wav_cache[spk_idx] = wav_path

            for segment_index, (spk_idx, segment_text) in enumerate(segments, 1):
                ref_wav_path = spk_wav_cache.get(spk_idx)
                control = control_map.get(spk_idx, "")
                reference_text = reference_text_map.get(spk_idx, "")
                has_control = bool(control)

                logger.info(
                    "Generating segment %d/%d for spk%d: %d chars, "
                    "has_ref=%s, has_control=%s, has_reference_text=%s",
                    segment_index,
                    len(segments),
                    spk_idx,
                    len(segment_text),
                    ref_wav_path is not None,
                    has_control,
                    bool(reference_text),
                )

                final_text = f"({control}){segment_text}" if has_control else segment_text
                generate_kwargs = {
                    "text": final_text,
                    "cfg_value": cfg_value,
                    "inference_timesteps": inference_steps,
                    "normalize": normalize_text,
                    "denoise": False,
                    "max_len": max_len,
                    "retry_badcase": retry_badcase,
                }

                if has_control:
                    if ref_wav_path is not None and is_v2:
                        generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None and reference_text:
                    generate_kwargs["prompt_wav_path"] = ref_wav_path
                    generate_kwargs["prompt_text"] = reference_text
                    if is_v2:
                        generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None and is_v2:
                    generate_kwargs["reference_wav_path"] = ref_wav_path
                elif ref_wav_path is not None:
                    raise ValueError(
                        f"spk{spk_idx} uses a VoxCPM v1 reference audio and "
                        "requires a matching [spkN] transcript in reference_texts. "
                        "Automatic ASR is intentionally disabled for generation."
                    )
                elif reference_text:
                    raise ValueError(
                        f"reference_texts contains [spk{spk_idx}] but audio_{spk_idx} "
                        "is not connected."
                    )

                step_counter += 1
                pbar.update_absolute(step_counter, total_steps)

                wav_np = _generate_with_quality_guard(
                    voxcpm_model,
                    generate_kwargs,
                    target_text=segment_text,
                    sample_rate=sample_rate,
                    retry_badcase=retry_badcase,
                )

                step_counter += inference_steps + 1
                pbar.update_absolute(step_counter, total_steps)

                generated_segments.append(_normalize_rms(wav_np))

            combined = np.concatenate(generated_segments, axis=-1)
            peak = np.max(np.abs(combined))
            if peak > 0.99:
                combined = combined * (0.99 / peak)
            wav_tensor = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)

            pbar.update_absolute(total_steps, total_steps)
            return ({"waveform": wav_tensor, "sample_rate": sample_rate},)

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
