"""The job `audio` field (handler.py): format and length checks, the QwenVL
duration choice, and the R2VA graph wiring. Runs without ComfyUI or RunPod:

    python -m unittest discover -s tests

R2VA_JSON=<path to MiniMaxH3-Singularity-R2VA.json from ForgeHub> also runs the
real workflow through _patch.
"""
import base64
import copy
import io
import json
import os
import sys
import types
import unittest
import wave
from pathlib import Path

sys.modules.setdefault("runpod", types.ModuleType("runpod"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import handler  # noqa: E402


def wav(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


def b64(raw):
    return base64.b64encode(raw).decode()


# The R2VA wiring that matters (as in ForgeHub's MiniMaxH3-Singularity-R2VA.json):
# R2VA -> BasicGuider (conditioning) and SamplerCustomAdvanced (latent);
# VAEDecodeAudio -> CreateVideo and VHS_VideoCombine; seconds via PrimitiveFloat;
# the QwenVL enhancer's duration choice.
R2VA = {
    "111": {"class_type": "PrimitiveFloat", "inputs": {"value": 10}},
    "107": {"class_type": "ComfyMathExpression", "inputs": {"expression": "a", "values.a": ["111", 0]}},
    "24": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    "276": {"class_type": "QwenVL_Unified", "inputs": {"prompt": "x", "duration": "5s", "seed": 1}},
    "224": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "prompt": ["276", 0], "length": ["107", 1], "audio_vae": ["24", 0], "ref_image_size": "max"}},
    "16": {"class_type": "BasicGuider", "inputs": {"conditioning": ["224", 0]}},
    "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["16", 0], "latent_image": ["224", 1], "noise": ["15", 0]}},
    "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 5}},
    "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
    "91": {"class_type": "CreateVideo", "inputs": {"fps": 24, "audio": ["23", 0]}},
    "126": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 48, "audio": ["23", 0], "save_output": True}},
}


class Checks(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(handler._audio_ext(wav(1)), "wav")
        self.assertEqual(handler._audio_ext(b"fLaC" + b"\0" * 12), "flac")
        self.assertEqual(handler._audio_ext(b"OggS" + b"\0" * 12), "ogg")
        self.assertEqual(handler._audio_ext(b"ID3\x04" + b"\0" * 12), "mp3")
        self.assertEqual(handler._audio_ext(b"\xff\xfb\x90\x00" + b"\0" * 12), "mp3")
        self.assertEqual(handler._audio_ext(b"\0\0\0\x20ftypM4A " + b"\0" * 4), "m4a")
        with self.assertRaisesRegex(ValueError, "unknown format"):
            handler._audio_ext(b"<html>" + b"\0" * 12)

    def test_wav_length(self):
        self.assertAlmostEqual(handler._audio_seconds(wav(3.5), "wav"), 3.5)

    def test_prepare(self):
        job = {"audio": "data:audio/wav;base64," + b64(wav(7.6))}
        raw, ext, secs = handler._prepare_audio(job)
        self.assertEqual((ext, round(secs, 2)), ("wav", 7.6))
        self.assertIs(handler._prepare_audio(job), job["_audio"])  # fetched once
        self.assertIsNone(handler._prepare_audio({}))
        for s in (1.5, 15.5):
            with self.assertRaisesRegex(ValueError, "must be 2-15 s"):
                handler._prepare_audio({"audio": b64(wav(s))})
        handler._prepare_audio({"audio": b64(wav(2))})
        handler._prepare_audio({"audio": b64(wav(15))})
        with self.assertRaisesRegex(ValueError, "audio_mode"):
            handler._prepare_audio({"audio": b64(wav(3)), "audio_mode": "dub"})
        with self.assertRaisesRegex(ValueError, "neither a link nor base64"):
            handler._prepare_audio({"audio": "not base64!"})

    def test_qwen_duration(self):
        for secs, want in ((2, "5s"), (5, "5s"), (5.1, "10s"), (7.6, "10s"), (15, "15s"), (16, "20s"), (30, "20s")):
            self.assertEqual(handler._qwen_duration(secs), want, secs)


class Wiring(unittest.TestCase):
    def test_lipsync(self):
        g = copy.deepcopy(R2VA)
        load = handler._inject_audio(g, "j_audio.wav")
        guide = next(k for k, n in g.items() if n["class_type"] == "MiniMaxH3AddGuide")
        self.assertEqual(g[load], {"class_type": "LoadAudio", "inputs": {"audio": "j_audio.wav"}, "_meta": {"title": "Job audio"}})
        self.assertEqual(g[guide]["inputs"], {"positive": ["224", 0], "latent": ["224", 1], "audio_vae": ["24", 0], "audio": [load, 0], "frame_idx": 0})
        self.assertEqual(g["16"]["inputs"]["conditioning"], [guide, 0])
        self.assertEqual(g["14"]["inputs"]["latent_image"], ["224", 1])  # the latent stays
        self.assertEqual(g["126"]["inputs"]["audio"], [load, 0])  # keep_audio
        self.assertEqual(g["91"]["inputs"]["audio"], [load, 0])

    def test_lipsync_decoded_audio(self):
        for kw in ({"keep": False}, {"start_frame": 24}):
            g = copy.deepcopy(R2VA)
            handler._inject_audio(g, "a.wav", **kw)
            self.assertEqual(g["126"]["inputs"]["audio"], ["23", 0], kw)
        g = copy.deepcopy(R2VA)
        handler._inject_audio(g, "a.wav", start_frame=24)
        self.assertEqual(next(n for n in g.values() if n["class_type"] == "MiniMaxH3AddGuide")["inputs"]["frame_idx"], 24)

    def test_reference(self):
        g = copy.deepcopy(R2VA)
        a = handler._inject_audio(g, "a.wav", "reference")
        b = handler._inject_audio(g, "b.wav", "reference")
        self.assertEqual(g["224"]["inputs"]["ref_audios.ref_audio_0"], [a, 0])
        self.assertEqual(g["224"]["inputs"]["ref_audios.ref_audio_1"], [b, 0])
        self.assertNotEqual(a, b)
        self.assertFalse(any(n["class_type"] == "MiniMaxH3AddGuide" for n in g.values()))
        self.assertEqual(g["16"]["inputs"]["conditioning"], ["224", 0])
        self.assertEqual(g["126"]["inputs"]["audio"], ["23", 0])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "R2VA workflow"):
            handler._inject_audio({"1": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {}}}, "a.wav")
        g = copy.deepcopy(R2VA)
        del g["224"]["inputs"]["audio_vae"]
        with self.assertRaisesRegex(ValueError, "audio_vae"):
            handler._inject_audio(g, "a.wav")


class Patch(unittest.TestCase):
    """The whole _patch, uploads stubbed."""

    def setUp(self):
        self.uploads = []
        self._upload = handler._upload
        handler._upload = lambda name, data: (self.uploads.append((name, len(data))), name)[1]

    def tearDown(self):
        handler._upload = self._upload

    def test_audio_sets_the_length(self):
        g = handler._patch(copy.deepcopy(R2VA), {"prompt": "she sings", "audio": b64(wav(7.6)), "job_name": "t1"})
        self.assertEqual(self.uploads, [("t1_audio.wav", len(wav(7.6)))])
        self.assertAlmostEqual(g["111"]["inputs"]["value"], 7.6)
        self.assertEqual(g["276"]["inputs"]["duration"], "10s")
        self.assertTrue(any(n["class_type"] == "MiniMaxH3AddGuide" for n in g.values()))

    def test_seconds_win(self):
        g = handler._patch(copy.deepcopy(R2VA), {"prompt": "x", "audio": b64(wav(7.6)), "seconds": 5, "job_name": "t2"})
        self.assertEqual(g["111"]["inputs"]["value"], 5.0)
        self.assertEqual(g["276"]["inputs"]["duration"], "5s")

    def test_seconds_without_audio(self):
        g = handler._patch(copy.deepcopy(R2VA), {"prompt": "x", "seconds": 8})
        self.assertEqual(g["111"]["inputs"]["value"], 8.0)
        self.assertEqual(g["276"]["inputs"]["duration"], "10s")  # "8s" isn't a QwenVL choice
        self.assertEqual(self.uploads, [])
        self.assertFalse(any(n["class_type"] in ("LoadAudio", "MiniMaxH3AddGuide") for n in g.values()))

    def test_seconds_leave_a_text_prompt(self):
        g = {"5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "old"}},
             "6": {"class_type": "QwenVL_Unified", "inputs": {"prompt": ["5", 0], "duration": "5s"}}}
        g = handler._patch(g, {"prompt": "a cat", "seconds": 6})
        self.assertEqual(g["5"]["inputs"]["value"], "a cat")
        self.assertEqual(g["6"]["inputs"]["duration"], "10s")

    @unittest.skipUnless(os.environ.get("R2VA_JSON"), "R2VA_JSON not set")
    def test_forgehub_r2va(self):
        wf = json.loads(Path(os.environ["R2VA_JSON"]).read_text(encoding="utf-8"))
        g = handler._patch(wf, {"prompt": "she sings", "audio": b64(wav(9.2)), "images": [b64(b"png")], "job_name": "t3",
                                "upscale": False, "rife": False})
        r2v = next(k for k, n in g.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
        guide = next(k for k, n in g.items() if n["class_type"] == "MiniMaxH3AddGuide")
        load = next(k for k, n in g.items() if n["class_type"] == "LoadAudio")
        self.assertEqual(g[guide]["inputs"]["positive"], [r2v, 0])
        users = [(k, i) for k, n in g.items() for i, v in n["inputs"].items() if v == [r2v, 0]]
        self.assertEqual(users, [(guide, "positive")])  # every other user takes the guide's
        self.assertTrue(any(v == [guide, 0] for n in g.values() for v in n["inputs"].values()))
        saver = next(n for n in g.values() if n["class_type"] == "VHS_VideoCombine")
        self.assertEqual(saver["inputs"]["audio"], [load, 0])
        dur = next(n for n in g.values() if n["class_type"] == "PrimitiveFloat")
        self.assertAlmostEqual(dur["inputs"]["value"], 9.2)
        qwen = next(n for n in g.values() if n["class_type"] == "QwenVL_Unified")
        self.assertEqual(qwen["inputs"]["duration"], "10s")
        # every link points at a node that exists
        for k, n in g.items():
            for i, v in n["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                    self.assertIn(str(v[0]), g, f"{k}.{i} -> {v}")


if __name__ == "__main__":
    unittest.main()
