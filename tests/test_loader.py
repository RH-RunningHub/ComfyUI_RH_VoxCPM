import os
import sys
import tempfile
import types
import unittest

_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.models_dir = ""
_FOLDER_PATHS = {}


def _set_paths(mapping):
    _FOLDER_PATHS.clear()
    _FOLDER_PATHS.update(mapping)


_folder_paths_stub.get_folder_paths = lambda model_type="": list(
    _FOLDER_PATHS.get(model_type, [])
)
_folder_paths_stub.add_model_folder_path = lambda *_args, **_kwargs: None
sys.modules.setdefault("folder_paths", _folder_paths_stub)

import nodes.loader as loader  # noqa: E402

# Rebind in case another test module already imported nodes.loader with a
# different folder_paths stub in sys.modules.
loader.folder_paths = _folder_paths_stub


class LoaderLoraFallbackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = self._tmp.name
        self.dedicated = os.path.join(base, "voxcpm_lora")
        self.generic = os.path.join(base, "loras")
        os.makedirs(self.dedicated, exist_ok=True)
        os.makedirs(self.generic, exist_ok=True)

    def tearDown(self):
        _set_paths({})
        self._tmp.cleanup()

    def _paths(self):
        _set_paths({"voxcpm_lora": [self.dedicated], "loras": [self.generic]})

    def test_resolve_lora_falls_back_to_generic_loras_dir(self):
        name = "voxcpm-zhenzhen-3000.safetensors"
        full = os.path.join(self.generic, name)
        with open(full, "wb") as f:
            f.write(b"x")
        self._paths()
        self.assertEqual(loader._resolve_lora_path(name), full)

    def test_resolve_lora_prefers_dedicated_dir(self):
        name = "voxcpm-dual.safetensors"
        for base in (self.dedicated, self.generic):
            with open(os.path.join(base, name), "wb") as f:
                f.write(b"x")
        self._paths()
        self.assertEqual(
            loader._resolve_lora_path(name),
            os.path.join(self.dedicated, name),
        )

    def test_resolve_lora_missing_everywhere_returns_none(self):
        self._paths()
        self.assertIsNone(loader._resolve_lora_path("voxcpm-absent.safetensors"))

    def test_resolve_lora_none_and_unsafe_names(self):
        self._paths()
        self.assertIsNone(loader._resolve_lora_path(None))
        self.assertIsNone(loader._resolve_lora_path("None"))
        with self.assertRaises(ValueError):
            loader._resolve_lora_path("../escape.safetensors")

    def test_list_lora_files_surfaces_voxcpm_named_from_generic_dir(self):
        with open(os.path.join(self.generic, "voxcpm-zhenzhen-3000.safetensors"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(self.generic, "sdxl-style.safetensors"), "wb") as f:
            f.write(b"x")
        voice_dir = os.path.join(self.generic, "voxcpm-voice")
        os.makedirs(voice_dir, exist_ok=True)
        with open(os.path.join(voice_dir, "lora_weights.safetensors"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(self.dedicated, "my-voice.pth"), "wb") as f:
            f.write(b"x")
        self._paths()
        self.assertEqual(
            loader._list_lora_files(),
            ["None", "my-voice.pth", "voxcpm-voice", "voxcpm-zhenzhen-3000.safetensors"],
        )

    def test_list_lora_files_dedupes_across_dir_types(self):
        name = "voxcpm-dual.safetensors"
        for base in (self.dedicated, self.generic):
            with open(os.path.join(base, name), "wb") as f:
                f.write(b"x")
        self._paths()
        self.assertEqual(loader._list_lora_files(), ["None", name])


if __name__ == "__main__":
    unittest.main()
