import tempfile
import unittest
from pathlib import Path
from unittest import mock

import api


class AudioSeparatorModelTest(unittest.TestCase):
    model = "vocals_mel_band_roformer.ckpt"
    config = "vocals_mel_band_roformer.yaml"

    def test_empty_yaml_is_not_a_complete_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / self.model).write_bytes(b"model")
            (model_dir / self.config).write_text("\n", encoding="utf-8")

            self.assertFalse(api._separator_model_files_present(model_dir, self.model))

    def test_prepare_removes_empty_yaml_and_downloads_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / self.model).write_bytes(b"model")
            config_path = model_dir / self.config
            config_path.write_bytes(b"")

            def fake_run(cmd, **kwargs):
                self.assertFalse(config_path.exists())
                config_path.write_text("audio:\n  chunk_size: 352800\n", encoding="utf-8")

            with mock.patch.object(api, "_run_cmd", side_effect=fake_run) as run_cmd:
                api._prepare_separator_model("audio-separator", model_dir, self.model)
                api._prepare_separator_model("audio-separator", model_dir, self.model)

            run_cmd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
