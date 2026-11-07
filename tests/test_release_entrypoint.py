from pathlib import Path
import tempfile
import unittest

from src.main import recipe_arguments


class RecipeTests(unittest.TestCase):
    def test_arguments_preserve_spaces_without_shell_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text('{"cli_args": ["--edit-anchor-prefix", "High-level safety framing:"]}')
            self.assertEqual(recipe_arguments(path), ["--edit-anchor-prefix", "High-level safety framing:"])

    def test_non_string_arguments_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text('{"cli_args": ["--n", 4]}')
            with self.assertRaises(ValueError):
                recipe_arguments(path)
