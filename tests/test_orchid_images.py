import tempfile
import unittest
from pathlib import Path

from PIL import Image

from model_compression.src.orchid.images import load_orchid_rgb


class OrchidImageLoadingTests(unittest.TestCase):
    def test_palette_png_with_transparency_normalizes_to_rgb(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "Bti. patula.png"
            image = Image.new("P", (4, 4), color=1)
            image.putpalette([255, 255, 255, 32, 160, 64] + [0] * (768 - 6))
            image.info["transparency"] = 0
            image.save(path, format="PNG")

            decoded = load_orchid_rgb(path)

        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (4, 4))


if __name__ == "__main__":
    unittest.main()
