import unittest
from pathlib import Path
from PIL import Image

from src.dataset.imslp import load_imslp, PILImage, Metadata


class TestLoadImslp(unittest.TestCase):
    def test_load_imslp_yields_correct_types(self) -> None:
        manifest_path = Path("data/imslp/imslp.jsonl")
        image_dir = Path("data/imslp/images")
        
        generator = load_imslp(manifest_path, image_dir)
        
        # We use next() to only evaluate the first element of the generator.
        # This prevents the test from loading the entire dataset into memory.
        first_item = next(generator)
        
        self.assertIsInstance(first_item, PILImage)
        
        self.assertIsInstance(first_item.metadata, Metadata)
        self.assertIsInstance(first_item.metadata.score, str)
        self.assertIsInstance(first_item.metadata.page, int)
        self.assertIsInstance(first_item.metadata.name, str)
        
        self.assertIsInstance(first_item.image, Image.Image)


if __name__ == "__main__":
    unittest.main()
