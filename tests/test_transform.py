import unittest
from PIL import Image

from transform import to_numpy
from dataset.imslp import Data, Metadata

class TestTransformToNumpy(unittest.TestCase):
    def test_to_numpy_grayscale_has_channel_dim(self) -> None:
        # Create dummy metadata and a Grayscale ("L") PIL image
        metadata = Metadata(score="test", page=1, name="test.tiff")
        pil_img = Image.new("L", (256, 256))
        data = Data(metadata=metadata, image=pil_img)
        
        # Apply the transform
        result = to_numpy(data)
        
        # Verify the shape is 3D (C, H, W)
        self.assertEqual(result.image.shape, (1, 256, 256))
        self.assertEqual(len(result.image.shape), 3)

if __name__ == "__main__":
    unittest.main()
