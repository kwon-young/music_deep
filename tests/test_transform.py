import unittest
from PIL import Image

import transform.ssl as ssl_tf
from dataset.imslp import Data, Metadata
from music_types import SSLSample, PILImage

class TestTransformToNumpy(unittest.TestCase):
    def test_to_numpy_grayscale_has_channel_dim(self) -> None:
        # Create dummy metadata and a Grayscale ("L") PIL image
        metadata = Metadata(score="test", page=1, name="test.tiff")
        pil_img = Image.new("L", (256, 256))
        data = Data(metadata=metadata, sample=SSLSample(image=PILImage(pil_img)))
        
        # Apply the transform
        result = ssl_tf.to_numpy(data)
        
        # Verify the shape is 3D (C, H, W)
        self.assertEqual(result.sample.image.data.shape, (1, 256, 256))
        self.assertEqual(len(result.sample.image.data.shape), 3)

if __name__ == "__main__":
    unittest.main()
