import unittest

import torch
import torch.nn.functional as F

from music_types import Batch, NumPatches, PatchDim, Patches, TensorImage
from transform.core import extract_patches_img, resize_patches_img


class TestResizePatches(unittest.TestCase):
    def test_each_patch_is_its_own_region_resized(self) -> None:
        torch.manual_seed(0)
        ph = pw = 6
        image = torch.rand(2, 3, 2 * ph, 3 * pw)
        patches: Patches[Batch, NumPatches, PatchDim] = extract_patches_img(
            TensorImage(image), (ph, pw)
        )

        resized = resize_patches_img(patches, (8, 8))

        self.assertEqual(resized.image_shape, (3, 16, 24))
        for b in range(2):
            for i in range(patches.data.shape[1]):
                gy, gx = divmod(i, 3)
                region = image[b, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw]
                expected = F.interpolate(
                    region[None], size=(8, 8), mode="bilinear", align_corners=False
                )[0]
                got = resized.data[b, i].view(8, 8, 3).permute(2, 0, 1)
                torch.testing.assert_close(got, expected)

    def test_same_size_is_identity(self) -> None:
        image = torch.rand(1, 3, 12, 12)
        patches: Patches[Batch, NumPatches, PatchDim] = extract_patches_img(
            TensorImage(image), (6, 6)
        )
        self.assertIs(resize_patches_img(patches, (6, 6)), patches)


if __name__ == "__main__":
    unittest.main()
