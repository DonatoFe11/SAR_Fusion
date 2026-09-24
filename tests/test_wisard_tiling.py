import unittest

from PIL import Image

from sarfusion.data.wisard import WiSARDDataset


class TestWiSARDTileBoxes(unittest.TestCase):
    def test_boxes_crossing_each_tile_boundary_keep_only_the_visible_area(self):
        image = Image.new("RGB", (640, 640))
        cases = [
            (1, [300, 100, 80, 40], [0, 100, 60, 40]),
            (2, [50, 300, 40, 80], [50, 0, 40, 60]),
            (3, [300, 300, 80, 80], [0, 0, 60, 60]),
            (0, [270, 50, 80, 40], [270, 50, 50, 40]),
            (0, [50, 270, 40, 80], [50, 270, 40, 50]),
            (0, [50, 70, 40, 80], [50, 70, 40, 80]),
        ]
        for quadrant, bbox, expected in cases:
            with self.subTest(quadrant=quadrant, bbox=bbox):
                annotation = {"bbox": bbox, "category_id": 0, "area": bbox[2] * bbox[3]}
                target = {"image_id": 7, "annotations": [annotation]}
                tile, result = WiSARDDataset._get_tile(None, image, target, quadrant)
                self.assertEqual(tile.size, (320, 320))
                self.assertEqual(result["annotations"][0]["bbox"], expected)
                self.assertEqual(result["annotations"][0]["area"], expected[2] * expected[3])
                self.assertEqual(annotation["bbox"], bbox)
                self.assertEqual(result["image_id"], 7)

    def test_center_rule_assigns_a_boundary_box_to_one_tile(self):
        image = Image.new("RGB", (640, 640))
        counts = []
        for quadrant in range(4):
            target = {"image_id": 0, "annotations": [{"bbox": [300, 300, 40, 40]}]}
            _, result = WiSARDDataset._get_tile(None, image, target, quadrant)
            counts.append(len(result["annotations"]))
        self.assertEqual(counts, [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
