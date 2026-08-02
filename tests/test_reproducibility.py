import json
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sarfusion.utils.reproducibility import (
    ReproducibilityTrace,
    deterministic_grid_sample_bilinear,
    state_dict_digest,
)
from sarfusion.models.rtdetr_fusion import copy_matching_pretrained_label_heads


class TestDeterministicRTDetrSampling(unittest.TestCase):
    def test_forward_and_gradients_match_native_grid_sample(self):
        torch.manual_seed(7)
        native_input = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
        native_grid = (torch.rand(2, 4, 6, 2, dtype=torch.float64) * 2.4 - 1.2).requires_grad_()
        deterministic_input = native_input.detach().clone().requires_grad_()
        deterministic_grid = native_grid.detach().clone().requires_grad_()

        expected = F.grid_sample(
            native_input,
            native_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        actual = deterministic_grid_sample_bilinear(
            deterministic_input, deterministic_grid
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(
            deterministic_input.grad, native_input.grad, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            deterministic_grid.grad, native_grid.grad, atol=1e-11, rtol=1e-11
        )

    def test_state_dict_digest_changes_with_a_weight(self):
        state = {
            "weight": torch.arange(4, dtype=torch.float32),
            "scalar_buffer": torch.tensor(3, dtype=torch.int64),
        }
        before = state_dict_digest(state)
        state["weight"][2] += 1
        self.assertNotEqual(before, state_dict_digest(state))

    def test_trace_is_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/trace.jsonl"
            trace = ReproducibilityTrace(path)
            trace.write("batch", epoch=0, sample_indices=[3, 1])
            with open(path, encoding="utf-8") as file:
                record = json.loads(file.readline())
            self.assertEqual(record["event"], "batch")
            self.assertEqual(record["sample_indices"], [3, 1])

    def test_matching_pretrained_person_rows_are_reused(self):
        def detector(labels):
            count = len(labels)
            model = SimpleNamespace(
                enc_score_head=torch.nn.Linear(3, count),
                denoising_class_embed=torch.nn.Embedding(
                    count + 1, 3, padding_idx=count
                ),
            )
            return SimpleNamespace(
                config=SimpleNamespace(id2label=dict(enumerate(labels))),
                class_embed=torch.nn.ModuleList(
                    [torch.nn.Linear(3, count) for _ in range(2)]
                ),
                model=model,
            )

        source = detector(["person", "car"])
        target = detector(["person"])
        with torch.no_grad():
            for layer_index, layer in enumerate(source.class_embed):
                layer.weight[0].fill_(10 + layer_index)
                layer.bias[0].fill_(20 + layer_index)
            source.model.enc_score_head.weight[0].fill_(30)
            source.model.enc_score_head.bias[0].fill_(31)
            source.model.denoising_class_embed.weight[0].fill_(40)
            source.model.denoising_class_embed.weight[2].fill_(41)

        copied = copy_matching_pretrained_label_heads(
            target, source, {0: "person"}
        )

        self.assertEqual(copied, [0])
        for layer_index, layer in enumerate(target.class_embed):
            torch.testing.assert_close(
                layer.weight[0], torch.full((3,), 10.0 + layer_index)
            )
            torch.testing.assert_close(
                layer.bias[0], torch.tensor(20.0 + layer_index)
            )
        torch.testing.assert_close(
            target.model.enc_score_head.weight[0], torch.full((3,), 30.0)
        )
        torch.testing.assert_close(
            target.model.denoising_class_embed.weight[0], torch.full((3,), 40.0)
        )
        torch.testing.assert_close(
            target.model.denoising_class_embed.weight[1], torch.full((3,), 41.0)
        )


if __name__ == "__main__":
    unittest.main()
