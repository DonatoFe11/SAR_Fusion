import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from sarfusion.models import (
    MODEL_REGISTRY,
    build_fusion_rt_detr_v2,
    build_rtdetr_v2,
)
from sarfusion.utils.reproducibility import (
    RTDETR_V2_FAM_TRAINING_SOURCE_FILES,
    RTDETR_V2_FAM_TRAINING_SOURCE_MANIFEST_ID,
    build_training_source_manifest,
    prepare_rtdetr_model_for_determinism,
)
from sarfusion.utils.utils import load_yaml
from sarfusion.experiment.run import Run
from sarfusion.models.checkpoints import complete_shared_state_dict_aliases


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_v2_additive_sequence_validation_seed40.yaml"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_v2_fam_sequence_validation_seed40.yaml"
)
PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_v2_fam_runtime_probe.yaml"
)
CHECKPOINT = "PekingU/rtdetr_v2_r50vd"
REVISION = "282494075698cab9faa1096ae26856890030c817"
HAS_RTDETR_V2 = (
    importlib.util.find_spec("transformers.models.rt_detr_v2") is not None
)

if HAS_RTDETR_V2:
    from transformers import RTDetrV2ForObjectDetection
    from transformers.models.rt_detr.configuration_rt_detr_resnet import (
        RTDetrResNetConfig,
    )
    from transformers.models.rt_detr_v2.configuration_rt_detr_v2 import (
        RTDetrV2Config,
    )

    from sarfusion.models.rtdetr_v2_fusion import (
        HISTORICAL_FAM_INITIALIZATION,
        RTDetrV2FusionBackbone,
        RTDetrV2FusionForObjectDetection,
    )
    from sarfusion.models.rtdetr_fusion import (
        FeatureAlignmentModule,
        copy_matching_pretrained_label_heads,
    )


def tiny_rtdetr_v2_config(num_labels=1, num_denoising=0):
    backbone = RTDetrResNetConfig(
        num_channels=3,
        embedding_size=8,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="bottleneck",
        downsample_in_first_stage=False,
        out_indices=[2, 3, 4],
    )
    labels = ["person", "car"][:num_labels]
    return RTDetrV2Config(
        backbone_config=backbone,
        encoder_hidden_dim=32,
        encoder_in_channels=[32, 64, 128],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        encode_proj_layers=[2],
        d_model=32,
        num_queries=10,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        num_feature_levels=3,
        decoder_n_levels=3,
        decoder_n_points=2,
        decoder_layers=1,
        decoder_attention_heads=4,
        num_denoising=num_denoising,
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
    )


class TestRTDetrV2StaticContract(unittest.TestCase):
    def test_public_registry_and_factories_are_separate_from_v1(self):
        self.assertIs(MODEL_REGISTRY["rtdetr_v2"], build_rtdetr_v2)
        self.assertIs(
            MODEL_REGISTRY["fusion_rtdetr_v2"], build_fusion_rt_detr_v2
        )
        signature = inspect.signature(build_fusion_rt_detr_v2)
        self.assertIn("pretrained_model_name", signature.parameters)
        self.assertIn("pretrained_revision", signature.parameters)
        self.assertIn("use_fam", signature.parameters)
        self.assertIn("fam_initialization", signature.parameters)
        standard_signature = inspect.signature(build_rtdetr_v2)
        self.assertIn(
            "reuse_pretrained_class_head", standard_signature.parameters
        )

    def test_v2_source_manifest_is_separate_and_complete(self):
        manifest = build_training_source_manifest(
            RTDETR_V2_FAM_TRAINING_SOURCE_MANIFEST_ID
        )
        self.assertEqual(
            manifest["manifest_id"],
            RTDETR_V2_FAM_TRAINING_SOURCE_MANIFEST_ID,
        )
        expected_files = {
            "environment.yml",
            "requirements-rtdetrv2.txt",
            "main.py",
            *(
                path.relative_to(REPO_ROOT).as_posix()
                for path in (REPO_ROOT / "sarfusion").rglob("*.py")
            ),
        }
        self.assertEqual(
            set(manifest["files"]),
            expected_files,
        )
        self.assertTrue(
            set(RTDETR_V2_FAM_TRAINING_SOURCE_FILES).issubset(
                expected_files
            )
        )
        self.assertIn(
            "sarfusion/models/rtdetr_v2_fusion.py", manifest["files"]
        )
        self.assertIn(
            "sarfusion/experiment/modality_consistency.py",
            manifest["files"],
        )
        self.assertIn(
            "sarfusion/experiment/box_guided_alignment.py",
            manifest["files"],
        )
        self.assertIn("requirements-rtdetrv2.txt", manifest["files"])
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_seed40_protocols_are_matched_except_fam_identity(self):
        control = load_yaml(CONTROL_PATH)
        candidate = load_yaml(CANDIDATE_PATH)

        self.assertEqual(control["parameters"]["seed"], [40])
        self.assertEqual(candidate["parameters"]["seed"], [40])
        for protocol in (control, candidate):
            params = protocol["parameters"]
            self.assertFalse(params["run_test"][0])
            self.assertEqual(params["strict_checkpoint_loading"], [True])
            self.assertEqual(params["train"]["max_epochs"], [10])
            self.assertNotIn("early_stopping_patience", params["train"])
            self.assertEqual(params["train"]["watch_metric"], ["map_50"])
            self.assertEqual(params["dataloader"]["batch_size"], [4])
            self.assertEqual(
                params["dataset"]["preprocessor"]["path"], [CHECKPOINT]
            )
            self.assertEqual(
                params["dataset"]["preprocessor"]["revision"],
                [REVISION],
            )
            self.assertEqual(
                params["dataset"]["preprocessor"]["use_fast"],
                [False],
            )
            model = params["model"]
            self.assertEqual(model["name"], ["fusion_rtdetr_v2"])
            self.assertEqual(
                model["params"]["pretrained_model_name"], [CHECKPOINT]
            )
            self.assertEqual(
                model["params"]["pretrained_revision"], [REVISION]
            )
            self.assertEqual(
                model["params"]["fam_variant"], ["current_dcnv2"]
            )
            self.assertEqual(
                model["params"]["fam_initialization"],
                ["historical_hf_post_init"],
            )

        control_params = control["parameters"]
        candidate_params = candidate["parameters"]
        control_model = control_params["model"]["params"]
        candidate_model = candidate_params["model"]["params"]
        self.assertEqual(control_model["use_fam"], [False])
        self.assertEqual(candidate_model["use_fam"], [True])

        # Remove the one scientific factor plus metadata that identifies each
        # run; every actual training/data setting must then be identical.
        control_model = dict(control_model)
        candidate_model = dict(candidate_model)
        control_model.pop("use_fam")
        candidate_model.pop("use_fam")
        self.assertEqual(control_model, candidate_model)
        control_rest = deepcopy(control_params)
        candidate_rest = deepcopy(candidate_params)
        for key in ("model", "tracker"):
            control_rest.pop(key)
            candidate_rest.pop(key)
        self.assertEqual(control_rest, candidate_rest)

    def test_probe_is_two_runs_and_cannot_save_scientific_checkpoints(self):
        probe = load_yaml(PROBE_PATH)
        params = probe["parameters"]
        self.assertEqual(params["model"]["params"]["use_fam"], [False, True])
        self.assertEqual(params["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(params["train"]["max_epochs"], [1])
        self.assertEqual(params["train"]["save_checkpoints"], [False])
        self.assertIn("ExcludeFromCampaign", params["tracker"]["tags"][0])

    def test_v2_checkpoint_restore_requests_strict_key_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "best"
            checkpoint_dir.mkdir()
            save_file(
                {"model.weight": torch.ones(2, 2)},
                checkpoint_dir / "model.safetensors",
            )

            run = object.__new__(Run)
            run.tracker = SimpleNamespace(local_dir=directory)
            run.params = {"strict_checkpoint_loading": True}
            run.model = MagicMock()
            run.model.load_state_dict.return_value = SimpleNamespace(
                missing_keys=[], unexpected_keys=[]
            )

            run.restore_model("best")

            _, kwargs = run.model.load_state_dict.call_args
            self.assertIs(kwargs["strict"], True)

    def test_safetensors_alias_completion_is_exact_and_strict(self):
        tied = torch.nn.Module()
        tied.primary = torch.nn.Linear(3, 2)
        tied.alias = tied.primary
        state = tied.state_dict()
        serialized_once = {
            key: value.clone()
            for key, value in state.items()
            if key.startswith("primary.")
        }

        completed, restored = complete_shared_state_dict_aliases(
            tied, serialized_once
        )

        self.assertEqual(
            restored,
            {
                "alias.weight": "primary.weight",
                "alias.bias": "primary.bias",
            },
        )
        tied.load_state_dict(completed, strict=True)

        incomplete = {"primary.weight": state["primary.weight"].clone()}
        completed, _ = complete_shared_state_dict_aliases(tied, incomplete)
        with self.assertRaises(RuntimeError):
            tied.load_state_dict(completed, strict=True)

    @unittest.skipIf(HAS_RTDETR_V2, "Only relevant in the historical env")
    def test_historical_environment_fails_with_an_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "Transformers >=4.49"):
            build_fusion_rt_detr_v2(id2label={0: "person"})


@unittest.skipUnless(HAS_RTDETR_V2, "Requires the RT-DETRv2 environment")
class TestRTDetrV2ModelIntegration(unittest.TestCase):
    def test_matching_head_transfer_supports_disabled_denoising(self):
        source = RTDetrV2ForObjectDetection(
            tiny_rtdetr_v2_config(num_labels=2, num_denoising=2)
        )
        target = RTDetrV2ForObjectDetection(
            tiny_rtdetr_v2_config(num_labels=1, num_denoising=0)
        )
        copied = copy_matching_pretrained_label_heads(
            target, source, target.config.id2label
        )
        self.assertEqual(copied, [0])
        torch.testing.assert_close(
            target.class_embed[0].weight[0],
            source.class_embed[0].weight[0],
        )

    def test_current_fam_reproduces_effective_historical_initialization(self):
        raw_fam = FeatureAlignmentModule(8)
        self.assertEqual(
            torch.count_nonzero(raw_fam.offset_conv.weight).item(), 0
        )

        config = tiny_rtdetr_v2_config()
        config.num_channels = 4
        model = RTDetrV2FusionForObjectDetection(config, use_fam=True)
        self.assertEqual(
            model.fam_initialization, HISTORICAL_FAM_INITIALIZATION
        )
        for fam in model.model.backbone.fam_modules:
            self.assertGreater(
                torch.count_nonzero(fam.offset_conv.weight).item(), 0
            )

        with self.assertRaisesRegex(ValueError, "supports only"):
            RTDetrV2FusionForObjectDetection(
                config,
                use_fam=True,
                fam_initialization="implicit_or_unknown",
            )

    def test_dual_backbone_and_full_detector_forward(self):
        config = tiny_rtdetr_v2_config()
        config.num_channels = 4
        backbone = RTDetrV2FusionBackbone(config, use_fam=True)
        backbone.eval()
        pixels = torch.randn(1, 4, 64, 64)
        mask = torch.ones(1, 64, 64, dtype=torch.bool)

        with torch.no_grad():
            features = backbone(pixels, mask)

        self.assertEqual(len(features), 3)
        self.assertEqual(len(backbone.fam_modules), 3)
        self.assertEqual(
            [tuple(feature.shape) for feature, _ in features],
            [
                (1, 32, 8, 8),
                (1, 64, 4, 4),
                (1, 128, 2, 2),
            ],
        )

        detector = RTDetrV2FusionForObjectDetection(config, use_fam=True)
        detector.eval()
        with torch.no_grad():
            output = detector(pixels, pixel_mask=mask)
        self.assertEqual(tuple(output.logits.shape), (1, 10, 1))
        self.assertEqual(tuple(output.pred_boxes.shape), (1, 10, 4))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.pred_boxes).all())

    def test_training_uses_rtdetr_v2_loss_contract(self):
        config = tiny_rtdetr_v2_config(num_denoising=2)
        config.num_channels = 4
        model = RTDetrV2FusionForObjectDetection(config, use_fam=True)
        model.train()
        output = model(
            torch.randn(1, 4, 128, 128),
            pixel_mask=torch.ones(1, 128, 128, dtype=torch.bool),
            labels=[
                {
                    "class_labels": torch.tensor([0], dtype=torch.long),
                    "boxes": torch.tensor(
                        [[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32
                    ),
                }
            ],
        )
        self.assertEqual(model.loss_type, "RTDetrV2ForObjectDetection")
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for fam in model.model.backbone.fam_modules
                for parameter in fam.parameters()
            )
        )

    def test_all_fam_levels_are_eager_trainable_and_receive_gradients(self):
        config = tiny_rtdetr_v2_config()
        config.num_channels = 4
        backbone = RTDetrV2FusionBackbone(config, use_fam=True)
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in backbone.named_parameters()
            if "fam_modules" in name
        }
        self.assertTrue(parameters_before)
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-3)

        features = backbone(
            torch.randn(2, 4, 64, 64),
            torch.ones(2, 64, 64, dtype=torch.bool),
        )
        loss = sum(feature.square().mean() for feature, _ in features)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for level, fam in enumerate(backbone.fam_modules):
            gradients = [
                parameter.grad
                for parameter in fam.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients, f"FAM level {level} has no parameters")
            self.assertTrue(
                all(gradient is not None for gradient in gradients),
                f"FAM level {level} has missing gradients",
            )
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )

        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(
                    parameter.detach(), parameters_before[name]
                )
                for name, parameter in backbone.named_parameters()
                if name in parameters_before
            )
        )

    def test_pretrained_transfer_preserves_rng_and_shared_initialization(self):
        base_config = tiny_rtdetr_v2_config(
            num_labels=2, num_denoising=2
        )
        torch.manual_seed(17)
        base = RTDetrV2ForObjectDetection(base_config)

        def build(use_fam):
            with patch.object(
                RTDetrV2ForObjectDetection,
                "from_pretrained",
                return_value=base,
            ):
                return RTDetrV2FusionForObjectDetection.from_pretrained(
                    CHECKPOINT,
                    id2label={0: "person"},
                    label2id={"person": 0},
                    use_fam=use_fam,
                    reuse_pretrained_class_head=True,
                    revision=REVISION,
                )

        torch.manual_seed(40)
        initial_rng = torch.random.get_rng_state().clone()
        control = build(False)
        control_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(initial_rng)
        candidate = build(True)
        candidate_rng = torch.random.get_rng_state().clone()

        torch.testing.assert_close(control_rng, initial_rng)
        torch.testing.assert_close(candidate_rng, initial_rng)
        control_state = control.state_dict()
        candidate_state = candidate.state_dict()
        shared_keys = sorted(set(control_state) & set(candidate_state))
        self.assertTrue(shared_keys)
        for key in shared_keys:
            with self.subTest(key=key):
                torch.testing.assert_close(
                    candidate_state[key], control_state[key], atol=0, rtol=0
                )

        report = candidate.pretrained_transfer_report
        self.assertEqual(report["revision"], REVISION)
        self.assertGreater(report["directly_transferred_tensors"], 0)
        self.assertGreater(report["rgb_backbone_tensors"], 0)
        self.assertEqual(
            report["rgb_backbone_tensors"], report["ir_backbone_tensors"]
        )

        clone = RTDetrV2FusionForObjectDetection(
            candidate.config, use_fam=True
        )
        clone.load_state_dict(candidate.state_dict(), strict=True)

    def test_deterministic_mode_fails_closed_for_v2(self):
        config = tiny_rtdetr_v2_config()
        config.num_channels = 4
        model = RTDetrV2FusionForObjectDetection(config, use_fam=True)
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            prepare_rtdetr_model_for_determinism(model)


if __name__ == "__main__":
    unittest.main()
