"""Real HDF5-to-optimizer tests using a small CPU classifier."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bhenet.preprocessing import PreprocessConfig
from bhenet.runtime import capture_rng_state, load_checkpoint, seed_everything
from bhenet.train import evaluate, run_training


class TinyTokenizer:
    def __call__(self, texts, *, return_tensors, padding):
        assert return_tensors == "pt" and padding is True
        ids = torch.tensor([[1, int(text.removeprefix("sample-")) + 3, 2] for text in texts])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(3, 29)
        self.frozen = nn.Parameter(torch.linspace(0.0, 0.1, 29), requires_grad=False)
        self.calls = []

    def forward(self, text, image, *positions):
        self.calls.append((self.training, torch.is_inference_mode_enabled(), int(text[0, 1])))
        features = self.normalization(image).mean(dim=(-2, -1))
        return self.output(self.dropout(features)) + self.frozen


@pytest.fixture
def sample_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    paths = []
    for index in range(6):
        path = tmp_path / f"sample-{index}.h5"
        with h5py.File(path, "w") as handle:
            handle["BF1"] = "A" if index % 2 == 0 else "C"
            handle["BF2"] = "None"
            handle["index"] = index
            handle.create_dataset("pois_infor", data=[f"sample-{index}"], dtype=h5py.string_dtype("utf-8"))
            handle["build_center"] = np.array([0.5, 0.5], dtype=np.float32)
            handle["pois_cor_x"] = np.array([0.1], dtype=np.float32)
            handle["pois_cor_y"] = np.array([0.2], dtype=np.float32)
            handle["x_mat"] = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
            handle["y_mat"] = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
            handle["out_image"] = np.arange(3 * 6 * 8, dtype=np.uint8).reshape(3, 6, 8) + index
            handle["out_label_0"] = np.zeros((1, 6, 8), dtype=np.uint8)
            handle["out_road"] = (np.arange(6 * 8).reshape(1, 6, 8) % 3 == 0).astype(np.uint8)
        paths.append(path)
    return paths[:3], paths[3:]


@pytest.fixture
def preprocessing():
    return PreprocessConfig(image_size=8, grid_size=2, min_tokens=4)


def make_model():
    torch.manual_seed(17)
    return TinyModel()


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_tree_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_training_resume_matches_uninterrupted_optimizer_scheduler_and_rng(sample_files, preprocessing, tmp_path):
    train_files, val_files = sample_files
    tokenizer = TinyTokenizer()
    full_model = make_model()
    initial_frozen = full_model.frozen.detach().clone()
    options = dict(learning_rate=1e-3, eta_min=1e-5, scheduler_t_max=4,
                   preprocess_config=preprocessing, seed=2025)
    full = run_training(full_model, tokenizer, train_files, val_files, tmp_path / "full", epochs=3, **options)
    full_rng = capture_rng_state()
    partial_model = make_model()
    partial = run_training(partial_model, tokenizer, train_files, val_files, tmp_path / "resumed", epochs=1, **options)
    resumed_model = make_model()
    resumed = run_training(resumed_model, tokenizer, train_files, val_files, tmp_path / "resumed",
                           epochs=3, resume=partial["last_checkpoint"], **options)
    assert_tree_equal(full_model.state_dict(), resumed_model.state_dict())
    assert_tree_equal(full_rng, capture_rng_state())
    assert torch.equal(initial_frozen, resumed_model.frozen)
    assert full["global_step"] == resumed["global_step"] == 9
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full["best_macro_recall"] == resumed["best_macro_recall"]
    saved_full = torch.load(full["last_checkpoint"], weights_only=True)
    saved_resumed = torch.load(resumed["last_checkpoint"], weights_only=True)
    assert_tree_equal(saved_full["optimizer"], saved_resumed["optimizer"])
    assert_tree_equal(saved_full["scheduler"], saved_resumed["scheduler"])
    assert saved_resumed["scheduler"]["last_epoch"] == 9
    assert saved_resumed["epoch"] == 3
    assert len(saved_resumed["optimizer"]["param_groups"][0]["params"]) == 4
    rows = [json.loads(line) for line in (tmp_path / "resumed" / "samples.jsonl").read_text().splitlines()]
    for epoch in (1, 2, 3):
        validation = [row for row in rows if row["epoch"] == epoch and row["event"] == "validation_sample"]
        assert [row["identifier"] for row in validation] == ["3", "4", "5"]
    assert [row["epoch"] for row in resumed["epoch_records"]] == [1, 2, 3]
    assert all(not training and inference for training, inference, _ in resumed_model.calls if not training)
    best = load_checkpoint(resumed["best_checkpoint"], make_model())
    assert best["epoch"] == resumed["best_epoch"]


def test_evaluation_restores_rng_modes_and_batchnorm_and_counts_absent_classes(sample_files, preprocessing):
    model = make_model()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[0] = 20
    model.train()
    model.dropout.eval()
    modes = [module.training for module in model.modules()]
    before = {name: value.clone() for name, value in model.named_buffers()}
    seed_everything(31)
    rng = capture_rng_state()
    metrics, records = evaluate(model, TinyTokenizer(), sample_files[1], preprocess_config=preprocessing)
    assert [row["identifier"] for row in records] == ["3", "4", "5"]
    assert metrics["OA29"] == pytest.approx(1 / 3)
    assert metrics["macro_recall"] == pytest.approx(1 / 29)
    assert metrics["represented_macro_recall"] == pytest.approx(1 / 2)
    assert [module.training for module in model.modules()] == modes
    assert_tree_equal(before, dict(model.named_buffers()))
    assert_tree_equal(rng, capture_rng_state())


@pytest.mark.parametrize("failure", ["logits", "gradient"])
def test_nonfinite_values_stop_before_optimizer_update(sample_files, preprocessing, tmp_path, failure):
    class PoisonGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float("nan"))

    class PoisonModel(TinyModel):
        def forward(self, *args):
            value = super().forward(*args)
            return value * float("nan") if failure == "logits" else PoisonGradient.apply(value)

    model = PoisonModel()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(FloatingPointError, match="non-finite"):
        run_training(model, TinyTokenizer(), *sample_files, tmp_path / "bad", epochs=1,
                     preprocess_config=preprocessing)
    assert_tree_equal(before, dict(model.named_parameters()))
    assert not (tmp_path / "bad" / "last.pt").exists()


def test_training_only_run_and_output_protection(sample_files, preprocessing, tmp_path):
    directory = tmp_path / "training"
    result = run_training(make_model(), TinyTokenizer(), sample_files[0], [], directory,
                          epochs=1, preprocess_config=preprocessing)
    assert result["best_checkpoint"] is None
    assert result["epoch_records"][0]["validation"] is None
    assert not (directory / "best.pt").exists()
    with pytest.raises(FileExistsError, match="empty output directory"):
        run_training(make_model(), TinyTokenizer(), sample_files[0], [], directory,
                     epochs=1, preprocess_config=preprocessing)
    with pytest.raises(ValueError, match="metadata mismatch"):
        run_training(make_model(), TinyTokenizer(), sample_files[0], [], directory,
                     epochs=2, learning_rate=3e-5, resume=result["last_checkpoint"],
                     preprocess_config=preprocessing)


def test_resume_rejects_changed_order_and_recovers_epoch_log(sample_files, preprocessing, tmp_path):
    train_files, val_files = sample_files
    directory = tmp_path / "training"
    result = run_training(make_model(), TinyTokenizer(), train_files, val_files, directory,
                          epochs=1, preprocess_config=preprocessing)
    with pytest.raises(ValueError, match="metadata mismatch: train_signature"):
        run_training(make_model(), TinyTokenizer(), list(reversed(train_files)), val_files, directory,
                     epochs=2, preprocess_config=preprocessing, resume=result["last_checkpoint"])
    (directory / "epochs.jsonl").write_text("", encoding="utf-8")
    resumed = run_training(make_model(), TinyTokenizer(), train_files, val_files, directory,
                           epochs=1, preprocess_config=preprocessing, resume=result["last_checkpoint"])
    assert resumed["global_step"] == 3
    assert len(resumed["epoch_records"]) == 1
    assert len(list(directory.glob("recovered_epochs_*.jsonl"))) == 1
    rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
    assert rows == resumed["epoch_records"]


def test_training_checkpoint_to_ordered_unlabelled_predictions(sample_files, preprocessing, tmp_path):
    import csv
    from dataclasses import replace
    import h5py
    from bhenet.predict import run_prediction
    model = make_model()
    trained = run_training(model, TinyTokenizer(), sample_files[0], [], tmp_path / "run",
                           epochs=1, preprocess_config=preprocessing)
    files = list(reversed(sample_files[1]))
    for path in files:
        with h5py.File(path, "r+") as handle:
            handle["BF1"][()] = ""
            handle["BF2"][()] = ""
    output = tmp_path / "predictions.csv"
    result = run_prediction(make_model(), TinyTokenizer(), files, output,
                            checkpoint=trained["last_checkpoint"],
                            preprocess_config=replace(preprocessing, max_tokens=4))
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert result["samples"] == 3
    assert [row["building_id"] for row in rows] == ["5", "4", "3"]
    assert [row["filename"] for row in rows] == [str(path) for path in files]
    for row in rows:
        assert row["BF1p"] + row["BF2p"] == row["predicted_code"]
        scores = [float(value) for name, value in row.items() if name.startswith("score_")]
        assert len(scores) == 29 and sum(scores) == pytest.approx(1)
        assert float(row["softmax_score"]) == max(scores)
    with pytest.raises(FileExistsError):
        run_prediction(model, TinyTokenizer(), files, output,
                       checkpoint=trained["last_checkpoint"], preprocess_config=preprocessing)


def test_prediction_rejects_changed_tokenizer_and_failed_commit(sample_files, preprocessing, tmp_path, monkeypatch):
    import bhenet.predict as prediction
    trained = run_training(make_model(), TinyTokenizer(), sample_files[0], [], tmp_path / "run",
                           epochs=1, preprocess_config=preprocessing)
    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = 999
    with pytest.raises(ValueError, match="tokenizer_signature"):
        prediction.run_prediction(make_model(), tokenizer, sample_files[1], tmp_path / "wrong.csv",
                                  checkpoint=trained["last_checkpoint"], preprocess_config=preprocessing)
    assert not (tmp_path / "wrong.csv").exists()
    def fail_commit(*args):
        raise OSError("simulated CSV commit failure")
    monkeypatch.setattr(prediction.os, "replace", fail_commit)
    with pytest.raises(OSError, match="simulated CSV"):
        prediction.run_prediction(make_model(), TinyTokenizer(), sample_files[1], tmp_path / "failed.csv",
                                  checkpoint=trained["last_checkpoint"], preprocess_config=preprocessing)
    assert not (tmp_path / "failed.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_repairs_best_and_journal_after_last_checkpoint_commit(sample_files, preprocessing, tmp_path, monkeypatch):
    import bhenet.train as engine
    original_evaluate, original_copy = engine.evaluate, engine.copy_checkpoint
    counts = {"evaluation": 0, "copy": 0}

    def improving_evaluation(*args, **kwargs):
        metrics, rows = original_evaluate(*args, **kwargs)
        counts["evaluation"] += 1
        metrics["macro_recall"] = counts["evaluation"] / 29
        return metrics, rows

    def interrupted_copy(*args, **kwargs):
        counts["copy"] += 1
        if counts["copy"] == 2:
            raise OSError("simulated interruption after checkpoint commit")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", improving_evaluation)
    monkeypatch.setattr(engine, "copy_checkpoint", interrupted_copy)
    directory = tmp_path / "interrupted"
    with pytest.raises(OSError, match="simulated interruption"):
        run_training(make_model(), TinyTokenizer(), *sample_files, directory, epochs=2,
                     preprocess_config=preprocessing)
    assert torch.load(directory / "last.pt", weights_only=True)["epoch"] == 2
    assert torch.load(directory / "best.pt", weights_only=True)["epoch"] == 1
    resumed = run_training(make_model(), TinyTokenizer(), *sample_files, directory, epochs=2,
                           preprocess_config=preprocessing, resume=directory / "last.pt")
    assert resumed["global_step"] == 6
    assert resumed["best_epoch"] == 2
    assert counts["evaluation"] == 2
    assert torch.load(directory / "best.pt", weights_only=True)["epoch"] == 2
    rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert len(list(directory.glob("recovered_epochs_*.jsonl"))) == 1
