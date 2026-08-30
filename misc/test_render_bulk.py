"""Bulk rendering: the loop around the per-episode renderer.

What is worth testing here is not the drawing (test_render covers that) but the batch's promises:
a bad episode does not end the run, a killed run resumes, and the zip holds what was rendered.
"""

import pathlib
import sys
import types
import zipfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from misc.render_bulk import parse_episodes  # noqa: E402
from misc.render_bulk import zip_folder  # noqa: E402


@pytest.mark.parametrize(
    ("spec", "expect"),
    [
        ("all", [0, 1, 2, 3, 4]),
        ("3", [3]),
        ("0-2", [0, 1, 2]),
        ("0,3,4", [0, 3, 4]),
        ("1-2,4", [1, 2, 4]),
        ("4,4,4", [4]),  # a repeated index is one render, not four
    ],
)
def test_episode_specs(spec, expect):
    assert parse_episodes(spec, 5) == expect


def test_out_of_range_is_an_error_not_a_trim():
    """Asking for 0-49 of a 20-episode dataset means one of the two numbers is wrong. Quietly
    rendering 20 of them hides which, and the batch looks complete."""
    with pytest.raises(SystemExit, match="out of range"):
        parse_episodes("0-9", 5)


def test_zip_holds_the_renders_flat_and_uncompressed():
    """Stored, not deflated: the payload is h264 in mp4, so deflate spends CPU per byte to save
    roughly none. Flat, so opening the zip gives the videos rather than a chain of folders."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        folder = pathlib.Path(td) / "renders"
        folder.mkdir()
        for ep in (0, 1):
            (folder / f"run_ep{ep:03d}.mp4").write_bytes(b"\x00" * 2048)
        dest = zip_folder(folder, pathlib.Path(td) / "renders.zip")

        with zipfile.ZipFile(dest) as z:
            assert sorted(z.namelist()) == ["run_ep000.mp4", "run_ep001.mp4"]
            assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_one_bad_episode_does_not_end_the_batch(monkeypatch, tmp_path, capsys):
    """Episode 1 having nothing to render is not a reason to lose 2 and 3."""
    import misc.render_bulk as rb

    rendered = []

    def fake_render(args, progress=None):
        if args.episode == 1:
            raise SystemExit("nothing to render -- no frame had both action/action_samples and images")
        pathlib.Path(args.out).write_bytes(b"\x00" * 16)
        rendered.append(args.episode)

    class FakeReader:
        def __init__(self, *a, **k):
            pass

        def load(self):
            pass

        num_episodes = 4

    monkeypatch.setattr(rb, "DatasetReader", FakeReader, raising=False)
    monkeypatch.setitem(
        sys.modules, "misc.dataset_reader", types.SimpleNamespace(DatasetReader=FakeReader, SequentialImages=None)
    )
    # The name run_bulk actually calls. Patching misc.render_deploy_samples.render would not bind
    # here, because render_bulk imported it at module level -- and the batch would swallow every
    # resulting ImportError as a "failed episode" and still look like it ran.
    monkeypatch.setattr(rb, "render", fake_render)
    monkeypatch.setattr(
        sys, "argv", ["render_bulk", "--repo-id", "x/run", "--root", str(tmp_path), "--out-dir", str(tmp_path / "o")]
    )

    with pytest.raises(SystemExit) as e:
        rb.main()
    assert e.value.code == 1, "a batch with failures must exit non-zero"
    assert rendered == [0, 2, 3]
    assert (tmp_path / "o.zip").exists(), "the episodes that DID render are still worth zipping"
    cap = capsys.readouterr()
    assert "ep1: FAILED" in cap.out, "the failure is visible as it happens"
    assert "ep1: nothing to render" in cap.err, "and again in the end-of-run summary, with the reason"


def test_the_picker_lists_datasets_not_render_output(tmp_path):
    """Rendering writes `<name>_renders/` folders, and bulk writes more of them. Listing every
    directory put 14 of them in the GUI's dataset picker alongside the 48 real recordings; a
    LeRobot dataset is a folder with meta/info.json, so that is the test."""
    from misc.dataset_reader import list_datasets

    (tmp_path / "a_real_run" / "meta").mkdir(parents=True)
    (tmp_path / "a_real_run" / "meta" / "info.json").write_text("{}")
    (tmp_path / "a_real_run_renders").mkdir()
    (tmp_path / "a_real_run_renders" / "a_real_run_ep000.mp4").write_bytes(b"\x00")
    (tmp_path / ".hidden").mkdir()

    assert list_datasets(str(tmp_path)) == ["a_real_run"]
