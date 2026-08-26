"""The render GUI: what it reads off a dataset, and the args it hands the renderer."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from misc.render_gui import RenderGUI


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_an_empty_root_says_so_instead_of_failing(qapp, tmp_path):
    gui = RenderGUI(str(tmp_path))
    try:
        assert gui.dataset_combo.count() == 0
        assert "No dataset folders" in gui.info.text()
        assert gui.episode_combo.count() == 0
    finally:
        gui.close()


def test_it_leaves_the_recorded_numbers_for_the_renderer_to_recover(qapp, tmp_path):
    """The point of the window: nothing the recording already knows is typed in. `candidates` and
    `horizon` stay None so render() reads them off the dataset -- a mistyped candidate count is a
    reshape error and a mistyped horizon silently draws every chunk in the wrong place."""
    gui = RenderGUI(str(tmp_path))
    try:
        args = gui._build_args()
        assert args.candidates is None and args.horizon is None
        assert args.replans == 0 and args.hold == 1
        # Speed is the written frame rate, not a re-encode, and it is relative to the dataset's
        # own rate -- see test_speed_is_relative_to_the_dataset_not_a_fixed_rate.
        gui._fps = 30
        gui.speed_spin.setValue(1.0)
        assert gui._build_args().fps == 30
        gui.speed_spin.setValue(4.0)
        assert gui._build_args().fps == 120
    finally:
        gui.close()


def test_render_is_refused_without_an_episode(qapp, tmp_path):
    """An empty root has no episode to render; pressing the button must say so rather than start a
    worker that fails deep inside the renderer."""
    gui = RenderGUI(str(tmp_path))
    try:
        gui._on_render()
        assert "episode" in gui.status.text().lower()
        assert gui.render_btn.isEnabled()
    finally:
        gui.close()


def test_a_long_dataset_list_scrolls(qapp, tmp_path):
    """A styled popup comes with ScrollBarAlwaysOff and ignores maxVisibleItems, so a list longer
    than the screen simply lost its tail with nothing to drag -- and one root here holds 41
    datasets."""
    from PyQt5 import QtCore

    gui = RenderGUI(str(tmp_path))
    try:
        for combo in (gui.dataset_combo, gui.episode_combo):
            assert combo.view().verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
            assert combo.maxVisibleItems() > 1
            assert "combobox-popup: 0" in combo.styleSheet()
    finally:
        gui.close()


def test_speed_is_relative_to_the_dataset_not_a_fixed_rate(qapp, tmp_path):
    """One rendered frame per recorded tick, so writing at the dataset's own rate is real time.
    Multiplying a fixed 10 made "speed 1.0" a third-speed video on 30 fps footage -- which reads
    as the renderer being slow rather than the file being slowed down."""
    gui = RenderGUI(str(tmp_path))
    try:
        gui._fps = 30
        gui.speed_spin.setValue(1.0)
        assert gui._build_args().fps == 30, "1.0 is real time"
        gui.speed_spin.setValue(4.0)
        assert gui._build_args().fps == 120
        gui._fps = 60  # a 60 fps recording writes at 60 for real time
        gui.speed_spin.setValue(1.0)
        assert gui._build_args().fps == 60
    finally:
        gui.close()


def test_progress_is_reported_and_throttled_to_percents():
    """A render is tens of seconds; the window should say how far it is. Emitting per frame would
    post 9000 events for a 9000-frame episode to show 100 distinct states, so the worker only
    emits when the whole percent changes."""
    from misc.render_gui import _RenderWorker

    worker = _RenderWorker(None)
    seen = []
    worker.progressed.connect(lambda w, t: seen.append(w))

    # The reporting closure is what render() is handed; drive it directly rather than rendering.
    last = [-1]

    def report(written, total):
        pct = (100 * written) // max(total, 1)
        if pct != last[0]:
            last[0] = pct
            worker.progressed.emit(written, total)

    for i in range(1, 1001):
        report(i, 1000)
    # 0 % through 100 % inclusive -- 101 distinct states, not 100.
    assert len(seen) == 101, f"one event per percent, got {len(seen)}"
    assert seen[-1] == 1000


def test_the_bar_hides_when_the_render_finishes(qapp, tmp_path):
    gui = RenderGUI(str(tmp_path))
    try:
        gui.progress.setVisible(True)
        gui._on_progress(30, 120)
        assert gui.progress.value() == 25
        assert "30 / 120" in gui.progress.format()
        gui._on_done(True, str(tmp_path / "nope.mp4"))
        assert not gui.progress.isVisible()
        assert gui.render_btn.isEnabled()
    finally:
        gui.close()


def _with_episodes(gui, lengths=(100, 200, 300)):
    """Populate the combo the way _refresh_episodes does, without needing a dataset on disk."""
    from misc.render_gui import BULK_LABEL

    gui._episodes = [(i, n) for i, n in enumerate(lengths)]
    gui.episode_combo.blockSignals(True)
    gui.episode_combo.clear()
    gui.episode_combo.addItem(BULK_LABEL)
    for ep, n in gui._episodes:
        gui.episode_combo.addItem(f"episode {ep}  ·  {n} frames")
    gui.episode_combo.blockSignals(False)
    return gui


def test_the_bulk_entry_shifts_every_episode_row_by_one(qapp, tmp_path):
    """Putting "all episodes" first means row N is episode N-1. Getting this wrong renders the
    neighbouring episode and looks entirely successful -- the video plays, it is just the wrong
    one."""
    gui = _with_episodes(RenderGUI(str(tmp_path)))
    try:
        gui.episode_combo.setCurrentIndex(0)
        assert gui._bulk_selected() is True
        assert gui._current_episode() is None, "the bulk row is not an episode"

        for row, expect in ((1, 0), (2, 1), (3, 2)):
            gui.episode_combo.setCurrentIndex(row)
            assert gui._bulk_selected() is False
            assert gui._current_episode() == expect
    finally:
        gui.close()


def test_output_is_a_folder_for_a_batch_and_a_file_for_one_episode(qapp, tmp_path):
    """The batch writes many mp4s and a zip beside them, so its output cannot be a .mp4 path."""
    gui = RenderGUI(str(tmp_path))
    try:
        # Name the dataset FIRST: adding it fires currentTextChanged -> _refresh_episodes, which
        # would clear whatever the helper had put in the combo (the folder is not on disk here).
        gui.dataset_combo.addItem("my_run")
        gui.dataset_combo.setCurrentIndex(0)
        _with_episodes(gui)

        gui.episode_combo.setCurrentIndex(0)
        gui._sync_out()
        assert gui.out_edit.text().endswith("my_run_renders")
        assert not gui.out_edit.text().endswith(".mp4")

        gui.episode_combo.setCurrentIndex(2)
        gui._sync_out()
        assert gui.out_edit.text().endswith("my_run_ep1.mp4")
    finally:
        gui.close()


def test_no_bulk_entry_when_there_are_no_episodes(qapp, tmp_path):
    """An empty dataset must not offer to render all of nothing -- and row 0 would then be an
    episode row, which is what the offset in _current_episode depends on."""
    gui = RenderGUI(str(tmp_path))
    try:
        assert gui.episode_combo.count() == 0
        assert gui._bulk_selected() is False
        assert gui._current_episode() is None
    finally:
        gui.close()
