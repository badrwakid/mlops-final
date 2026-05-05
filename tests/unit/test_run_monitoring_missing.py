from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_run_monitoring_missing_artifacts_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(tmp_path)
    import monitoring.run_monitoring as rm

    fake_cfg = MagicMock()
    fake_cfg.paths.model = str(tmp_path / "missing-model.pkl")
    fake_cfg.paths.preprocessor = str(tmp_path / "missing-pre.pkl")
    fake_cfg.paths.reference = str(tmp_path / "missing-ref.parquet")
    fake_cfg.paths.production = str(tmp_path / "missing-prod.parquet")

    monkeypatch.setattr(rm, "load_config", lambda: fake_cfg)
    with pytest.raises(SystemExit) as excinfo:
        rm.main()
    assert excinfo.value.code == 2
