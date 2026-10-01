import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from triage.common import ARTIFACTS, ROOT

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "evaluation.json").exists(), reason="Build official artifacts first")


def test_all_views_and_scenarios(monkeypatch):
    monkeypatch.setenv("APP_MODE", "hosted")
    monkeypatch.setattr("triage.briefs.requests.Session.post", lambda *a, **k: pytest.fail("Hosted inference call"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Manufacturing quality triage"
    app.slider[0].set_value(0).run()
    assert not app.exception
    app.button[0].click().run()
    assert app.slider[0].value == 20
    app.selectbox[0].select_index(1).run()
    assert not app.exception
    for view in ("Model performance", "Data and methods"):
        app.sidebar.radio[0].set_value(view).run()
        assert not app.exception
    app.selectbox[0].select_index(5).run()
    assert not app.exception

