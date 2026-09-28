"""Opt-in browser acceptance tests over the real gateway and offline workers.

Build adh-frontend first, then set ADH_FRONTEND_DIST to its dist directory.
Every browser request is routed through TestClient; no external service is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from .test_runs_api import _await_status, _make_client

DIST = os.environ.get("ADH_FRONTEND_DIST")
pytestmark = pytest.mark.skipif(
    not DIST, reason="Set ADH_FRONTEND_DIST for browser acceptance tests"
)


@pytest.fixture
def browser_app(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    from adh.service import app as gateway
    from adh.service.spa import mount_spa

    monkeypatch.setenv("ADH_API_KEY", "")
    monkeypatch.setenv("ADH_GENERATOR_MODELS", "")
    monkeypatch.setenv("ADH_JUDGE_MODELS", "")
    monkeypatch.setattr(gateway, "mount_spa", lambda app: mount_spa(app, str(DIST)))
    with _make_client(tmp_path) as client, playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, color_scheme="light"
        )

        def route_request(route):
            request = route.request
            url = urlsplit(request.url)
            assert url.hostname == "adh.test", f"Unexpected network request: {request.url}"
            response = client.request(
                request.method,
                url.path + (f"?{url.query}" if url.query else ""),
                content=request.post_data_buffer,
                headers={"Content-Type": request.headers.get("content-type", "application/json")},
            )
            route.fulfill(
                status=response.status_code,
                body=response.content,
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k not in {"content-length", "content-encoding"}
                },
            )

        context.route("**/*", route_request)
        page = context.new_page()
        page.set_default_timeout(5000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://adh.test/spec")
        playwright.expect(
            page.get_by_role("heading", name="Job Spec Builder", exact=True)
        ).to_be_visible()
        yield page, client, playwright.expect
        context.close()
        browser.close()
        assert not errors, errors


def _import(page, text):
    page.get_by_role("button", name="Import / preview YAML").click()
    page.get_by_label("YAML document").fill(text)
    page.get_by_role("button", name="Apply YAML", exact=True).click()


def test_validate_export_start_and_restart_preserve_settings(browser_app, tmp_path):
    page, client, expect = browser_app
    page.get_by_role("button", name="Validate & review", exact=True).click()
    expect(page.get_by_role("heading", name="Resolve 3 validation issues")).to_be_visible()
    assert client.get("/api/runs").json() == []

    draft = client.get("/api/specs/schema").json()["defaults"]
    draft.update(
        name="Builder acceptance",
        description="Rust async macro patterns and Tokio runtime internals",
        motivation="Teach async Rust",
        generation_params={"temperature": 0.2, "stop": ["END"]},
        maintainer="Dataset team",
        out_of_scope_uses=["Lending decisions"],
    )
    draft["discovery"]["channels"] = ["datasets", "web"]
    draft["discovery"]["constraints"]["include_domains"] = ["example.com"]
    draft["limits"]["max_runtime_seconds"] = 600
    _import(page, yaml.safe_dump(draft))
    expect(page.get_by_text("YAML imported and validated.", exact=True)).to_be_visible()

    page.get_by_role("button", name="Save draft", exact=True).click()
    page.get_by_label("Job name", exact=False).fill("Temporary")
    page.get_by_role("button", name="Restore draft", exact=True).click()
    expect(page.get_by_label("Job name", exact=False)).to_have_value("Builder acceptance")
    page.get_by_role("button", name="Validate & review", exact=True).click()
    expect(page.get_by_role("button", name="Start discovery", exact=True)).to_be_enabled()

    with page.expect_download() as download:
        page.get_by_role("button", name="Export YAML", exact=True).click()
    path = tmp_path / "exported.yaml"
    download.value.save_as(path)
    exported = yaml.safe_load(path.read_text())
    assert exported["generation_params"] == draft["generation_params"]
    assert exported["discovery"] == draft["discovery"]
    assert exported["limits"] == draft["limits"]

    page.get_by_role("button", name="Start discovery", exact=True).click()
    page.wait_for_url("**/runs/*")
    run_id = page.url.rsplit("/", 1)[-1]
    _await_status(client, run_id, "awaiting_approval")
    saved = client.get(f"/api/runs/{run_id}/spec").json()
    # The service assigns job_id when none was provided.
    assert {k: v for k, v in saved.items() if k != "job_id"} == {
        k: v for k, v in exported.items() if k != "job_id"
    }
    client.post(f"/api/runs/{run_id}/cancel")
    _await_status(client, run_id, "cancelled")
    page.goto(f"http://adh.test/runs/{run_id}/edit")
    expect(page.get_by_role("heading", name="Edit & restart job")).to_be_visible()
    page.get_by_label("Job name", exact=False).fill("Revised builder job")
    page.get_by_role("button", name="Validate & review", exact=True).click()
    page.get_by_role("button", name="Restart with changes", exact=True).click()
    page.wait_for_url(lambda url: "/runs/" in url and not url.endswith("/edit"))
    new_id = page.url.rsplit("/", 1)[-1]
    assert new_id != run_id
    restarted = client.get(f"/api/runs/{new_id}/spec").json()
    assert restarted == {**saved, "name": "Revised builder job"}


def test_invalid_imports_and_incompatible_schema_are_repairable(browser_app):
    page, client, expect = browser_app
    page.get_by_label("Job name", exact=False).fill("Keep my draft")
    _import(page, "name: [")
    expect(
        page.get_by_role("heading", name="Resolve 1 validation issue", exact=True)
    ).to_be_visible()
    page.get_by_role("button", name="Guided form").click()
    expect(page.get_by_label("Job name", exact=False)).to_have_value("Keep my draft")

    draft = client.get("/api/specs/schema").json()["defaults"]
    draft.update(
        name="Old job",
        description="Card rewards",
        motivation="Comparison",
        verifiability_tier=1,
        record_schema_ref="http://schema.org/Text",
        judge_config=None,
    )
    _import(page, yaml.safe_dump(draft))
    expect(page.locator("#spec-record_schema_ref-errors")).to_contain_text("record_schema_ref")
    page.get_by_label("Use record settings for").select_option("text")
    page.get_by_role("button", name="Validate & review", exact=True).click()
    expect(page.get_by_role("button", name="Start discovery", exact=True)).to_be_enabled()

    page.locator("summary").filter(has_text="Budgets & identifiers").click()
    page.get_by_label("Maximum LLM tokens", exact=False).fill("-1")
    page.get_by_role("button", name="Validate spec", exact=True).click()
    expect(page.locator("#spec-limits-max_llm_tokens-errors")).not_to_be_empty()
    assert client.get("/api/runs").json() == []


def test_layout_and_contract_coverage(browser_app, tmp_path):
    page, client, expect = browser_app
    # Hidden sections still render every field from the server contract.
    properties = client.get("/api/specs/schema").json()["json_schema"]["properties"]
    for name in properties:
        assert page.locator(f"#spec-{name}").count() == 1, name
    artifacts = Path(os.environ.get("ADH_BROWSER_ARTIFACTS", str(tmp_path)))
    artifacts.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        page.evaluate(
            "theme => { document.documentElement.className = theme; localStorage.setItem('adh_theme', theme); }",
            theme,
        )
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            expect(page.get_by_label("Job name", exact=False)).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                theme,
                width,
            )
            assert page.locator(".app-content").evaluate(
                "el => el.scrollWidth <= el.clientWidth"
            ), (theme, width)
            page.screenshot(path=str(artifacts / f"spec-{theme}-{width}.png"))
    # Structured JSON editor rejects syntax errors without losing the draft.
    page.locator("summary").filter(has_text="Generation & diversity").click()
    page.get_by_label("Generation parameters (JSON)", exact=True).fill('{"temperature":')
    page.get_by_role("button", name="Validate spec", exact=True).click()
    expect(page.locator("#spec-generation_params-errors")).to_contain_text("Enter valid JSON")
    assert (
        json.loads(page.evaluate("localStorage.getItem('adh_jobspec_draft_v1')") or "null") is None
    )
