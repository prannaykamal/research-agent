"""Serve the single-page research UI from the LangGraph server at /app.

Mounted through ``http.app`` in ``langgraph.json``. The routes only serve static
files and profile metadata; they never touch graph execution.
"""

from dataclasses import asdict
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.utils.guardrails import MAX_ANALYSTS, MIN_ANALYSTS
from src.utils.profiles import DEFAULT_PROFILE, PROFILES

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _profile_summary(name: str) -> dict:
    profile = PROFILES[name]
    return {
        "models": {
            tier: asdict(getattr(profile, tier))
            for tier in ("heavy", "medium", "writer", "light", "panel")
        },
        "max_passes": profile.max_research_loops,
        "max_tool_calls_per_pass": profile.max_tool_calls_per_pass,
        "max_researcher_turns": profile.max_researcher_turns,
        "deadline_seconds": profile.analyst_deadline_seconds,
        "allowed_tools": list(profile.allowed_tools),
        "max_analysts": profile.max_analysts,
    }


async def profiles(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "default": DEFAULT_PROFILE,
            "min_analysts": MIN_ANALYSTS,
            "max_analysts": MAX_ANALYSTS,
            "profiles": {name: _profile_summary(name) for name in PROFILES},
        }
    )


async def to_app(_request: Request) -> RedirectResponse:
    return RedirectResponse("/app/")


app = Starlette(
    routes=[
        Route("/app", to_app),
        Route("/app/profiles.json", profiles),
        Mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True, check_dir=False)),
    ]
)
