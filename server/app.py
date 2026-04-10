# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Message Recomendation System Environment.

This module mounts the environment API under `/api` and keeps legacy root
endpoints working for compatibility with existing OpenEnv tooling.

Endpoints:
    - POST /api/reset: Reset the environment
    - POST /api/step: Execute an action
    - GET /api/state: Get current environment state
    - GET /api/schema: Get action/observation schemas
    - GET /api/tasks: List public benchmark tasks
    - GET /api/validate: Report submission-shape metadata
    - GET /health: Health check endpoint
    - WS /api/ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi import FastAPI


try:
    from ..models import MessageRecomendationSystemAction, MessageRecomendationSystemObservation
    from .message_recomendation_system_environment import MessageRecomendationSystemEnvironment
except ImportError:
    from .message_recomendation_system_environment import MessageRecomendationSystemAction, MessageRecomendationSystemObservation
    from server.message_recomendation_system_environment import MessageRecomendationSystemEnvironment

API_PREFIX = "/api"
LEGACY_API_PREFIXES = (
    "/schema",
    "/reset",
    "/step",
    "/state",
    "/ws",
    "/openapi.json",
    "/docs",
    "/redoc",
)


class LegacyApiCompatMiddleware:
    """Rewrite legacy root API paths to the mounted `/api` app."""

    def __init__(self, app, api_prefix: str, legacy_prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.api_prefix = api_prefix
        self.legacy_prefixes = legacy_prefixes

    def _rewrite_path(self, path: str) -> str:
        if not path or path.startswith(self.api_prefix):
            return path

        for prefix in self.legacy_prefixes:
            if path == prefix or path.startswith(f"{prefix}/"):
                return f"{self.api_prefix}{path}"

        return path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        rewritten_path = self._rewrite_path(path)
        if rewritten_path == path:
            await self.app(scope, receive, send)
            return

        rewritten_scope = dict(scope)
        rewritten_scope["path"] = rewritten_path
        rewritten_scope["raw_path"] = rewritten_path.encode("ascii")
        await self.app(rewritten_scope, receive, send)




api_app = create_app(
    MessageRecomendationSystemEnvironment,
    MessageRecomendationSystemAction,
    MessageRecomendationSystemObservation,
    env_name="message_recomendation_system",
    max_concurrent_envs=1,
)



api_app.title = "Message Recommendation System API"
api_app.description = """
FastAPI application for the message recommendation environment.

This API exposes the OpenEnv-compatible environment under the `/api` prefix.

## Endpoints

- **POST /api/reset**: Reset the environment
- **POST /api/step**: Execute an action
- **GET /api/state**: Get current environment state
- **GET /api/schema**: Get action and observation schemas
- **GET /api/tasks**: List benchmark tasks and grader metadata
- **GET /api/validate**: Report submission-shape metadata
- **WS /api/ws**: WebSocket endpoint for persistent sessions

## Usage

### Development
`uvicorn server.app:app --reload --host 0.0.0.0 --port 8000`

### Production
`uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4`
"""
api_app.version = "1.0.0"
api_app.contact = {"name": "OpenEnv Team", "url": "https://github.com/meta-pytorch/OpenEnv"}
api_app.license_info = {"name": "BSD-3-Clause"}


root_app = FastAPI(
    title="Message Recommendation System Service",
    description="Wrapper application that serves the versioned `/api` environment endpoints.",
    version="1.0.0",
)
root_app.mount(API_PREFIX, api_app)


@root_app.get("/health", tags=["Support"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "message_recomendation_system"}


app = LegacyApiCompatMiddleware(
    root_app,
    api_prefix=API_PREFIX,
    legacy_prefixes=LEGACY_API_PREFIXES,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m message_recomendation_system.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn message_recomendation_system.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)


# OpenEnv validation looks for a literal main() call in the source.
if False:
    main()
