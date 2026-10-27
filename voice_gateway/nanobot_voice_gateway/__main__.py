"""CLI entrypoint for the nanobot voice gateway."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .app import create_app
from .config import GatewayConfig

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    env = GatewayConfig.from_env()
    parser = argparse.ArgumentParser(
        prog="nanobot-voice-gateway",
        description="SSE voice gateway for the nanobot Python SDK",
    )
    parser.add_argument("--host", default=env.host)
    parser.add_argument("--port", type=int, default=env.port)
    parser.add_argument("--api-key", default=env.api_key)
    parser.add_argument("--config", dest="nanobot_config", default=env.nanobot_config)
    parser.add_argument("--workspace", default=env.workspace)
    parser.add_argument("--model-preset", default=env.model_preset)
    parser.add_argument(
        "--max-concurrent-turns",
        type=int,
        default=env.max_concurrent_turns,
    )
    parser.add_argument("--cancel-timeout", type=float, default=env.cancel_timeout)
    parser.add_argument(
        "--heartbeat-interval", type=float, default=env.heartbeat_interval
    )
    parser.add_argument("--turn-ttl", type=float, default=env.turn_ttl)
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable session snapshots and rollback",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the gateway."""
    logging.basicConfig(
        level=os.environ.get("NANOBOT_VOICE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = GatewayConfig(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        nanobot_config=args.nanobot_config,
        workspace=args.workspace,
        model_preset=args.model_preset,
        max_concurrent_turns=args.max_concurrent_turns,
        cancel_timeout=args.cancel_timeout,
        heartbeat_interval=args.heartbeat_interval,
        turn_ttl=args.turn_ttl,
        snapshot_enabled=not args.no_snapshot,
    )
    asyncio.run(_serve(config))


async def _serve(config: GatewayConfig) -> None:
    from aiohttp import web
    from nanobot import Nanobot

    async with Nanobot.from_config(
        config.nanobot_config,
        workspace=config.workspace,
        model_preset=config.model_preset,
    ) as bot:
        app = create_app(bot, config)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()
        LOGGER.info(
            "nanobot voice gateway listening on http://%s:%s",
            config.host,
            config.port,
        )
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    main()
