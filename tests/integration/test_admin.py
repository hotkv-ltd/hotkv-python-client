# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for OBSERVE/ALERTS (enterprise observability; against a
server that answers NOLICENSE these are wrapped in `skip_if_not_licensed` and
skipped)."""

from __future__ import annotations


def test_observe_info_and_metrics(client, skip_if_not_licensed):
    def body():
        info = client.observe_info()
        assert "uptime_seconds" in info
        assert isinstance(info["connected_clients"], int)
        metrics = client.observe_metrics()
        assert "num_workers" in metrics

    skip_if_not_licensed(body)


def test_alerts_info(client, skip_if_not_licensed):
    def body():
        info = client.alerts_info()
        assert "enabled" in info
        assert isinstance(info["enabled"], bool)
        assert "supported_providers" in info

    skip_if_not_licensed(body)


def test_alerts_configuration_roundtrip(client, skip_if_not_licensed):
    # `enabled` in ALERTS INFO also requires a webhook_url to be configured
    # (none is configured here); `flag_enabled` reflects only the
    # ON/OFF toggle itself, so check that instead.
    def body():
        try:
            assert client.alerts_setseverity("warning") is True
            assert client.alerts_disable() is True
            assert client.alerts_info()["flag_enabled"] is False
            assert client.alerts_enable() is True
            assert client.alerts_info()["flag_enabled"] is True
        finally:
            client.alerts_disable()

    skip_if_not_licensed(body)
