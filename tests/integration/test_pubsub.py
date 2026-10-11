# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Integration tests for publish/subscribe."""

from __future__ import annotations

import pytest

import hotkv


def test_subscribe_and_receive_message(client, test_url, prefix):
    channel = prefix + "chan"
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    assert pubsub.subscribed
    assert channel.encode() in pubsub.channels

    publisher = hotkv.from_url(test_url)
    try:
        assert publisher.execute("PUBLISH", channel, "hello") == 1
        message = None
        for _ in range(50):
            message = pubsub.get_message(timeout=0.2, ignore_subscribe_messages=True)
            if message is not None:
                break
        assert message is not None
        assert message.type == "message"
        assert message.channel == channel.encode()
        assert message.data == b"hello"
    finally:
        pubsub.close()
        publisher.close()


def test_psubscribe_pattern_message(client, test_url, prefix):
    pattern = prefix + "news.*"
    channel = prefix + "news.sports"
    pubsub = client.pubsub()
    pubsub.psubscribe(pattern)
    assert pattern.encode() in pubsub.patterns

    publisher = hotkv.from_url(test_url)
    try:
        publisher.execute("PUBLISH", channel, "goal")
        message = None
        for _ in range(50):
            message = pubsub.get_message(timeout=0.2, ignore_subscribe_messages=True)
            if message is not None:
                break
        assert message is not None
        assert message.type == "pmessage"
        assert message.pattern == pattern.encode()
        assert message.channel == channel.encode()
        assert message.data == b"goal"
    finally:
        pubsub.close()
        publisher.close()


def test_unsubscribe(client, prefix):
    channel = prefix + "chan2"
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    assert channel.encode() in pubsub.channels
    pubsub.unsubscribe(channel)
    assert channel.encode() not in pubsub.channels
    assert not pubsub.subscribed
    pubsub.close()


def test_listen_generator_yields_messages(client, test_url, prefix):
    channel = prefix + "listenchan"
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    publisher = hotkv.from_url(test_url)
    try:
        publisher.execute("PUBLISH", channel, "via-listen")
        received = []
        for message in pubsub:
            received.append(message)
            break
        assert received and received[0].data == b"via-listen"
    finally:
        pubsub.close()
        publisher.close()
