import asyncio

from locate_anything_service.middleware import RequestBodyLimitMiddleware


def test_rejects_oversized_content_length_before_downstream() -> None:
    downstream_called = False
    sent_messages = []

    async def downstream(_scope, _receive, _send) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message) -> None:
        sent_messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [(b"content-length", b"6")],
    }
    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=5)

    asyncio.run(middleware(scope, receive, send))

    assert downstream_called is False
    assert sent_messages[0]["status"] == 413


def test_rejects_chunked_body_while_streaming() -> None:
    chunks = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent_messages = []

    async def downstream(_scope, receive, _send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break

    async def receive():
        return next(chunks)

    async def send(message) -> None:
        sent_messages.append(message)

    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": []}
    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=5)

    asyncio.run(middleware(scope, receive, send))

    assert sent_messages[0]["status"] == 413
