from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator

import pytest


def _run(script: str, **env: str) -> str:
    environ = {key: value for key, value in os.environ.items() if key not in {'LLM_API_ENDPOINT', 'ARENA_PROMPT_LLM'}}
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)], env={**environ, **env}, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class _Endpoint(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).requests.append((self.path, body))
        reply = json.dumps({'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'pong'}]}, 'finishReason': 'STOP'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args: object) -> None:
        del args


@pytest.fixture
def endpoint() -> Iterator[tuple[str, list[tuple[str, dict]]]]:
    _Endpoint.requests = []
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', _Endpoint.requests
    finally:
        server.shutdown()
        server.server_close()


def test_the_default_model_follows_arena_prompt_llm() -> None:
    script = 'from arena_simulation_setup.utils.llm import DEFAULT_MODEL; print(DEFAULT_MODEL)'
    assert _run(script) == 'gemini-3.5-flash'
    assert _run(script, ARENA_PROMPT_LLM='local-test-model') == 'local-test-model'


def test_an_unknown_name_raises_attribute_error() -> None:
    from arena_simulation_setup.utils import llm

    with pytest.raises(AttributeError, match="has no attribute 'mistral'"):
        _ = llm.mistral


def test_without_an_endpoint_the_shims_pass_the_libraries_through() -> None:
    pytest.importorskip('google.genai')
    pytest.importorskip('openai')
    assert (
        _run("""
        import openai
        from google import genai as upstream
        from arena_simulation_setup.utils import llm
        print(llm.genai.Client is upstream.Client, llm.genai.types is upstream.types, llm.openai.OpenAI is openai.OpenAI)
        """)
        == 'True True True'
    )


def test_with_an_endpoint_a_local_cache_is_expanded_into_the_request(endpoint: tuple[str, list[tuple[str, dict]]]) -> None:
    pytest.importorskip('google.genai')
    url, requests = endpoint
    out = _run(
        """
        from arena_simulation_setup.utils.llm import genai
        client = genai.Client()
        cache = client.caches.create(model='m', config=genai.types.CreateCachedContentConfig(system_instruction='be brief', contents=['context line']))
        print([c.name for c in client.caches.list()])
        reply = client.models.generate_content(model='m', contents='ping', config=genai.types.GenerateContentConfig(cached_content=cache.name, temperature=0.0))
        print(reply.text)
        client.caches.delete(name=cache.name)
        print([c.name for c in client.caches.list()])
        """,
        LLM_API_ENDPOINT=url,
    )
    assert out.splitlines() == ["['local-cache-1']", 'pong', '[]']
    [(path, body)] = requests
    assert path == '/v1beta/models/m:generateContent'
    assert [part['text'] for content in body['contents'] for part in content['parts']] == ['context line', 'ping']
    assert body['systemInstruction']['parts'] == [{'text': 'be brief'}]
    assert 'cachedContent' not in body


class _Slow(http.server.BaseHTTPRequestHandler):
    release = threading.Event()
    responded = threading.Event()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers['Content-Length']))
        self.release.wait(60.0)
        type(self).responded.set()
        self.send_response(500)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        del args


def test_with_an_endpoint_the_genai_client_keeps_its_timeout() -> None:
    pytest.importorskip('google.genai')
    _Slow.release = threading.Event()
    _Slow.responded = threading.Event()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = _run(
            """
            from arena_simulation_setup.utils.llm import genai
            client = genai.Client(http_options=genai.types.HttpOptions(timeout=300))
            try:
                client.models.generate_content(model='m', contents='ping')
            except Exception as exc:
                print(type(exc).__name__)
            """,
            LLM_API_ENDPOINT=f'http://127.0.0.1:{server.server_address[1]}',
        )
        answered = _Slow.responded.is_set()
    finally:
        _Slow.release.set()
        server.shutdown()
        server.server_close()
    assert out, out
    assert not answered, f"the client waited for the stalled server instead of timing out: {out}"


def test_with_an_endpoint_the_openai_clients_keep_their_options() -> None:
    pytest.importorskip('openai')
    out = _run(
        """
        from arena_simulation_setup.utils import llm
        for client in (llm.openai.OpenAI(timeout=7.5, max_retries=4), llm.openai.AsyncOpenAI(timeout=7.5, max_retries=4)):
            print(client.timeout, client.max_retries, client.base_url, client.api_key)
        """,
        LLM_API_ENDPOINT='http://127.0.0.1:9',
    )
    assert out.splitlines() == ['7.5 4 http://127.0.0.1:9/v1/ sk-local'] * 2
