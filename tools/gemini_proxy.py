"""
Proxy local OpenAI-compativel -> Gemini (generateContent).

Motivacao
---------
A API Gemini gratuita limita ~5 requisicoes/minuto e ~20 requisicoes/dia por
projeto/modelo. Uma avaliacao completa consome ~60 chamadas (15 geracoes +
45 juizes), o que estoura essa cota. Este proxy roda localmente, expoe uma API
identica a `/v1/chat/completions` da OpenAI e rotaciona varias API keys (de
projetos distintos) para multiplicar a cota efetiva, respeitando o limite de
requisicoes por minuto.

Ele e OPCIONAL: so e necessario quando a cota gratuita nao e suficiente. Com
uma key/plano com cota adequada, use direto `LLM_PROVIDER=google`.

Uso
---
1. Defina as keys (uma por projeto) em uma das formas:
   - variavel de ambiente:  export GEMINI_PROXY_KEYS="AIza...,AIza...,AIza..."
   - arquivo JSON:          tools/keys.json  ->  {"keys": ["AIza...", "AIza..."]}
2. Rode:
       python tools/gemini_proxy.py            # escuta em 127.0.0.1:8765
3. No .env do projeto:
       LLM_PROVIDER=openai
       OPENAI_API_KEY=dummy
       OPENAI_API_BASE=http://127.0.0.1:8765/v1
       OPENAI_BASE_URL=http://127.0.0.1:8765/v1
       LLM_MODEL=gemini-3.5-flash
       EVAL_MODEL=gemini-3.5-flash-lite
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
PORT = int(os.getenv("GEMINI_PROXY_PORT", "8765"))
MIN_INTERVAL = float(os.getenv("GEMINI_PROXY_MIN_INTERVAL", "13"))
KEYS_FILE = Path(os.getenv("GEMINI_PROXY_KEYS_FILE", "tools/keys.json"))

LOCK = threading.Lock()
LAST_USE = {}
DEAD = set()
COUNTS = {}


def load_keys():
    env_keys = os.getenv("GEMINI_PROXY_KEYS", "")
    if env_keys.strip():
        return [k.strip() for k in env_keys.split(",") if k.strip()]
    if KEYS_FILE.exists():
        return list(json.loads(KEYS_FILE.read_text())["keys"])
    raise SystemExit(
        "Nenhuma key encontrada. Defina GEMINI_PROXY_KEYS ou crie tools/keys.json"
    )


KEYS = load_keys()


def build_gemini_payload(messages, temperature):
    system_parts = []
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role == "system":
            system_parts.append(content)
        else:
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )

    payload = {"contents": contents or [{"role": "user", "parts": [{"text": ""}]}]}
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    payload["generationConfig"] = {"temperature": temperature}
    return payload


def call_gemini(key, model, payload):
    url = f"{GEMINI_BASE}/{model}:generateContent?key={key}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def pick_and_call(model, payload):
    attempts = 0
    while attempts < len(KEYS) * 4:
        attempts += 1
        with LOCK:
            candidates = [k for k in KEYS if (k, model) not in DEAD]
            if not candidates:
                raise RuntimeError(f"Todas as keys esgotadas para o modelo {model}")
            now = time.time()
            candidates.sort(key=lambda k: LAST_USE.get((k, model), 0.0))
            key = candidates[0]
            wait = MIN_INTERVAL - (now - LAST_USE.get((key, model), 0.0))
        if wait > 0:
            time.sleep(wait)
        with LOCK:
            LAST_USE[(key, model)] = time.time()
        try:
            text = call_gemini(key, model, payload)
            with LOCK:
                COUNTS[(key, model)] = COUNTS.get((key, model), 0) + 1
            return text
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            if e.code == 429:
                if "PerDay" in detail or "per day" in detail.lower():
                    with LOCK:
                        DEAD.add((key, model))
                    continue
                time.sleep(MIN_INTERVAL)
                continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {detail[:200]}")
    raise RuntimeError("Numero maximo de tentativas excedido")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/models"):
            self._send(200, {"object": "list", "data": []})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid json"}})
            return

        model = req.get("model", "gemini-3.5-flash")
        temperature = req.get("temperature", 0)
        payload = build_gemini_payload(req.get("messages", []), temperature)

        try:
            text = pick_and_call(model, payload)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": {"message": str(e)}})
            return

        self._send(
            200,
            {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )


def main():
    print(f"Proxy OpenAI->Gemini em http://127.0.0.1:{PORT}/v1  ({len(KEYS)} keys)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
