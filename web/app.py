#!/usr/bin/env python3
"""ShritGPT web chat."""

import json
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from gpt.chat import get_bot
from gpt.paths import CHECKPOINT

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = Flask(__name__, static_folder=str(STATIC))


def _parse_chat_request():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return None, None, None, None, None, ("empty message", 400)
    history = data.get("history") or []
    temperature = float(data.get("temperature", 0.5))
    top_k = int(data.get("top_k", 25))
    max_tokens = int(data.get("max_tokens", 180))
    return message, history, temperature, top_k, max_tokens, None


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    message, history, temperature, top_k, max_tokens, err = _parse_chat_request()
    if err:
        return jsonify({"error": err[0]}), err[1]

    try:
        bot = get_bot()
        reply = bot.reply(
            message,
            history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"generation failed: {e}"}), 500

    return jsonify({"reply": reply})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    message, history, temperature, top_k, max_tokens, err = _parse_chat_request()
    if err:
        return jsonify({"error": err[0]}), err[1]

    def events():
        try:
            bot = get_bot()
            for chunk in bot.reply_stream(
                message,
                history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except FileNotFoundError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'generation failed: {e}'})}\n\n"

    return Response(
        stream_with_context(events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/health")
def health():
    return jsonify({
        "ok": CHECKPOINT.exists(),
        "name": "ShritGPT",
        "stream": True,
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204


def main():
    print("Loading ShritGPT model (first message may take a few sec)...")
    try:
        get_bot()
        print("Model ready.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    print("Open http://127.0.0.1:5050")
    print("Routes: /  /api/chat  /api/chat/stream  /api/health")
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
