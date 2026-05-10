#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Studio — DeepSeek × GPT-Image-2
Multi-user web service. Run: python app.py
Each user manages their own API keys in the browser (stored in localStorage).
"""

import os, json, base64, threading, re, uuid, time, sqlite3
from datetime import datetime
from queue import Queue, Empty
from flask import Flask, render_template_string, request, jsonify, send_file, Response, stream_with_context
import io, urllib.request, urllib.error, requests as _requests
from openai import OpenAI

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")
_db_lock = threading.Lock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _db_init():
    with _db_lock, _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            cid      TEXT PRIMARY KEY,
            sid      TEXT NOT NULL,
            preview  TEXT DEFAULT '',
            msgs     TEXT DEFAULT '[]',
            started  REAL,
            updated  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_conv_sid ON conversations(sid, updated DESC);
        CREATE TABLE IF NOT EXISTS images (
            iid      TEXT PRIMARY KEY,
            sid      TEXT NOT NULL,
            cid      TEXT DEFAULT '',
            title    TEXT DEFAULT '',
            prompt   TEXT DEFAULT '',
            filepath TEXT DEFAULT '',
            ts       TEXT,
            created  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_img_sid ON images(sid, created DESC);
        """)

_db_init()

def db_upsert_conv(cid, sid, preview, msgs):
    with _db_lock, _db() as c:
        now = time.time()
        c.execute("""
            INSERT INTO conversations(cid,sid,preview,msgs,started,updated)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(cid) DO UPDATE SET msgs=excluded.msgs, updated=excluded.updated
        """, (cid, sid, preview, json.dumps(msgs, ensure_ascii=False), now, now))

def db_save_image(iid, sid, cid, title, prompt, filepath, ts):
    with _db_lock, _db() as c:
        c.execute("""
            INSERT OR IGNORE INTO images(iid,sid,cid,title,prompt,filepath,ts,created)
            VALUES(?,?,?,?,?,?,?,?)
        """, (iid, sid, cid, title, prompt, filepath, ts, time.time()))

# optional Pillow for compression
try:
    from PIL import Image as PilImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

def compress_b64(b64_str, max_kb=1400):
    """Compress base64 image to ≤max_kb KB, returns raw bytes."""
    raw = base64.b64decode(b64_str)
    if not _HAS_PIL or len(raw) <= max_kb * 1024:
        return raw
    img = PilImage.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), PilImage.LANCZOS)
    quality = 88
    while quality >= 50:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        if len(buf.getvalue()) <= max_kb * 1024:
            return buf.getvalue()
        quality -= 8
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=50)
    return buf.getvalue()

def build_multipart(fields: dict, files: list) -> tuple:
    """Build multipart/form-data body. files = [(field, filename, data, mime)]"""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for field, filename, data, mime in files:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode() + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"

app = Flask(__name__)

# ── per-session state ─────────────────────────────────────────────────────────
sessions      = {}
sessions_lock = threading.Lock()
gen_queue     = Queue()
SESSION_TTL   = 3600 * 6   # 6h idle → cleanup

def get_session(sid):
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = {
                "history": [],
                "prompts": {},
                "images":  {},
                "conv_id": str(uuid.uuid4()),
                "last_active": time.time()
            }
        sessions[sid]["last_active"] = time.time()
        return sessions[sid]

def _cleanup_loop():
    while True:
        time.sleep(1800)
        now = time.time()
        with sessions_lock:
            dead = [s for s, v in sessions.items() if now - v["last_active"] > SESSION_TTL]
            for s in dead:
                del sessions[s]

threading.Thread(target=_cleanup_loop, daemon=True).start()

# ── GPT-4o via FrankAI ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位专业的AI图像生成顾问，擅长为GPT-Image-2设计高质量的英文提示词。

工作流程：
1. 与用户沟通，深入了解他们想要的图像风格、数量、用途、细节
2. 如果用户上传了参考图，请仔细分析图片的风格、构图、色调、主体等特征
3. 充分沟通后，生成结构化提示词供用户确认
4. 根据用户反馈修改提示词

当你准备好输出提示词时，每个提示词用以下格式包裹（可一次输出多个）：
[PROMPT]
标题：<简短中文标题，5字以内>
中文说明：<一句话描述画面内容，让中文用户看懂这张图要生成什么>
提示词：<详细英文prompt，描述画面内容、风格、光线、构图等>
[/PROMPT]

在未充分了解需求前，先进行对话沟通，不要急于输出提示词。"""

def call_gpt4o(messages, frank_key, image_b64=None, model="gpt-5.4",
               endpoint="https://api.frankai.cc/v1"):
    if not frank_key:
        raise ValueError("请先填写 GPT Key（点右上角 🔑）")

    msg_list = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    if image_b64 and msg_list[-1]["role"] == "user":
        last_text = msg_list[-1]["content"]
        msg_list[-1] = {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
                {"type": "text", "text": last_text}
            ]
        }

    # 确保 endpoint 以 /chat/completions 结尾
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1").rstrip("/") + "/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": msg_list,
        "stream": True          # 此 API 必须 stream:True 才返回内容
    }).encode()

    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {frank_key}"}
    try:
        resp = _requests.post(url, data=payload, headers=headers,
                              timeout=120, stream=True, verify=False)
        resp.encoding = "utf-8"   # 强制 UTF-8，防止被误判为 Latin-1
        resp.raise_for_status()
    except _requests.exceptions.HTTPError as e:
        try:
            msg = e.response.json().get("error", {}).get("message", e.response.text)
        except Exception:
            msg = str(e)
        raise ValueError(f"GPT {e.response.status_code}: {msg}")
    except Exception as e:
        raise ValueError(f"请求失败: {e}")

    # 解析 SSE 流：逐行读取，拼接 delta.content
    chunks = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        line = (raw_line or "").strip()
        if not line.startswith("data:"):
            continue
        payload_str = line[5:].strip()
        if payload_str == "[DONE]":
            break
        try:
            obj = json.loads(payload_str)
            for ch in (obj.get("choices") or []):
                content = ch.get("delta", {}).get("content") or ""
                if content:
                    chunks.append(content)
        except Exception:
            continue

    if chunks:
        return "".join(chunks)
    raise ValueError("API 返回了空内容，请检查 Key、模型名和接口地址是否正确")

def stream_gpt4o(messages, frank_key, image_b64=None, model="gpt-5.4",
                 endpoint="https://api.frankai.cc/v1"):
    if not frank_key:
        raise ValueError("请先填写 GPT Key（点左下角 🔑）")
    msg_list = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    if image_b64 and msg_list[-1]["role"] == "user":
        last_text = msg_list[-1]["content"]
        msg_list[-1] = {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
            {"type": "text", "text": last_text}
        ]}
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1").rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({"model": model, "messages": msg_list, "stream": True}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {frank_key}"}
    resp = _requests.post(url, data=payload, headers=headers, timeout=120, stream=True, verify=False)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    for raw_line in resp.iter_lines(decode_unicode=True):
        line = (raw_line or "").strip()
        if not line.startswith("data:"): continue
        payload_str = line[5:].strip()
        if payload_str == "[DONE]": break
        try:
            obj = json.loads(payload_str)
            for ch in (obj.get("choices") or []):
                content = ch.get("delta", {}).get("content") or ""
                if content: yield content
        except Exception:
            continue

def parse_prompts(text):
    blocks = re.findall(r'\[PROMPT\](.*?)\[/PROMPT\]', text, re.DOTALL)
    results = []
    for b in blocks:
        title_m = re.search(r'标题[：:]\s*(.+)', b)
        desc_m  = re.search(r'中文说明[：:]\s*(.+?)(?=\n提示词|\n标题|\Z)', b, re.DOTALL)
        text_m  = re.search(r'提示词[：:]\s*(.+?)(?=\n标题|\Z)', b, re.DOTALL)
        if text_m:
            results.append({
                "title": title_m.group(1).strip() if title_m else "图像",
                "desc":  desc_m.group(1).strip() if desc_m else "",
                "text":  text_m.group(1).strip()
            })
    return results

# ── image generation ─────────────────────────────────────────────────────────
def _frank_post(url, body, content_type, key, timeout=300):
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": content_type,
                 "Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw).get("error", {}).get("message", raw)
        except Exception:
            msg = raw
        raise ValueError(f"FrankAI {e.code}: {msg}")

def do_generate(sid, pid, prompt_text, size, quality, n, frank_keys, ref_images=None):
    sess = get_session(sid)
    try:
        sess["prompts"][pid]["status"] = "generating"
        keys = [k for k in frank_keys if k and k.strip()]
        if not keys:
            raise ValueError("请先填写 FrankAI API Key（点右上角 🔑）")
        key = keys[hash(pid) % len(keys)]

        ref = [r for r in (ref_images or []) if r]  # non-empty b64 strings
        ref = ref[:4]  # cap at 4 for best quality

        if ref:
            # ── edits endpoint (with reference images) ──
            fields = {"prompt": prompt_text, "model": "gpt-image-2",
                      "size": size, "quality": quality, "n": str(n),
                      "response_format": "b64_json"}
            files  = []
            for i, b64 in enumerate(ref):
                img_bytes = compress_b64(b64)
                mime = "image/webp" if _HAS_PIL else "image/png"
                files.append(("image[]", f"ref_{i}.{'webp' if _HAS_PIL else 'png'}",
                               img_bytes, mime))
            body, ct = build_multipart(fields, files)
            data = _frank_post("https://api.frankai.cc/v1/images/edits",
                               body, ct, key)
        else:
            # ── generations endpoint (text only) ──
            body = json.dumps({
                "model": "gpt-image-2", "prompt": prompt_text,
                "size": size, "quality": quality, "n": n,
                "response_format": "b64_json"
            }).encode()
            data = _frank_post("https://api.frankai.cc/v1/images/generations",
                               body, "application/json", key)

        iids  = []
        title = sess["prompts"].get(pid, {}).get("title", "")
        cid   = sess.get("conv_id", "")
        for item in data.get("data", []):
            iid = str(uuid.uuid4())
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            sess["images"][iid] = {
                "b64": item["b64_json"], "ts": ts, "prompt_text": prompt_text
            }
            iids.append(iid)
            os.makedirs("outputs", exist_ok=True)
            fpath = os.path.join("outputs", f"{iid}.png")
            with open(fpath, "wb") as f:
                f.write(base64.b64decode(item["b64_json"]))
            db_save_image(iid, sid, cid, title, prompt_text, fpath, ts)

        sess["prompts"][pid]["images"] = iids
        sess["prompts"][pid]["status"] = "done"
    except Exception as e:
        if pid in sess.get("prompts", {}):
            sess["prompts"][pid]["status"] = "error"
            sess["prompts"][pid]["error"]  = str(e)

def _gen_worker():
    while True:
        try:
            job = gen_queue.get(timeout=2)
        except Empty:
            continue
        threading.Thread(target=do_generate, kwargs=job, daemon=True).start()

threading.Thread(target=_gen_worker, daemon=True).start()

# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/gallery/<path:filename>")
def gallery_image(filename):
    gallery_dir = os.path.join(os.path.dirname(__file__), "gallery")
    return send_file(os.path.join(gallery_dir, filename), mimetype="image/png")

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    d         = request.json
    sid       = d.get("sid", "")
    user_text = d.get("message", "").strip()
    image_b64 = d.get("image_b64")
    gpt_key      = d.get("gpt_key", "").strip()
    gpt_model    = d.get("gpt_model", "gpt-5.4").strip() or "gpt-5.4"
    gpt_endpoint = d.get("gpt_endpoint", "").strip() or "https://api.frankai.cc/v1"

    if not sid or not user_text:
        return jsonify({"error": "missing sid or message"}), 400

    sess = get_session(sid)
    sess["history"].append({"role": "user", "content": user_text})

    try:
        reply = call_gpt4o(sess["history"], gpt_key, image_b64,
                           model=gpt_model, endpoint=gpt_endpoint)
    except Exception as e:
        sess["history"].pop()
        return jsonify({"error": str(e)}), 500

    sess["history"].append({"role": "assistant", "content": reply})
    # persist conversation
    preview = user_text[:60]
    db_upsert_conv(sess["conv_id"], sid, preview, sess["history"])

    parsed   = parse_prompts(reply)
    new_pids = []
    for p in parsed:
        pid = str(uuid.uuid4())
        sess["prompts"][pid] = {
            "title": p["title"], "desc": p.get("desc",""),
            "text": p["text"], "status": "pending", "images": [], "error": ""
        }
        new_pids.append(pid)

    display = re.sub(r'\[PROMPT\].*?\[/PROMPT\]', '', reply, flags=re.DOTALL).strip()
    return jsonify({
        "reply": display,
        "new_prompts": [
            {"pid": pid,
             "title": sess["prompts"][pid]["title"],
             "desc":  sess["prompts"][pid]["desc"],
             "text":  sess["prompts"][pid]["text"]}
            for pid in new_pids
        ]
    })

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    d            = request.json
    sid          = d.get("sid", "")
    user_text    = d.get("message", "").strip()
    image_b64    = d.get("image_b64")
    gpt_key      = d.get("gpt_key", "").strip()
    gpt_model    = d.get("gpt_model", "gpt-5.4").strip() or "gpt-5.4"
    gpt_endpoint = d.get("gpt_endpoint", "").strip() or "https://api.frankai.cc/v1"
    if not sid or not user_text:
        return jsonify({"error": "missing sid or message"}), 400
    sess = get_session(sid)
    sess["history"].append({"role": "user", "content": user_text})

    def generate():
        chunks = []
        try:
            for chunk in stream_gpt4o(sess["history"], gpt_key, image_b64,
                                      model=gpt_model, endpoint=gpt_endpoint):
                chunks.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            sess["history"].pop()
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return
        full_reply = "".join(chunks)
        if not full_reply:
            sess["history"].pop()
            yield f"event: error\ndata: {json.dumps({'error': 'API 返回了空内容，请检查 Key 和模型名'}, ensure_ascii=False)}\n\n"
            return
        sess["history"].append({"role": "assistant", "content": full_reply})
        db_upsert_conv(sess["conv_id"], sid, user_text[:60], sess["history"])
        parsed = parse_prompts(full_reply)
        new_pids = []
        for p in parsed:
            pid = str(uuid.uuid4())
            sess["prompts"][pid] = {
                "title": p["title"], "desc": p.get("desc", ""),
                "text":  p["text"],  "status": "pending", "images": [], "error": ""
            }
            new_pids.append(pid)
        display = re.sub(r'\[PROMPT\].*?\[/PROMPT\]', '', full_reply, flags=re.DOTALL).strip()
        done_payload = {
            "display": display,
            "new_prompts": [
                {"pid": pid, "title": sess["prompts"][pid]["title"],
                 "desc":  sess["prompts"][pid]["desc"],
                 "text":  sess["prompts"][pid]["text"]}
                for pid in new_pids
            ]
        }
        yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache",
                             "Connection": "keep-alive"})

@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    sid = (request.json or {}).get("sid", "")
    if sid in sessions:
        sessions[sid]["history"] = []
        sessions[sid]["conv_id"] = str(uuid.uuid4())
        sessions[sid]["prompts"] = {}
        sessions[sid]["images"]  = {}
    return jsonify({"ok": True})

@app.route("/api/prompt/create", methods=["POST"])
def api_prompt_create():
    d    = request.json
    sid  = d.get("sid", "")
    text = d.get("text", "").strip()
    if not sid or not text:
        return jsonify({"error": "missing params"}), 400
    sess  = get_session(sid)
    pid   = str(uuid.uuid4())
    title = text[:10]
    sess["prompts"][pid] = {
        "title": title, "desc": "", "text": text,
        "status": "pending", "images": [], "error": ""
    }
    return jsonify({"pid": pid, "title": title, "text": text})

@app.route("/api/prompt/<pid>", methods=["PATCH", "DELETE"])
def api_prompt(pid):
    d   = request.json or {}
    sid = d.get("sid") or request.args.get("sid", "")
    sess = get_session(sid)
    if pid not in sess["prompts"]:
        return jsonify({"error": "not found"}), 404
    if request.method == "DELETE":
        del sess["prompts"][pid]
        return jsonify({"ok": True})
    if "text"  in d: sess["prompts"][pid]["text"]  = d["text"]
    if "title" in d: sess["prompts"][pid]["title"] = d["title"]
    return jsonify({"ok": True})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    d          = request.json
    sid        = d.get("sid", "")
    pid        = d.get("pid")
    size       = d.get("size", "1024x1024")
    quality    = d.get("quality", "medium")
    n          = int(d.get("n", 1))
    frank_keys = d.get("frank_keys", [])
    ref_images = d.get("ref_images", [])   # list of base64 strings (max 4)

    sess = get_session(sid)
    if pid not in sess["prompts"]:
        return jsonify({"error": "not found"}), 404

    sess["prompts"][pid]["status"] = "queued"
    sess["prompts"][pid]["images"] = []
    gen_queue.put({
        "sid": sid, "pid": pid,
        "prompt_text": sess["prompts"][pid]["text"],
        "size": size, "quality": quality, "n": n,
        "frank_keys": frank_keys,
        "ref_images": ref_images
    })
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    sid  = request.args.get("sid", "")
    sess = get_session(sid)
    return jsonify({
        pid: {"status": v["status"], "images": v["images"],
              "error": v.get("error", "")}
        for pid, v in sess["prompts"].items()
    })

def _serve_image(iid, as_attachment=False):
    # 1. check in-memory (current session)
    for sess in sessions.values():
        if iid in sess.get("images", {}):
            raw   = base64.b64decode(sess["images"][iid]["b64"])
            fname = f"{sess['images'][iid]['ts']}_{iid[:8]}.png"
            return send_file(io.BytesIO(raw), mimetype="image/png",
                             as_attachment=as_attachment, download_name=fname)
    # 2. fall back to disk (historical)
    fpath = os.path.join("outputs", f"{iid}.png")
    if os.path.exists(fpath):
        return send_file(fpath, mimetype="image/png",
                         as_attachment=as_attachment,
                         download_name=f"{iid[:8]}.png")
    return "", 404

@app.route("/api/image/<sid>/<iid>")
def api_image(sid, iid):
    return _serve_image(iid)

@app.route("/api/image/<sid>/<iid>/download")
def api_image_download(sid, iid):
    return _serve_image(iid, as_attachment=True)

@app.route("/api/translate", methods=["POST"])
def api_translate():
    d            = request.json
    text         = d.get("text", "").strip()
    gpt_key      = d.get("gpt_key", "").strip()
    gpt_model    = d.get("gpt_model", "gpt-5.4").strip() or "gpt-5.4"
    gpt_endpoint = d.get("gpt_endpoint", "").strip() or "https://api.frankai.cc/v1"
    if not text or not gpt_key:
        return jsonify({"error": "缺少参数"}), 400
    try:
        result = call_gpt4o(
            [{"role": "user", "content":
              f"请将以下英文图像生成提示词完整翻译成中文，保留专业术语，只返回翻译内容，不要解释：\n\n{text}"}],
            gpt_key, model=gpt_model, endpoint=gpt_endpoint
        )
        return jsonify({"translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/restore", methods=["POST"])
def api_session_restore():
    """Load a historical conversation into the current session."""
    d   = request.json
    sid = d.get("sid", "")
    cid = d.get("cid", "")
    with _db_lock, _db() as c:
        row = c.execute(
            "SELECT msgs FROM conversations WHERE cid=? AND sid=?", (cid, sid)
        ).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    msgs = json.loads(row["msgs"])
    sess = get_session(sid)
    sess["history"]  = msgs
    sess["conv_id"]  = cid
    sess["prompts"]  = {}
    sess["images"]   = {}
    # re-parse prompts so cards can be recreated
    new_pids = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text","") for p in content if isinstance(p,dict))
        for p in parse_prompts(content):
            pid = str(uuid.uuid4())
            sess["prompts"][pid] = {
                "title": p["title"], "desc": p.get("desc",""),
                "text":  p["text"],  "status": "pending",
                "images": [], "error": ""
            }
            new_pids.append(pid)
    return jsonify({
        "messages": [
            {"role": m["role"],
             "content": m["content"] if isinstance(m["content"], str) else
                        next((p.get("text","") for p in m["content"]
                              if isinstance(p,dict) and p.get("type")=="text"), "")}
            for m in msgs if m.get("role") in ("user","assistant")
        ],
        "prompts": [
            {"pid": pid,
             "title": sess["prompts"][pid]["title"],
             "desc":  sess["prompts"][pid]["desc"],
             "text":  sess["prompts"][pid]["text"]}
            for pid in new_pids
        ]
    })

@app.route("/api/history/images")
def api_history_images():
    sid  = request.args.get("sid", "")
    page = int(request.args.get("page", 0))
    size = 24
    with _db_lock, _db() as c:
        rows = c.execute(
            "SELECT iid,title,prompt,ts FROM images WHERE sid=? ORDER BY created DESC LIMIT ? OFFSET ?",
            (sid, size, page * size)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/conversations")
def api_history_conversations():
    sid = request.args.get("sid", "")
    with _db_lock, _db() as c:
        rows = c.execute(
            "SELECT cid,preview,updated FROM conversations WHERE sid=? ORDER BY updated DESC LIMIT 50",
            (sid,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/conversation/<cid>")
def api_history_conversation(cid):
    sid = request.args.get("sid", "")
    with _db_lock, _db() as c:
        row = c.execute(
            "SELECT cid,preview,msgs,updated FROM conversations WHERE cid=? AND sid=?",
            (cid, sid)
        ).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    d = dict(row)
    d["msgs"] = json.loads(d["msgs"])
    return jsonify(d)

# ── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Image Studio</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}

/* ── Dark theme (default) ── */
:root{
  --bg:#212121;--sidebar-bg:#171717;--surface:#2f2f2f;--card:#3a3a3a;
  --border:#444;--accent:#ab68ff;--accent-hover:#9055e8;
  --text:#ececec;--text-secondary:#b0b0b8;--muted:#8e8ea0;
  --green:#19c37d;--red:#ef4444;--yellow:#f59e0b;
  --radius:16px;--sidebar-w:258px;
  --input-bg:#2f2f2f;--shadow:rgba(0,0,0,.4);
}
/* ── Light theme ── */
[data-theme="light"]{
  --bg:#ffffff;--sidebar-bg:#f5f5f5;--surface:#f0f0f0;--card:#e8e8e8;
  --border:#d8d8d8;--accent:#7c3aed;--accent-hover:#6d28d9;
  --text:#1a1a1a;--text-secondary:#4a4a5a;--muted:#6b6b80;
  --green:#059669;--red:#dc2626;--yellow:#d97706;
  --input-bg:#f5f5f5;--shadow:rgba(0,0,0,.12);
}
body{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  height:100vh;display:flex;flex-direction:row;overflow:hidden;
  transition:background .2s,color .2s}

/* ── sidebar ── */
.sidebar{width:var(--sidebar-w);flex-shrink:0;background:var(--sidebar-bg);
  display:flex;flex-direction:column;height:100vh;
  border-right:1px solid var(--border);transition:background .2s}
.sidebar-head{display:flex;align-items:center;gap:8px;padding:14px 12px 10px;flex-shrink:0}
.logo{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;flex:1;color:var(--text)}
.logo-icon{width:26px;height:26px;background:linear-gradient(135deg,#7c3aed,#a78bfa);
  border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
.new-chat-btn{width:30px;height:30px;border-radius:8px;background:none;
  border:1px solid var(--border);color:var(--muted);cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:15px;
  transition:.15s;flex-shrink:0}
.new-chat-btn:hover{border-color:var(--accent);color:var(--accent)}
.side-tabs{display:flex;flex-shrink:0;padding:0 10px 4px;gap:2px}
.stab{flex:1;padding:6px 0;font-size:12px;cursor:pointer;color:var(--muted);
  background:none;border:none;border-bottom:2px solid transparent;
  transition:.15s;font-family:inherit;font-weight:500}
.stab.active{color:var(--accent);border-bottom-color:var(--accent)}
.side-body{flex:1;overflow-y:auto;padding:4px 8px 8px}
.side-section-label{font-size:10px;color:var(--muted);padding:8px 6px 3px;
  font-weight:600;letter-spacing:.05em;text-transform:uppercase}
.side-conv-item{padding:8px 10px;border-radius:8px;cursor:pointer;
  transition:.15s;margin-bottom:1px}
.side-conv-item:hover{background:var(--surface)}
.side-conv-item.active{background:rgba(124,58,237,.15)}
.side-conv-preview{font-size:13px;color:var(--text);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;line-height:1.4}
.side-img-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.side-img-item{border-radius:6px;overflow:hidden;cursor:pointer;aspect-ratio:1;
  background:var(--card);position:relative}
.side-img-item img{width:100%;height:100%;object-fit:cover;display:block;transition:.2s}
.side-img-item:hover img{opacity:.8}
.side-empty{color:var(--muted);text-align:center;padding:28px 10px;font-size:12px;line-height:1.6}
.sidebar-foot{flex-shrink:0;padding:8px 10px;border-top:1px solid var(--border);display:flex;gap:6px}
.foot-btn{flex:1;background:none;border:1px solid var(--border);color:var(--muted);
  border-radius:8px;padding:7px 6px;font-size:11px;cursor:pointer;transition:.15s;
  display:flex;align-items:center;justify-content:center;gap:4px}
.foot-btn:hover{border-color:var(--accent);color:var(--text)}
.theme-btn{width:32px;height:32px;padding:0;flex:none}

/* ── main area ── */
.main-area{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden}
.topbar{display:flex;align-items:center;padding:10px 18px;flex-shrink:0;
  border-bottom:1px solid transparent}
.topbar-title{font-size:14px;font-weight:500;color:var(--muted);flex:1;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.icon-btn{background:none;border:none;color:var(--muted);font-size:18px;cursor:pointer;
  width:34px;height:34px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;transition:.15s}
.icon-btn:hover{background:var(--surface);color:var(--text)}

/* conversation */
.conv-wrap{flex:1;overflow-y:auto;display:flex;flex-direction:column;
  align-items:center;padding:0 16px 190px}
.conv{width:100%;max-width:760px}

/* messages */
.msg-group{padding:16px 0}
.msg-user-row{display:flex;justify-content:flex-end}
.msg-user-bubble{background:var(--surface);border-radius:18px 18px 4px 18px;
  padding:12px 16px;max-width:70%;font-size:15px;line-height:1.6;
  white-space:pre-wrap;word-break:break-word}
.msg-user-img{display:flex;justify-content:flex-end;margin-bottom:6px}
.msg-user-img img{max-height:120px;border-radius:12px;object-fit:cover}
.msg-ai-row{display:flex;gap:12px;align-items:flex-start}
.ai-avatar{width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,#7c3aed,#a78bfa);
  display:flex;align-items:center;justify-content:center;
  font-size:13px;flex-shrink:0;margin-top:2px}
.ai-body{flex:1;min-width:0}
.msg-ai-text{font-size:15px;line-height:1.75;word-break:break-word}
.msg-ai-text p{margin-bottom:8px}
.msg-ai-text p:last-child{margin-bottom:0}
.msg-ai-text h1,.msg-ai-text h2,.msg-ai-text h3{font-weight:700;margin:12px 0 6px;color:var(--text)}
.msg-ai-text h1{font-size:17px}.msg-ai-text h2{font-size:15px}.msg-ai-text h3{font-size:14px}
.msg-ai-text ul,.msg-ai-text ol{padding-left:20px;margin-bottom:8px}
.msg-ai-text li{margin-bottom:3px;line-height:1.6}
.msg-ai-text strong{color:#e2e8f0;font-weight:600}
.msg-ai-text em{color:var(--muted);font-style:italic}
.msg-ai-text code{background:var(--bg);border:1px solid var(--border);border-radius:4px;
  padding:1px 5px;font-family:monospace;font-size:13px;color:#a78bfa}
.msg-ai-text pre{background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;overflow-x:auto;margin-bottom:8px}
.msg-ai-text pre code{background:none;border:none;padding:0;color:#e2e8f0;font-size:13px}
.msg-ai-text blockquote{border-left:3px solid var(--accent);padding-left:12px;
  color:var(--muted);margin:6px 0}
.msg-ai-text hr{border:none;border-top:1px solid var(--border);margin:10px 0}
.msg-ai-text a{color:var(--accent);text-decoration:none}
.msg-ai-text a:hover{text-decoration:underline}
.msg-ai-text table{border-collapse:collapse;width:100%;margin-bottom:8px;font-size:13px}
.msg-ai-text th,.msg-ai-text td{border:1px solid var(--border);padding:5px 10px;text-align:left}
.msg-ai-text th{background:var(--surface);font-weight:600}

/* prompt cards */
.prompt-cards{display:flex;flex-direction:column;gap:10px;margin-top:12px}
.pcard{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;transition:.2s border-color}
.pcard:hover{border-color:#666}
.pcard-header{padding:12px 16px 0;display:flex;align-items:center;gap:8px}
.pcard-num{width:22px;height:22px;border-radius:50%;background:var(--accent);
  display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;flex-shrink:0}
.pcard-title{font-weight:600;font-size:14px;flex:1}
.status-badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500}
.sb-pending{background:#3a3a3a;color:var(--muted)}
.sb-queued,.sb-generating{background:#422e00;color:var(--yellow)}
.sb-done{background:#0a2e1a;color:var(--green)}
.sb-error{background:#2e0a0a;color:var(--red)}
.pcard-prompt{margin:10px 16px;background:var(--bg);border:1px solid var(--border);
  border-radius:10px;padding:10px 12px;font-size:13px;line-height:1.6;
  color:#ccc;min-height:60px;outline:none;word-break:break-word;white-space:pre-wrap}
.pcard-prompt[contenteditable=true]:focus{border-color:var(--accent);color:var(--text)}
.pcard-controls{padding:8px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
/* global ref images bar */
.global-ref-wrap{margin-top:12px;background:var(--card);border:1px solid var(--border);
  border-radius:12px;overflow:hidden}
.global-ref-bar{padding:10px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.global-ref-label{font-size:13px;color:var(--muted);flex:1}
.global-ref-btn{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:5px 12px;font-size:12px;cursor:pointer;transition:.15s;
  white-space:nowrap;flex-shrink:0}
.global-ref-btn:hover{border-color:var(--accent);color:var(--accent)}
.global-ref-thumbs{display:flex;gap:8px;flex-wrap:wrap;padding:0 14px 12px}
.global-ref-thumbs:empty{display:none}
/* shared thumb style */
.ref-thumb{position:relative;flex-shrink:0}
.ref-thumb img{width:56px;height:56px;object-fit:cover;border-radius:8px;
  border:2px solid var(--border)}
.ref-thumb .rm-ref{position:absolute;top:-5px;right:-5px;width:16px;height:16px;
  background:var(--red);border-radius:50%;font-size:9px;display:flex;align-items:center;
  justify-content:center;cursor:pointer;color:#fff;line-height:1}
/* prompt card desc */
.pcard-desc{margin:4px 16px 0;font-size:12px;color:var(--muted);line-height:1.5;
  font-style:italic}
.pcard-translate-btn{margin:6px 16px 0;display:inline-flex;align-items:center;gap:4px;
  font-size:11px;color:var(--accent);cursor:pointer;background:none;border:none;
  padding:0;font-family:inherit;transition:.15s;opacity:.8}
.pcard-translate-btn:hover{opacity:1;text-decoration:underline}
.pcard-translation{margin:6px 16px 0;padding:8px 10px;background:rgba(124,58,237,.07);
  border-left:2px solid var(--accent);border-radius:0 6px 6px 0;
  font-size:12px;color:var(--text-secondary);line-height:1.7;display:none}
.pcard-translation.show{display:block}
.pcard-del-btn{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:13px;padding:2px 4px;border-radius:5px;transition:.15s;flex-shrink:0;
  margin-left:4px;line-height:1}
.pcard-del-btn:hover{background:rgba(239,68,68,.15);color:var(--red)}
/* card-level ref images */
.card-ref-zone{margin:8px 16px 0;padding:7px 10px;background:rgba(171,104,255,.06);
  border:1px dashed rgba(171,104,255,.3);border-radius:9px}
.card-ref-bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.card-ref-hint{font-size:11px;color:var(--muted);flex:1}
.card-ref-btn{font-size:11px;background:var(--card);border:1px solid var(--border);
  color:var(--text);border-radius:6px;padding:3px 8px;cursor:pointer;transition:.15s;white-space:nowrap}
.card-ref-btn:hover{border-color:var(--accent);color:var(--accent)}
.card-ref-clear{font-size:11px;color:var(--muted);cursor:pointer;transition:.15s}
.card-ref-clear:hover{color:var(--red)}
.card-ref-thumbs{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.card-ref-thumbs:empty{display:none}
.card-ref-thumbs .ref-thumb img{width:40px;height:40px}
select{background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:5px 10px;font-size:12px;outline:none;cursor:pointer}
select:focus{border-color:var(--accent)}
.pcard-actions{padding:10px 16px 14px;display:flex;gap:8px}
.btn-gen{background:var(--accent);color:#fff;border:none;border-radius:10px;
  padding:8px 18px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
.btn-gen:hover{background:#9055e8}
.btn-del{background:none;border:1px solid #5a1f1f;color:var(--red);
  border-radius:10px;padding:8px 14px;font-size:13px;cursor:pointer;transition:.15s}
.btn-del:hover{background:#3d1515;border-color:var(--red)}
.err-text{padding:4px 16px 10px;font-size:12px;color:var(--red)}
.img-result-grid{display:grid;gap:8px;padding:12px 16px 16px;
  grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.img-result-item{position:relative;border-radius:12px;overflow:hidden;
  cursor:pointer;background:var(--bg)}
.img-result-item img{width:100%;display:block;transition:.2s}
.img-result-item:hover img{opacity:.9}
.img-overlay{position:absolute;bottom:0;left:0;right:0;padding:10px 12px;
  background:linear-gradient(transparent,rgba(0,0,0,.8));
  display:flex;gap:6px;opacity:0;transition:.2s;justify-content:flex-end}
.img-result-item:hover .img-overlay{opacity:1}
.img-action-btn{background:rgba(255,255,255,.15);backdrop-filter:blur(8px);
  border:none;color:#fff;border-radius:7px;padding:5px 10px;
  font-size:11px;cursor:pointer;transition:.15s}
.img-action-btn:hover{background:rgba(255,255,255,.25)}
.gen-spinner{padding:20px 16px;display:flex;gap:12px;align-items:center}
.spinner{width:22px;height:22px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner-txt{font-size:13px;color:var(--muted)}
.thinking{display:flex;gap:4px;align-items:center;padding:4px 0}
.dot{width:7px;height:7px;border-radius:50%;background:var(--muted);animation:bounce .9s infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}

/* welcome */
.welcome{padding:48px 20px 20px;text-align:center}
.welcome h2{font-size:24px;font-weight:700;margin-bottom:8px}
.welcome p{color:var(--muted);font-size:14px;line-height:1.7}
.gallery-chips{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;
  max-width:560px;margin:22px auto 0}
.gallery-chip{position:relative;border-radius:12px;overflow:hidden;cursor:pointer;
  aspect-ratio:4/3;background:var(--card);border:2px solid transparent;transition:.2s}
.gallery-chip:hover{border-color:var(--accent);transform:translateY(-2px)}
.gallery-chip img{width:100%;height:100%;object-fit:cover;display:block;transition:.2s}
.gallery-chip:hover img{opacity:.85}
.gallery-chip-label{position:absolute;bottom:0;left:0;right:0;
  padding:22px 10px 8px;
  background:linear-gradient(transparent,rgba(0,0,0,.72));
  font-size:12px;font-weight:500;color:#fff;line-height:1.3;text-align:left}

/* input bar */
.input-bar{position:fixed;bottom:0;left:var(--sidebar-w);right:0;display:flex;
  justify-content:center;padding:12px 16px 20px;
  background:linear-gradient(transparent,var(--bg) 40%);pointer-events:none}
.input-inner{width:100%;max-width:760px;pointer-events:all}
.input-box{background:var(--input-bg);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;transition:.2s border-color,.2s background}
.input-box:focus-within{border-color:var(--muted)}
.ref-row{padding:8px 12px 0;display:none}
.ref-row img{height:52px;border-radius:8px;object-fit:cover}
.ref-rm{display:inline-flex;align-items:center;gap:4px;margin-left:6px;
  font-size:11px;color:var(--muted);cursor:pointer;vertical-align:middle}
.ref-rm:hover{color:var(--red)}
.input-row{display:flex;align-items:flex-end;gap:4px;padding:8px 8px 8px 14px}
#msgInput{flex:1;background:none;border:none;color:var(--text);
  font-size:15px;resize:none;outline:none;max-height:160px;
  font-family:inherit;line-height:1.55;padding:4px 0}
#msgInput::placeholder{color:var(--muted)}
.iactions{display:flex;gap:4px;align-items:flex-end;flex-shrink:0}
.mode-pill{flex-shrink:0;display:flex;align-items:center;gap:5px;
  padding:5px 9px;border-radius:7px;border:1px solid var(--border);
  color:var(--muted);font-size:11px;cursor:pointer;transition:.15s;
  background:none;font-family:inherit;white-space:nowrap;position:relative}
.mode-pill:hover{border-color:var(--accent);color:var(--text)}
.mode-pill.gen-mode{border-color:rgba(124,58,237,.4);color:var(--accent);
  background:rgba(124,58,237,.08)}
.send-menu{position:fixed;background:var(--sidebar-bg);border:1px solid var(--border);
  border-radius:10px;min-width:160px;box-shadow:0 6px 20px var(--shadow);
  display:none;z-index:200;overflow:hidden}
.send-menu.open{display:block}
.send-menu-item{padding:9px 14px;font-size:13px;cursor:pointer;
  display:flex;align-items:center;gap:8px;color:var(--text-secondary);transition:.15s}
.send-menu-item:hover{background:var(--surface);color:var(--text)}
.send-menu-item.active{color:var(--accent)}
.attach-btn{width:34px;height:34px;border-radius:9px;background:var(--card);
  border:1px solid var(--border);color:var(--muted);cursor:pointer;display:flex;
  align-items:center;justify-content:center;font-size:16px;transition:.15s}
.attach-btn:hover{border-color:var(--accent);color:var(--accent)}
.chat-ref-row{padding:8px 12px 0;display:none;align-items:center;gap:8px}
.chat-ref-row img{height:52px;border-radius:8px;object-fit:cover}
.send-btn{width:34px;height:34px;border-radius:9px;background:var(--accent);
  border:none;color:#fff;cursor:pointer;display:flex;align-items:center;
  justify-content:center;font-size:16px;transition:.15s;flex-shrink:0}
.send-btn:hover{background:#9055e8}
.send-btn:disabled{background:var(--border);cursor:not-allowed}
.input-hint{font-size:11px;color:var(--muted);text-align:center;
  margin-top:6px;padding:0 4px}

/* config modal */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;
  align-items:center;justify-content:center;z-index:999;backdrop-filter:blur(6px)}
.modal-bg.open{display:flex}
.modal{background:var(--sidebar-bg);border:1px solid var(--border);border-radius:18px;
  padding:26px;width:460px;max-width:95vw;box-shadow:0 20px 60px var(--shadow)}
.modal h3{font-size:17px;font-weight:700;margin-bottom:4px}
.modal-sub{font-size:12px;color:var(--muted);margin-bottom:20px;line-height:1.5}
.key-section{margin-bottom:18px}
.key-section h4{font-size:11px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.frow{margin-bottom:10px;display:flex;align-items:center;gap:8px}
.frow label{width:56px;font-size:12px;color:var(--muted);flex-shrink:0}
.frow input{flex:1;background:var(--bg);border:1px solid var(--border);
  color:var(--text);border-radius:9px;padding:8px 11px;
  font-size:13px;outline:none;font-family:monospace;transition:.15s background}
.frow input:focus{border-color:var(--accent)}
.key-tip{font-size:11px;color:var(--muted);margin-top:6px;
  padding:8px 10px;background:rgba(255,255,255,.04);
  border-radius:7px;line-height:1.5}
.key-tip a{color:var(--accent)}
.modal-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:20px}
.model-opts{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.model-opt{display:flex;align-items:center;gap:5px;padding:5px 14px;
  background:var(--bg);border:1px solid var(--border);border-radius:8px;
  cursor:pointer;font-size:12px;transition:.15s;font-family:monospace;color:var(--muted)}
.model-opt.selected{border-color:var(--accent);color:var(--accent);background:rgba(124,58,237,.1)}
.model-opt input[type=radio]{display:none}
.btn-cancel{background:none;border:1px solid var(--border);color:var(--muted);
  border-radius:9px;padding:8px 16px;font-size:13px;cursor:pointer;transition:.15s}
.btn-cancel:hover{border-color:var(--muted);color:var(--text)}
.btn-save{background:var(--accent);color:#fff;border:none;
  border-radius:9px;padding:8px 20px;font-size:13px;font-weight:600;
  cursor:pointer;transition:.15s}
.btn-save:hover{background:#9055e8}

/* lightbox */
/* load-more sidebar button */
.side-load-more{text-align:center;padding:6px 0}
.side-load-more button{background:none;border:1px solid var(--border);color:var(--muted);
  border-radius:7px;padding:5px 14px;font-size:11px;cursor:pointer;transition:.15s;font-family:inherit}
.side-load-more button:hover{border-color:var(--accent);color:var(--text)}

.lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;
  align-items:center;justify-content:center;z-index:1000;gap:0}
.lb.open{display:flex}
.lb-prompt-panel{width:260px;flex-shrink:0;padding:24px 20px;
  background:rgba(255,255,255,.06);border-radius:16px 0 0 16px;
  align-self:center;display:none;max-height:80vh;overflow-y:auto}
.lb-prompt-panel.visible{display:block}
.lb-prompt-label{font-size:10px;color:rgba(255,255,255,.45);font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px}
.lb-prompt-text{font-size:12px;color:rgba(255,255,255,.82);line-height:1.75;
  word-break:break-word;white-space:pre-wrap}
.lb-use-btn{margin-top:14px;width:100%;background:rgba(171,104,255,.25);
  border:1px solid rgba(171,104,255,.5);color:#d4aaff;border-radius:8px;
  padding:7px 12px;font-size:12px;cursor:pointer;transition:.15s;font-family:inherit}
.lb-use-btn:hover{background:rgba(171,104,255,.4);color:#fff}
.lb-main{display:flex;flex-direction:column;align-items:center;gap:14px}
.lb-main img{max-width:min(70vw,960px);max-height:82vh;border-radius:16px;object-fit:contain}
.lb-prompt-panel.visible ~ .lb-main img{border-radius:0 16px 16px 0}
.lb-actions{display:flex;gap:10px}
.lb-dl{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  color:#fff;border-radius:10px;padding:8px 18px;font-size:13px;
  text-decoration:none;transition:.15s}
.lb-dl:hover{background:rgba(255,255,255,.2)}
.lb-close{background:none;border:1px solid rgba(255,255,255,.2);
  color:#fff;border-radius:10px;padding:8px 18px;font-size:13px;cursor:pointer;transition:.15s}
.lb-close:hover{background:rgba(255,255,255,.1)}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-track{background:transparent}
</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>

<!-- Left sidebar -->
<div class="sidebar">
  <div class="sidebar-head">
    <div class="logo"><div class="logo-icon">✦</div>AI Image Studio</div>
    <button class="new-chat-btn" onclick="clearConv()" title="新对话">✏</button>
  </div>
  <div class="side-tabs">
    <button class="stab active" id="stab-conv" onclick="switchSideTab('conv')">💬 对话</button>
    <button class="stab" id="stab-img" onclick="switchSideTab('img')">🖼 图库</button>
  </div>
  <div class="side-body" id="sbody-conv">
    <div id="sideConvList"></div>
    <div class="side-empty" id="sideConvEmpty" style="display:none">还没有历史对话</div>
  </div>
  <div class="side-body" id="sbody-img" style="display:none">
    <div class="side-img-grid" id="sideImgGrid"></div>
    <div class="side-load-more" id="sideImgMore" style="display:none">
      <button onclick="loadMoreSideImages()">加载更多</button>
    </div>
    <div class="side-empty" id="sideImgEmpty">还没有生成过图片</div>
  </div>
  <div class="sidebar-foot">
    <button class="foot-btn" id="keyBtn" onclick="openConfig()">🔑 API Key</button>
    <button class="foot-btn theme-btn" onclick="toggleTheme()" title="切换亮/暗模式" id="themeBtn">🌙</button>
  </div>
</div>

<!-- Main area -->
<div class="main-area">
  <div class="topbar">
    <span class="topbar-title" id="topbarTitle">AI 图像创作助手</span>
  </div>

  <div class="conv-wrap" id="convWrap">
    <div class="conv" id="conv">
      <div class="welcome" id="welcome">
        <h2>AI 图像创作助手</h2>
        <p>与 ChatGPT 沟通你的创意，由 GPT-Image-2 高质量生成<br>
           首次使用请点击左下角 🔑 填写 API Key</p>
        <div class="gallery-chips" id="welcomeChips"></div>
      </div>
    </div>
  </div>

  <div class="input-bar">
  <div class="input-inner">
    <div class="input-box">
      <div class="chat-ref-row" id="chatRefRow">
        <img id="chatRefImg" src="" alt="">
        <span class="ref-rm" onclick="removeChatRef()">✕ 移除</span>
      </div>
      <div class="input-row">
        <button class="mode-pill" id="modePill" onclick="toggleSendMenu(event)" title="切换发送模式">
          💬 <span id="modeLabel">对话</span>
        </button>
        <textarea id="msgInput" rows="1" placeholder="描述你想要的图像，或上传参考图让 ChatGPT 分析…"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <div class="iactions">
          <label class="attach-btn" title="上传参考图（ChatGPT 识图分析）">
            🖼<input type="file" id="chatFileInput" accept="image/*" style="display:none"
               onchange="onChatImagePicked(this)">
          </label>
          <button class="send-btn" id="sendBtn" onclick="sendOrQuick()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="send-menu" id="sendMenu">
        <div class="send-menu-item active" id="smi-chat" onclick="setSendMode('chat')">
          <span>💬</span> ChatGPT 对话
        </div>
        <div class="send-menu-item" id="smi-gen" onclick="setSendMode('gen')">
          <span>🎨</span> 直接生图
        </div>
      </div>
    </div>
    <div class="input-hint">Key 仅保存在你的浏览器中，不上传服务器</div>
  </div>
</div>

</div><!-- /main-area -->

<!-- config modal -->
<div class="modal-bg" id="cfgModal">
  <div class="modal">
    <h3>🔑 API Key 配置</h3>
    <p class="modal-sub">Key 只保存在你的浏览器，不上传服务器</p>

    <div class="key-section">
      <h4>ChatGPT Key — 对话 &amp; 识图</h4>
      <div class="frow">
        <label>Key</label>
        <input id="cfgGpt" type="password" placeholder="sk-…（必填）" autocomplete="off">
      </div>
      <div class="frow" style="align-items:flex-start">
        <label style="padding-top:6px">模型</label>
        <div class="model-opts" id="modelOpts">
          <label class="model-opt selected" data-val="gpt-5.4">
            <input type="radio" name="gptModel" value="gpt-5.4" checked>gpt-5.4
          </label>
        </div>
      </div>
    </div>

    <div class="key-section">
      <h4>GPT-IMAGE-2 生图 Key（最多 3 个并发）</h4>
      <div class="frow"><label>Key 1</label><input id="cfgF0" type="password" placeholder="sk-…（必填）" autocomplete="off"></div>
      <div class="frow"><label>Key 2</label><input id="cfgF1" type="password" placeholder="sk-…（可选）" autocomplete="off"></div>
      <div class="frow"><label>Key 3</label><input id="cfgF2" type="password" placeholder="sk-…（可选）" autocomplete="off"></div>
    </div>

    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeCfg()">取消</button>
      <button class="btn-save" onclick="saveCfg()">保存</button>
    </div>
  </div>
</div>

<!-- lightbox -->
<div class="lb" id="lb" onclick="closeLb()">
  <div class="lb-prompt-panel" id="lbPromptPanel" onclick="event.stopPropagation()">
    <div class="lb-prompt-label">Prompt</div>
    <div class="lb-prompt-text" id="lbPromptText"></div>
    <button class="lb-use-btn" id="lbUseBtn" onclick="useLbPrompt()" style="display:none">↳ 使用此提示词</button>
  </div>
  <div class="lb-main" onclick="event.stopPropagation()">
    <img id="lbImg" src="" alt="">
    <div class="lb-actions">
      <a id="lbDl" href="#" class="lb-dl" download>⬇ 下载原图</a>
      <button class="lb-close" onclick="closeLb()">关闭</button>
    </div>
  </div>
</div>

<script>
// ── markdown renderer ─────────────────────────────────────────────────────────
function md(text) {
  if (typeof marked !== 'undefined') {
    marked.setOptions({breaks: true, gfm: true});
    return marked.parse(text);
  }
  return esc(text).replace(/\n/g, '<br>');
}

// ── theme ──────────────────────────────────────────────────────────────────────
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeBtn').textContent = t === 'dark' ? '🌙' : '☀️';
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', next); applyTheme(next);
}
applyTheme(localStorage.getItem('theme') || 'dark');

// ── session id (per browser) ──────────────────────────────────────────────────
let SID = localStorage.getItem('sid');
if (!SID) { SID = crypto.randomUUID(); localStorage.setItem('sid', SID); }

// ── keys ──────────────────────────────────────────────────────────────────────
const K = () => JSON.parse(localStorage.getItem('apikeys') || '{"gpt":"","gpt_model":"gpt-5.4","gpt_endpoint":"","fk":["","",""]}');
const hasKeys = () => { const k=K(); return !!(k.gpt && k.fk.some(x=>x && x.trim())); };
const setKeyBtn = () => {
  const ok = hasKeys();
  const btn = document.getElementById('keyBtn');
  btn.textContent = ok ? '🟢 已配置' : '🔑 API Key';
};
setKeyBtn();

function openConfig() {
  const k = K();
  document.getElementById('cfgGpt').value = k.gpt || '';
  // highlight selected model chip
  const saved = k.gpt_model || 'gpt-5.4';
  document.querySelectorAll('#modelOpts .model-opt').forEach(el => {
    const val = el.dataset.val;
    el.classList.toggle('selected', val === saved);
    el.querySelector('input').checked = (val === saved);
  });
  [0,1,2].forEach(i => { document.getElementById('cfgF'+i).value = (k.fk||[])[i] || ''; });
  document.getElementById('cfgModal').classList.add('open');
}
function closeCfg() { document.getElementById('cfgModal').classList.remove('open'); }
function saveCfg() {
  const sel = document.querySelector('#modelOpts input[type=radio]:checked');
  const cfg = {
    gpt:          document.getElementById('cfgGpt').value.trim(),
    gpt_model:    sel ? sel.value : 'gpt-5.4',
    gpt_endpoint: K().gpt_endpoint || '',   // preserve existing, not shown
    fk:        [0,1,2].map(i => document.getElementById('cfgF'+i).value.trim())
  };
  localStorage.setItem('apikeys', JSON.stringify(cfg));
  closeCfg(); setKeyBtn();
  appendSys('✅ API Key 已保存到本地');
}
// model chip click → toggle selected style
document.addEventListener('click', e => {
  const opt = e.target.closest('#modelOpts .model-opt');
  if (!opt) return;
  document.querySelectorAll('#modelOpts .model-opt').forEach(el=>el.classList.remove('selected'));
  opt.classList.add('selected');
  opt.querySelector('input').checked = true;
});

// ── state ─────────────────────────────────────────────────────────────────────
let prompts      = {};
let pollTimer    = null;
let msgCnt       = 0;
let activeCid    = null;
let sideTabCur   = 'conv';
let sideImgPage  = 0;

// ── conv helpers ──────────────────────────────────────────────────────────────
function hideWelcome() { document.getElementById('welcome')?.remove(); }

function appendUserMsg(text, imgUrl) {
  hideWelcome();
  const conv = document.getElementById('conv');
  const g = document.createElement('div');
  g.className = 'msg-group';
  if (imgUrl) g.innerHTML = `<div class="msg-user-img"><img src="${imgUrl}"></div>`;
  g.innerHTML += `<div class="msg-user-row"><div class="msg-user-bubble">${esc(text)}</div></div>`;
  conv.appendChild(g); scrollDown();
}

function appendAiMsg(text, newPrompts) {
  const conv = document.getElementById('conv');
  const cid  = 'pc' + (++msgCnt);
  const g = document.createElement('div');
  g.className = 'msg-group msg-ai-row';
  g.innerHTML = `
    <div class="ai-avatar">✦</div>
    <div class="ai-body">
      ${text ? `<div class="msg-ai-text">${md(text)}</div>` : ''}
      ${newPrompts?.length ? `
        <div class="global-ref-wrap" id="grw-${cid}">
          <div class="global-ref-bar">
            <span class="global-ref-label">🖼 批量参考图（选填，最多4张，应用到所有卡片）</span>
            <label class="global-ref-btn">
              + 上传图片
              <input type="file" accept="image/*" multiple style="display:none"
                onchange="addGlobalRef('${cid}',this)">
            </label>
          </div>
          <div class="global-ref-thumbs" id="grt-${cid}"></div>
        </div>
        <div class="prompt-cards" id="${cid}"></div>` : ''}
    </div>`;
  conv.appendChild(g);
  if (newPrompts?.length) {
    const box = document.getElementById(cid);
    newPrompts.forEach((p, i) => {
      prompts[p.pid] = {title:p.title, desc:p.desc||'', text:p.text,
                        status:'pending', images:[], error:'', _groupId: cid};
      const card = buildCard(p.pid, i+1);
      box.appendChild(card);
      prompts[p.pid]._el = card;
    });
  }
  scrollDown();
}

function appendThinking() {
  hideWelcome();
  const g = document.createElement('div');
  g.className = 'msg-group msg-ai-row'; g.id = 'thinking';
  g.innerHTML = `<div class="ai-avatar">✦</div>
    <div class="ai-body"><div class="thinking">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div></div>`;
  document.getElementById('conv').appendChild(g); scrollDown();
}
function removeThinking() { document.getElementById('thinking')?.remove(); }

function appendSys(t) {
  const el = document.createElement('div');
  el.style.cssText = 'text-align:center;font-size:12px;color:var(--muted);padding:8px';
  el.textContent = t;
  document.getElementById('conv').appendChild(el); scrollDown();
}

function scrollDown() {
  const w = document.getElementById('convWrap');
  w.scrollTop = w.scrollHeight;
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── prompt cards ──────────────────────────────────────────────────────────────
function buildCard(pid, num) {
  const card = document.createElement('div');
  card.className = 'pcard'; card.id = 'card-'+pid; card.dataset.num = num;
  renderCard(card, pid); return card;
}

function renderCard(card, pid) {
  const p   = prompts[pid];
  const num = card.dataset.num || 1;
  const sc  = {pending:'sb-pending',queued:'sb-queued',generating:'sb-generating',done:'sb-done',error:'sb-error'}[p.status]||'sb-pending';
  const sl  = {pending:'待生成',queued:'排队中',generating:'生成中…',done:'✓ 完成',error:'✗ 失败'}[p.status]||p.status;
  const can = ['pending','done','error'].includes(p.status);
  // Preserve user-selected values across re-renders
  const curSz = document.getElementById('sz-'+pid)?.value || p._sz || 'auto';
  const curQl = document.getElementById('ql-'+pid)?.value || p._ql || 'medium';
  const curNn = document.getElementById('nn-'+pid)?.value || '1 张';
  const szOpts = [
    ['1024x1024','1024×1024 正方形'],['1024x1536','1024×1536 竖版'],
    ['1536x1024','1536×1024 横版'],['2048x2048','2048×2048 大图'],['auto','auto 自动']
  ].map(([v,l])=>`<option value="${v}"${curSz===v?' selected':''}>${l}</option>`).join('');
  const qlOpts = [
    ['low','低质量 ¥0.007'],['medium','中质量 ¥0.06'],['high','高质量 ¥0.26（容易卡顿）']
  ].map(([v,l])=>`<option value="${v}"${curQl===v?' selected':''}>${l}</option>`).join('');
  const nnOpts = ['1 张','2 张','3 张','4 张']
    .map(l=>`<option${curNn===l?' selected':''}>${l}</option>`).join('');
  const promptJson = JSON.stringify(p.text || '');
  const thumbs = (p.images||[]).map(iid => `
    <div class="img-result-item" onclick="openLb('${iid}',${promptJson})">
      <img src="/api/image/${SID}/${iid}" loading="lazy">
      <div class="img-overlay">
        <a href="/api/image/${SID}/${iid}/download" download
           onclick="event.stopPropagation()" class="img-action-btn">⬇ 下载</a>
      </div>
    </div>`).join('');

  card.innerHTML = `
    <div class="pcard-header">
      <div class="pcard-num">${num}</div>
      <div class="pcard-title">${esc(p.title)}</div>
      <span class="status-badge ${sc}">${sl}</span>
      <button class="pcard-del-btn" onclick="delCard('${pid}')" title="删除此提示词">✕</button>
    </div>
    <div class="pcard-prompt" id="pt-${pid}" contenteditable="${can}"
         onblur="saveEdit('${pid}')">${esc(p.text)}</div>
    ${p.desc?`<div class="pcard-desc">中文说明：${esc(p.desc)}</div>`:''}
    <button class="pcard-translate-btn" onclick="translateCard('${pid}',this)">⟳ 全部翻译</button>
    <div class="pcard-translation" id="tr-${pid}"></div>
    ${p.error?`<div class="err-text">❌ ${esc(p.error)}</div>`:''}
    <div class="card-ref-zone">
      <div class="card-ref-bar">
        <span class="card-ref-hint" id="crh-${pid}">📎 单独上传参考图（覆盖全局）</span>
        <label class="card-ref-btn">
          + 上传
          <input type="file" accept="image/*" multiple style="display:none"
            onchange="addCardRef('${pid}',this)">
        </label>
        <span class="card-ref-clear" onclick="clearCardRef('${pid}')">清除</span>
      </div>
      <div class="card-ref-thumbs" id="crt-${pid}"></div>
    </div>
    <div class="pcard-controls">
      <select id="sz-${pid}" onchange="prompts['${pid}']._sz=this.value">${szOpts}</select>
      <select id="ql-${pid}" onchange="prompts['${pid}']._ql=this.value">${qlOpts}</select>
      <select id="nn-${pid}">${nnOpts}</select>
    </div>
    <div class="pcard-actions">
      ${can?`<button class="btn-gen" onclick="doGen('${pid}')">▶ 生成图像</button>`:''}
    </div>
    ${['queued','generating'].includes(p.status)?`
      <div class="gen-spinner"><div class="spinner"></div>
        <span class="spinner-txt">${p.status==='queued'?'排队等待中…':'图像生成中，请稍候…'}</span>
      </div>`:''}
    ${thumbs?`<div class="img-result-grid">${thumbs}</div>`:''}`;
}

async function translateCard(pid, btn) {
  const k = K();
  if (!k.gpt) { openConfig(); return; }
  const trEl = document.getElementById('tr-'+pid);
  if (!trEl) return;
  if (trEl.classList.contains('show')) {
    trEl.classList.remove('show'); btn.textContent = '⟳ 全部翻译'; return;
  }
  btn.textContent = '翻译中…'; btn.disabled = true;
  try {
    const res = await fetch('/api/translate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        text: prompts[pid]?.text || '',
        gpt_key: k.gpt, gpt_model: k.gpt_model||'gpt-5.4',
        gpt_endpoint: k.gpt_endpoint||''
      })
    });
    const d = await res.json();
    if (d.error) { btn.textContent = '⟳ 全部翻译'; alert('翻译失败: '+d.error); return; }
    trEl.textContent = d.translation;
    trEl.classList.add('show');
    btn.textContent = '▲ 收起翻译';
  } catch(e) {
    btn.textContent = '⟳ 全部翻译';
  } finally {
    btn.disabled = false;
  }
}

function saveEdit(pid) {
  const el = document.getElementById('pt-'+pid); if (!el) return;
  prompts[pid].text = el.innerText.trim();
  fetch(`/api/prompt/${pid}`, {
    method:'PATCH', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID, text:prompts[pid].text})
  });
}

// ── chat input ref image ──────────────────────────────────────────────────────
let chatRefB64 = null;

function onChatImagePicked(input) {
  const f = input.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = e => {
    chatRefB64 = e.target.result.split(',')[1];
    document.getElementById('chatRefImg').src = e.target.result;
    document.getElementById('chatRefRow').style.display = 'flex';
  };
  r.readAsDataURL(f);
}
function removeChatRef() {
  chatRefB64 = null;
  document.getElementById('chatRefRow').style.display = 'none';
  document.getElementById('chatRefImg').src = '';
  document.getElementById('chatFileInput').value = '';
}

// ── global reference images (per prompt-group) ────────────────────────────────
const groupRefs = {};   // groupId -> [{b64, url}]
const cardRefs  = {};   // pid     -> [{b64, url}]

function addGlobalRef(gid, input) {
  if (!groupRefs[gid]) groupRefs[gid] = [];
  const slots = 4 - groupRefs[gid].length;
  [...input.files].slice(0, slots).forEach(f => {
    const r = new FileReader();
    r.onload = e => {
      groupRefs[gid].push({b64: e.target.result.split(',')[1], url: e.target.result});
      renderGlobalThumbs(gid);
    };
    r.readAsDataURL(f);
  });
  input.value = '';
}
function rmGlobalRef(gid, idx) {
  groupRefs[gid].splice(idx, 1); renderGlobalThumbs(gid);
}
function renderGlobalThumbs(gid) {
  const row = document.getElementById('grt-'+gid); if (!row) return;
  const imgs = groupRefs[gid] || [];
  row.innerHTML = imgs.map((img, i) => `
    <div class="ref-thumb"><img src="${img.url}" alt="">
      <span class="rm-ref" onclick="rmGlobalRef('${gid}',${i})">✕</span></div>`).join('');
  const btn = document.querySelector(`#grw-${gid} .global-ref-btn`);
  if (btn) btn.style.display = imgs.length >= 4 ? 'none' : '';
}

// ── card-level ref images ─────────────────────────────────────────────────────
function addCardRef(pid, input) {
  if (!cardRefs[pid]) cardRefs[pid] = [];
  const slots = 4 - cardRefs[pid].length;
  [...input.files].slice(0, slots).forEach(f => {
    const r = new FileReader();
    r.onload = e => {
      cardRefs[pid].push({b64: e.target.result.split(',')[1], url: e.target.result});
      renderCardRefThumbs(pid);
    };
    r.readAsDataURL(f);
  });
  input.value = '';
}
function rmCardRef(pid, idx) {
  cardRefs[pid].splice(idx, 1); renderCardRefThumbs(pid);
}
function clearCardRef(pid) {
  cardRefs[pid] = []; renderCardRefThumbs(pid);
}
function renderCardRefThumbs(pid) {
  const row  = document.getElementById('crt-'+pid); if (!row) return;
  const hint = document.getElementById('crh-'+pid);
  const imgs = cardRefs[pid] || [];
  row.innerHTML = imgs.map((img, i) => `
    <div class="ref-thumb"><img src="${img.url}" alt="">
      <span class="rm-ref" onclick="rmCardRef('${pid}',${i})">✕</span></div>`).join('');
  if (hint) hint.textContent = imgs.length
    ? `📎 已上传 ${imgs.length} 张（覆盖全局）`
    : '📎 单独上传参考图（覆盖全局）';
}

async function doGen(pid) {
  const k = K();
  if (!k.fk.some(x=>x)) { openConfig(); return; }
  const size    = document.getElementById('sz-'+pid)?.value || '1024x1024';
  const quality = document.getElementById('ql-'+pid)?.value || 'medium';
  const n       = parseInt(document.getElementById('nn-'+pid)?.value) || 1;
  // 卡片自己的参考图优先；没有则用全局批量参考图
  const gid     = prompts[pid]._groupId || '';
  const cardImg = cardRefs[pid] || [];
  const globalImg = groupRefs[gid] || [];
  const refImgs = (cardImg.length ? cardImg : globalImg).map(x => x.b64);
  prompts[pid].status = 'queued'; prompts[pid].images = [];
  const card = prompts[pid]._el; if (card) renderCard(card, pid);
  await fetch('/api/generate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID, pid, size, quality, n,
                          frank_keys:k.fk, ref_images:refImgs})
  });
  startPoll();
}

async function delCard(pid) {
  await fetch(`/api/prompt/${pid}`, {
    method:'DELETE', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID})
  });
  prompts[pid]?._el?.remove(); delete prompts[pid];
}

// ── polling ───────────────────────────────────────────────────────────────────
function startPoll() { if (!pollTimer) pollTimer = setInterval(poll, 2500); }
function stopPoll()  { clearInterval(pollTimer); pollTimer = null; }

async function poll() {
  const res  = await fetch(`/api/status?sid=${SID}`);
  const data = await res.json();
  let active = false;
  for (const [pid, st] of Object.entries(data)) {
    if (!prompts[pid]) continue;
    const prev = {status: prompts[pid].status, len: (prompts[pid].images||[]).length};
    prompts[pid].status = st.status; prompts[pid].images = st.images; prompts[pid].error = st.error;
    if (st.status==='queued'||st.status==='generating') active = true;
    const card = prompts[pid]._el;
    if (card && (st.status!==prev.status || (st.images||[]).length!==prev.len))
      renderCard(card, pid);
  }
  if (!active) {
    stopPoll();
    if (sideTabCur==='img') loadSideImages(true);
  }
}

// ── send ──────────────────────────────────────────────────────────────────────
// ── send mode (chat / gen) ────────────────────────────────────────────────────
let _sendMode = 'chat';
function setSendMode(mode) {
  _sendMode = mode;
  const pill  = document.getElementById('modePill');
  const label = document.getElementById('modeLabel');
  const btn   = document.getElementById('sendBtn');
  const iChat = document.getElementById('smi-chat');
  const iGen  = document.getElementById('smi-gen');
  const sendSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;
  if (mode === 'chat') {
    pill.innerHTML = '💬 <span id="modeLabel">对话</span>';
    pill.classList.remove('gen-mode');
    btn.innerHTML  = sendSvg;
    iChat.classList.add('active'); iGen.classList.remove('active');
  } else {
    pill.innerHTML = '🎨 <span id="modeLabel">直接生图</span>';
    pill.classList.add('gen-mode');
    btn.innerHTML  = '🎨';
    iGen.classList.add('active'); iChat.classList.remove('active');
  }
  document.getElementById('sendMenu').classList.remove('open');
}
function toggleSendMenu(e) {
  e.stopPropagation();
  const menu = document.getElementById('sendMenu');
  if (menu.classList.contains('open')) { menu.classList.remove('open'); return; }
  const r = e.currentTarget.getBoundingClientRect();
  menu.style.right  = (window.innerWidth - r.right) + 'px';
  menu.style.bottom = (window.innerHeight - r.top + 6) + 'px';
  menu.classList.add('open');
}
function sendOrQuick() {
  if (_sendMode === 'gen') quickGen(); else sendMsg();
}
document.addEventListener('click', () => {
  document.getElementById('sendMenu')?.classList.remove('open');
});

function handleKey(e) {
  if (e.key==='Enter'&&!e.shiftKey) { e.preventDefault(); sendOrQuick(); }
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

async function sendMsg() {
  const k = K();
  if (!hasKeys()) { openConfig(); return; }
  const inp  = document.getElementById('msgInput');
  const text = inp.value.trim(); if (!text) return;
  const imgUrl = chatRefB64 ? document.getElementById('chatRefImg').src : null;
  inp.value = ''; inp.style.height = 'auto';
  const sendBtn  = document.getElementById('sendBtn');
  const arrowBtn = document.querySelector('.send-btn-arrow');
  sendBtn.disabled = true;
  if (arrowBtn) arrowBtn.disabled = true;
  appendUserMsg(text, imgUrl);
  const b64 = chatRefB64; removeChatRef(); appendThinking();

  try {
    const res = await fetch('/api/chat/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sid:SID, message:text, gpt_key:k.gpt,
              gpt_model:k.gpt_model||'gpt-5.4',
              gpt_endpoint:k.gpt_endpoint||'', image_b64:b64})
    });
    if (!res.ok) {
      const err = await res.json().catch(()=>({}));
      removeThinking(); appendSys('❌ '+(err.error||'HTTP '+res.status)); return;
    }
    removeThinking();

    // Create streaming bubble
    const conv  = document.getElementById('conv');
    const cid   = 'pc' + (++msgCnt);
    const g     = document.createElement('div');
    g.className = 'msg-group msg-ai-row';
    g.innerHTML = `<div class="ai-avatar">✦</div>
      <div class="ai-body" id="aibody-${cid}">
        <div class="msg-ai-text" id="stream-${cid}"></div>
      </div>`;
    conv.appendChild(g);
    hideWelcome();
    const streamEl = document.getElementById('stream-'+cid);
    const bodyEl   = document.getElementById('aibody-'+cid);

    // Read SSE stream
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', rawText = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';
      for (const block of blocks) {
        if (!block.trim()) continue;
        let evtType = '', dataStr = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) evtType  = line.slice(6).trim();
          if (line.startsWith('data:'))  dataStr  = line.slice(5).trim();
        }
        if (!dataStr) continue;
        try {
          const payload = JSON.parse(dataStr);
          if (evtType === 'error') {
            streamEl.textContent = '❌ '+(payload.error||'未知错误');
          } else if (evtType === 'done') {
            // Final render: markdown + prompt cards
            streamEl.innerHTML = md(payload.display || '');
            if (payload.new_prompts?.length) {
              const grw = document.createElement('div');
              grw.className = 'global-ref-wrap'; grw.id = 'grw-'+cid;
              grw.innerHTML = `<div class="global-ref-bar">
                <span class="global-ref-label">🖼 批量参考图（选填，最多4张，应用到所有卡片）</span>
                <label class="global-ref-btn">+ 上传图片
                  <input type="file" accept="image/*" multiple style="display:none"
                    onchange="addGlobalRef('${cid}',this)">
                </label></div>
                <div class="global-ref-thumbs" id="grt-${cid}"></div>`;
              bodyEl.appendChild(grw);
              const cardsEl = document.createElement('div');
              cardsEl.className = 'prompt-cards'; cardsEl.id = cid;
              bodyEl.appendChild(cardsEl);
              payload.new_prompts.forEach((p, i) => {
                prompts[p.pid] = {title:p.title, desc:p.desc||'', text:p.text,
                  status:'pending', images:[], error:'', _groupId:cid};
                const card = buildCard(p.pid, i+1);
                cardsEl.appendChild(card);
                prompts[p.pid]._el = card;
              });
            }
            loadSideConvs(); scrollDown();
          } else if (payload.text) {
            rawText += payload.text;
            // Show plain text while streaming, hide [PROMPT] blocks
            streamEl.textContent = rawText.replace(/\[PROMPT\][\s\S]*?(\[\/PROMPT\]|$)/g,'').trim();
            scrollDown();
          }
        } catch(e) { /* ignore malformed */ }
      }
    }
  } catch(e) {
    removeThinking(); appendSys('❌ 请求失败: '+e.message);
  } finally {
    sendBtn.disabled = false;
    if (arrowBtn) arrowBtn.disabled = false;
  }
}

async function quickGen() {
  const k = K();
  if (!k.fk.some(x=>x)) { openConfig(); return; }
  const inp  = document.getElementById('msgInput');
  const text = inp.value.trim(); if (!text) return;
  inp.value = ''; inp.style.height = 'auto';
  hideWelcome();

  const res = await fetch('/api/prompt/create', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID, text})
  });
  const d = await res.json();
  if (d.error) { appendSys('❌ '+d.error); return; }

  const conv = document.getElementById('conv');
  const cid  = 'pc'+(++msgCnt);
  const g    = document.createElement('div');
  g.className = 'msg-group msg-ai-row';
  g.innerHTML = `<div class="ai-avatar">🎨</div>
    <div class="ai-body">
      <div class="global-ref-wrap" id="grw-${cid}">
        <div class="global-ref-bar">
          <span class="global-ref-label">🖼 批量参考图（选填，最多4张，应用到所有卡片）</span>
          <label class="global-ref-btn">+ 上传图片
            <input type="file" accept="image/*" multiple style="display:none"
              onchange="addGlobalRef('${cid}',this)">
          </label>
        </div>
        <div class="global-ref-thumbs" id="grt-${cid}"></div>
      </div>
      <div class="prompt-cards" id="${cid}"></div>
    </div>`;
  conv.appendChild(g);

  const pid = d.pid;
  prompts[pid] = {title:d.title, desc:'', text:d.text,
    status:'pending', images:[], error:'', _groupId:cid};
  const card = buildCard(pid, 1);
  document.getElementById(cid).appendChild(card);
  prompts[pid]._el = card;
  scrollDown();
}

function fillInput(t) {
  const inp = document.getElementById('msgInput');
  inp.value = t; autoResize(inp); inp.focus();
}


async function clearConv() {
  await fetch('/api/chat/clear', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID})
  });
  prompts = {}; activeCid = null; stopPoll();
  document.getElementById('topbarTitle').textContent = 'AI 图像创作助手';
  document.getElementById('conv').innerHTML = `
    <div class="welcome" id="welcome">
      <h2>AI 图像创作助手</h2>
      <p>与 ChatGPT 沟通你的创意，由 GPT-Image-2 高质量生成<br>
         首次使用请点击左下角 🔑 填写 API Key</p>
      <div class="gallery-chips" id="welcomeChips2"></div>
      </div>
    </div>`;
  renderWelcomeChips('welcomeChips2');
  document.querySelectorAll('.side-conv-item').forEach(el=>el.classList.remove('active'));
}

let _lbPrompt = '';
function openLb(iid, prompt, showUse) {
  // iid can be a direct URL (starts with / or http) or an image id
  const isDirect = iid.startsWith('/') || iid.startsWith('http');
  document.getElementById('lbImg').src  = isDirect ? iid : `/api/image/${SID}/${iid}`;
  document.getElementById('lbDl').href  = isDirect ? iid : `/api/image/${SID}/${iid}/download`;
  const panel   = document.getElementById('lbPromptPanel');
  const txt     = document.getElementById('lbPromptText');
  const useBtn  = document.getElementById('lbUseBtn');
  _lbPrompt = prompt || '';
  if (_lbPrompt) {
    txt.textContent = _lbPrompt;
    panel.classList.add('visible');
    useBtn.style.display = (showUse !== false) ? 'block' : 'none';
  } else {
    panel.classList.remove('visible');
    useBtn.style.display = 'none';
  }
  document.getElementById('lb').classList.add('open');
}
function useLbPrompt() { fillInput(_lbPrompt); closeLb(); }
function closeLb() { document.getElementById('lb').classList.remove('open'); }

// ── sidebar ────────────────────────────────────────────────────────────────────
function switchSideTab(name) {
  document.getElementById('stab-conv').classList.toggle('active', name==='conv');
  document.getElementById('stab-img').classList.toggle('active', name==='img');
  document.getElementById('sbody-conv').style.display = name==='conv' ? 'block':'none';
  document.getElementById('sbody-img').style.display  = name==='img'  ? 'block':'none';
  sideTabCur = name;
  if (name==='conv') loadSideConvs();
  if (name==='img')  loadSideImages(true);
}

async function loadSideConvs() {
  const res  = await fetch(`/api/history/conversations?sid=${SID}`);
  const data = await res.json();
  const list  = document.getElementById('sideConvList');
  const empty = document.getElementById('sideConvEmpty');
  list.innerHTML = '';
  empty.style.display = data.length ? 'none' : 'block';
  let lastDate = '';
  data.forEach(conv => {
    const d = new Date(conv.updated * 1000);
    const dateStr = d.toLocaleDateString('zh-CN',{month:'long',day:'numeric'});
    if (dateStr !== lastDate) {
      const lbl = document.createElement('div');
      lbl.className = 'side-section-label';
      lbl.textContent = dateStr;
      list.appendChild(lbl); lastDate = dateStr;
    }
    const div = document.createElement('div');
    div.className = 'side-conv-item' + (conv.cid===activeCid ? ' active':'');
    div.dataset.cid = conv.cid;
    div.innerHTML = `<div class="side-conv-preview">${esc(conv.preview||'对话')}</div>`;
    div.onclick = () => restoreConv(conv.cid, conv.preview);
    list.appendChild(div);
  });
}

async function loadSideImages(reset=false) {
  if (reset) { sideImgPage=0; document.getElementById('sideImgGrid').innerHTML=''; }
  const res  = await fetch(`/api/history/images?sid=${SID}&page=${sideImgPage}`);
  const data = await res.json();
  const grid  = document.getElementById('sideImgGrid');
  const empty = document.getElementById('sideImgEmpty');
  const more  = document.getElementById('sideImgMore');
  empty.style.display = (grid.children.length===0 && data.length===0) ? 'block':'none';
  more.style.display  = data.length===24 ? 'block':'none';
  data.forEach(img => {
    const div = document.createElement('div');
    div.className = 'side-img-item';
    div.innerHTML = `<img src="/api/image/${SID}/${img.iid}" loading="lazy" alt="${esc(img.title||'')}">`;
    div.onclick = () => { openLb(img.iid, img.prompt || ''); };
    grid.appendChild(div);
  });
  sideImgPage++;
}
function loadMoreSideImages() { loadSideImages(false); }

async function restoreConv(cid, preview) {
  const res = await fetch('/api/session/restore', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:SID, cid})
  });
  const data = await res.json();
  if (data.error) { appendSys('❌ 恢复失败: '+data.error); return; }

  prompts = {}; stopPoll(); activeCid = cid;
  document.getElementById('conv').innerHTML = '';
  document.getElementById('topbarTitle').textContent = preview || '历史对话';
  document.querySelectorAll('.side-conv-item').forEach(el=>{
    el.classList.toggle('active', el.dataset.cid===cid);
  });

  const msgs = data.messages;
  const lastAiIdx = msgs.reduceRight((acc,m,i)=> acc===-1&&m.role==='assistant'?i:acc, -1);
  for (let i=0; i<msgs.length; i++) {
    const m = msgs[i];
    if (m.role==='user') appendUserMsg(m.content);
    else if (m.role==='assistant') {
      const display = m.content.replace(/\[PROMPT\][\s\S]*?\[\/PROMPT\]/g,'').trim();
      appendAiMsg(display, i===lastAiIdx ? data.prompts : []);
    }
  }
  appendSys('📂 已加载历史对话，可继续聊天');
  scrollDown();
}

// ── welcome gallery chips ─────────────────────────────────────────────────────
const GALLERY_CHIPS = [
  { img: '/gallery/mechanical-bouquet.png', label: '黄铜机械花束',
    prompt: 'Intent: product concept art. Background: warm neutral studio gradient. Foreground: soft shadow on a matte surface. Hero subject: bouquet of mechanical flowers made of brass gears and enamel petals. Finishing details: high-detail 3d render, crisp metal reflections, no text, no logos, no watermark. Camera: 70mm, eye-level, centered framing.' },
  { img: '/gallery/lavender-sunrise.png', label: '薰衣草日出风光',
    prompt: 'Intent: landscape photography print. Background: rolling lavender fields under a pastel sunrise sky. Foreground: dew-covered lavender blossoms in soft focus. Hero subject: the sun just above the horizon with gentle light rays. Finishing details: photorealistic, natural color grading, no logos or trademarks, no watermark. Camera: 35mm, eye-level, wide framing.' },
  { img: '/gallery/coral-reef.png', label: '珊瑚礁海龟',
    prompt: 'Intent: nature poster. Background: clear turquoise water fading to deeper blue. Foreground: vibrant coral clusters with small fish. Hero subject: sea turtle swimming across the center. Finishing details: underwater light rays, natural color grading, no logos or trademarks, no watermark. Camera: wide-angle, underwater eye-level.' },
  { img: '/gallery/kitchen-service-rush.png', label: '厨房纪实摄影',
    prompt: 'Use case: photorealistic-natural. Create a candid professional photo inside a restaurant kitchen during dinner service. A chef slides a pan across the pass, a line cook reaches for herbs, steam rises from pasta water, and a server waits with two plates. The scene should have natural flow: crossed arms, moving hands, steam, stacked plates, towels, and narrow walking space. Camera: 35mm, eye-level, available light, slight motion blur in hands. No staged smiles, no logos, no watermark.' }
];

function renderWelcomeChips(elId) {
  const el = document.getElementById(elId); if (!el) return;
  el.innerHTML = GALLERY_CHIPS.map((c, i) => `
    <div class="gallery-chip" data-idx="${i}" onclick="openGalleryChip(this.dataset.idx)">
      <img src="${c.img}" alt="" loading="lazy">
      <div class="gallery-chip-label">${c.label}</div>
    </div>`).join('');
}
function openGalleryChip(idx) {
  const c = GALLERY_CHIPS[+idx];
  if (c) openLb(c.img, c.prompt);
}

// ── auto-load sidebar on startup ──────────────────────────────────────────────
renderWelcomeChips('welcomeChips');
loadSideConvs();
</script>
</body>
</html>"""

# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"  AI Image Studio → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
