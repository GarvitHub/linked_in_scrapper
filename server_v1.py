"""
InfluenceIQ — Local Backend Server  v4
Cars24 Influencer Management System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL ONCE (run in CMD / Terminal):
    pip install flask flask-cors requests openai-whisper

FFMPEG (required for Whisper audio processing):
    Windows → https://www.gyan.dev/ffmpeg/builds/
              Download  ffmpeg-release-essentials.zip
              Extract → Add the /bin folder to Windows PATH
              Restart CMD after adding to PATH
    Mac     → brew install ffmpeg
    Linux   → sudo apt install ffmpeg

RUN:
    python server.py

Then open InfluenceIQ_v4.html in your browser.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, urllib3, os, re, tempfile, threading, json
import smtplib, base64
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

# Allow large uploads up to 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# ══════════════════════════════════════════════════════════
# Whisper model cache — loaded once per session, reused
# ══════════════════════════════════════════════════════════
_whisper_model      = None
_whisper_model_size = None
_whisper_lock       = threading.Lock()

def get_whisper_model(size='base'):
    global _whisper_model, _whisper_model_size
    with _whisper_lock:
        if _whisper_model is None or _whisper_model_size != size:
            print(f"\n  Loading Whisper '{size}' model (first run may download ~150 MB)...")
            try:
                import whisper
                _whisper_model      = whisper.load_model(size)
                _whisper_model_size = size
                print(f"  ✓ Whisper '{size}' ready")
            except ImportError:
                raise RuntimeError(
                    "Whisper not installed.\n"
                    "Fix: pip install openai-whisper"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")
        return _whisper_model


def score_keywords(text, keywords):
    """Score how many keywords/talking-points appear in the transcript."""
    tl      = text.lower()
    krs     = []
    matched = 0

    for kw in keywords:
        k = kw.strip()
        if not k:
            continue
        kl = k.lower()

        # 1. Exact match
        cnt   = len(re.findall(re.escape(kl), tl))
        found = cnt > 0

        # 2. Flexible spacing / hyphen
        if not found:
            try:
                p     = re.escape(kl).replace(r'\ ', r'[\s\-]+')
                found = bool(re.search(p, tl))
                cnt   = len(re.findall(p, tl)) if found else 0
            except re.error:
                pass

        # 3. All individual words present (multi-word phrases)
        if not found:
            words = [w for w in kl.split() if len(w) > 2]
            if len(words) > 1 and all(w in tl for w in words):
                found = True
                cnt   = 1

        krs.append({'keyword': k, 'found': found, 'count': cnt})
        if found:
            matched += 1

    total = len(krs)
    score = round((matched / total) * 100) if total > 0 else 100
    return krs, matched, total, score


def run_whisper(filepath, model_sz, keywords):
    """Core Whisper transcription — shared by both upload and URL endpoints."""
    model  = get_whisper_model(model_sz)

    print("  Transcribing...")
    result = model.transcribe(
        filepath,
        fp16=False,         # CPU-safe (no GPU required)
        verbose=False,
        task='transcribe',
    )

    text = (result.get('text') or '').strip()
    lang = result.get('language', 'unknown')
    segs = result.get('segments') or []
    dur  = int(segs[-1].get('end', 0)) if segs else 0

    print(f"  ✓ {len(text)} chars | lang={lang} | {dur}s")

    if not text:
        raise ValueError(
            "Whisper returned an empty transcript.\n"
            "Possible causes:\n"
            "  • Video has no spoken audio\n"
            "  • Audio is too quiet or is background noise only\n"
            "  • Video file may be corrupt\n"
            "Try the 'tiny' model or a different clip."
        )

    krs, matched, total, score = score_keywords(text, keywords)
    print(f"  Keywords: {matched}/{total} = {score}%")

    return {
        'success':         True,
        'transcript':      text,
        'language':        lang,
        'duration_sec':    dur,
        'keyword_results': krs,
        'matched':         matched,
        'total_keywords':  total,
        'score':           score,
    }


def friendly_error(e):
    """Convert technical Python exceptions to user-friendly messages."""
    msg = str(e)
    if 'ffmpeg' in msg.lower() or 'audioread' in msg.lower() or 'No such file' in msg.lower():
        return (
            "ffmpeg not found or not in PATH.\n"
            "Fix:\n"
            "1. Download from https://www.gyan.dev/ffmpeg/builds/\n"
            "2. Extract the ZIP\n"
            "3. Add the /bin folder to Windows PATH\n"
            "4. Restart CMD and run server.py again"
        )
    if 'No module named' in msg and 'whisper' in msg.lower():
        return "Whisper not installed. Run:  pip install openai-whisper"
    if 'CUDA' in msg or 'cuda' in msg:
        return "GPU/CUDA error — server.py runs fine on CPU. Restart and try again."
    if 'Permission' in msg or 'Access is denied' in msg:
        return "File permission error — try running CMD as Administrator"
    if '403' in msg:
        return "Access denied (403) — set Google Drive sharing to 'Anyone with the link'"
    if '404' in msg:
        return "File not found (404) — check the link is correct and the file exists"
    return msg


# ══════════════════════════════════════════════════════════
# GET /ping — health check
# ══════════════════════════════════════════════════════════
@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'server': 'InfluenceIQ v4'})


# ══════════════════════════════════════════════════════════
# GET /whisper/check — verify Whisper + ffmpeg installation
# ══════════════════════════════════════════════════════════
@app.route('/whisper/check')
def whisper_check():
    result = {'whisper': False, 'ffmpeg': False, 'installed': False, 'ready': False}

    # Check Whisper
    try:
        import whisper
        result['whisper']   = True
        result['installed'] = True
        result['version']   = getattr(whisper, '__version__', 'installed')
    except ImportError:
        result['whisper_error'] = 'Run: pip install openai-whisper'

    # Check ffmpeg
    import subprocess
    try:
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        result['ffmpeg'] = (r.returncode == 0)
        if not result['ffmpeg']:
            result['ffmpeg_error'] = 'ffmpeg returned a non-zero exit code'
    except FileNotFoundError:
        result['ffmpeg_error'] = 'ffmpeg not found in PATH — see setup instructions'
    except Exception as ex:
        result['ffmpeg_error'] = str(ex)

    result['ready'] = result['whisper'] and result['ffmpeg']
    return jsonify(result)


# ══════════════════════════════════════════════════════════
# POST /upload-and-transcribe — Whisper via direct file upload
#   multipart/form-data fields:
#     video      — video or audio file (mp4, mov, mkv, webm, mp3, wav…)
#     keywords   — JSON array string, e.g. '["cars24","test drive"]'
#     model      — tiny | base | small | medium | large  (default: base)
# ══════════════════════════════════════════════════════════
@app.route('/upload-and-transcribe', methods=['POST'])
def upload_and_transcribe():
    print("\n  ── /upload-and-transcribe ──")

    video_file = request.files.get('video')
    if not video_file or not video_file.filename:
        return jsonify({'error': 'No video file received. Select a file and try again.'}), 400

    # Parse keywords
    try:
        keywords = json.loads(request.form.get('keywords', '[]'))
    except Exception:
        keywords = []

    model_sz = request.form.get('model', 'base')
    if model_sz not in ('tiny', 'base', 'small', 'medium', 'large'):
        model_sz = 'base'

    filename = video_file.filename or 'upload.mp4'
    print(f"  File:     {filename}")
    print(f"  Model:    {model_sz}")
    print(f"  Keywords: {len(keywords)}")

    # Detect file extension
    suffix = '.mp4'
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mp3', '.wav', '.m4a', '.ogg', '.flac']:
        if filename.lower().endswith(ext):
            suffix = ext
            break

    tmp_path = None
    try:
        tmp      = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()

        video_file.save(tmp_path)

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"  Saved:    {size_mb:.1f} MB → {tmp_path}")

        if size_mb < 0.001:
            raise ValueError("Uploaded file is empty — please try again.")
        if size_mb > 500:
            raise ValueError(f"File too large ({size_mb:.0f} MB). Maximum is 500 MB.")

        data = run_whisper(tmp_path, model_sz, keywords)
        data['filename'] = filename
        return jsonify(data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': friendly_error(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print("  Temp file cleaned up")
            except Exception:
                pass  # Windows may briefly lock the file


# ══════════════════════════════════════════════════════════
# POST /transcribe — Whisper via URL (Google Drive / direct link)
#   JSON body:
#     url        — video URL (Google Drive share link or direct URL)
#     keywords   — array of talking points
#     model      — whisper model size (default: base)
# ══════════════════════════════════════════════════════════
@app.route('/transcribe', methods=['POST'])
def transcribe():
    body     = request.get_json() or {}
    url      = (body.get('url') or '').strip()
    keywords = body.get('keywords') or []
    model_sz = body.get('model', 'base')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if model_sz not in ('tiny', 'base', 'small', 'medium', 'large'):
        model_sz = 'base'

    print(f"\n  ── /transcribe (URL) ──")
    print(f"  URL:   {url[:80]}")
    print(f"  Model: {model_sz}")

    # Detect extension from URL
    suffix = '.mp4'
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mp3', '.wav', '.m4a']:
        if ext in url.lower():
            suffix = ext
            break

    tmp_path = None
    try:
        direct   = convert_gdrive_url(url)
        tmp      = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()

        print("  Downloading...")
        size_mb = download_file(direct, tmp_path)

        file_size = os.path.getsize(tmp_path)
        if file_size < 10 * 1024:
            raise ValueError(
                f"Downloaded file is too small ({file_size} bytes). "
                "Make sure Google Drive sharing is set to 'Anyone with the link'."
            )

        data = run_whisper(tmp_path, model_sz, keywords)
        return jsonify(data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': friendly_error(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def convert_gdrive_url(url):
    """Convert a Google Drive share link to a direct download URL."""
    for pat in [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)',
        r'open\?id=([a-zA-Z0-9_-]+)',
    ]:
        m = re.search(pat, url)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}&confirm=t"
    if 'dropbox.com' in url:
        url = re.sub(r'[?&]dl=0', '', url)
        return url + ('&' if '?' in url else '?') + 'dl=1'
    return url


def download_file(url, dest_path):
    """Download a file from a URL to dest_path, handling Google Drive confirmation."""
    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp    = session.get(url, stream=True, timeout=180, verify=False, headers=headers)

    # Handle Google Drive large-file confirmation cookie
    for key, value in resp.cookies.items():
        if key.startswith('download_warning'):
            resp = session.get(
                url + ('&' if '?' in url else '?') + f'confirm={value}',
                stream=True, timeout=180, verify=False, headers=headers
            )
            break

    # Handle confirmation token in HTML response
    ct = resp.headers.get('Content-Type', '')
    if 'text/html' in ct and 'drive.google.com' in url:
        m = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
        if m:
            resp = session.get(
                url + ('&' if '?' in url else '?') + f'confirm={m.group(1)}',
                stream=True, timeout=180, verify=False, headers=headers
            )

    if resp.status_code == 403:
        raise ValueError("Access denied (403) — set Google Drive sharing to 'Anyone with the link'.")
    if resp.status_code == 404:
        raise ValueError("File not found (404) — check the link.")
    resp.raise_for_status()

    total = 0
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB")
    return size_mb


# ══════════════════════════════════════════════════════════
# POST /send-mail — Gmail SMTP
#   JSON body:
#     to            — recipient email
#     subject       — email subject
#     body          — plain text body
#     from_email    — your Gmail address
#     from_name     — display name (default: Cars24 Influencer Team)
#     password      — Gmail App Password (16-char, NOT your normal password)
#     cc            — list of CC email strings (optional)
#     attachment_data — base64 encoded file (optional)
#     attachment_name — filename for the attachment (optional)
#     attachment_mime — MIME type e.g. application/pdf (optional)
# ══════════════════════════════════════════════════════════
@app.route('/send-mail', methods=['POST'])
def send_mail():
    print("\n  ── /send-mail ──")

    if request.is_json:
        d          = request.get_json() or {}
        g          = lambda k, default='': (d.get(k) or default)
        to         = g('to').strip()
        subject    = g('subject').strip()
        body       = g('body').strip()
        from_email = g('from_email').strip()
        from_name  = g('from_name', 'Cars24 Influencer Team').strip()
        password   = g('password').strip()
        cc_raw     = d.get('cc', [])
        cc         = ','.join(cc_raw) if isinstance(cc_raw, list) else str(cc_raw or '').strip()
        att_b64    = g('attachment_data')
        att_name   = g('attachment_name', 'attachment')
        att_mime   = g('attachment_mime', 'application/octet-stream')
    else:
        g          = lambda k, default='': (request.form.get(k) or default)
        to         = g('to').strip()
        subject    = g('subject').strip()
        body       = g('body').strip()
        from_email = g('from_email').strip()
        from_name  = g('from_name', 'Cars24 Influencer Team').strip()
        password   = g('password').strip()
        cc         = g('cc').strip()
        att_b64    = g('attachment_data')
        att_name   = g('attachment_name', 'attachment')
        att_mime   = g('attachment_mime', 'application/octet-stream')

    print(f"  To:      {to}")
    print(f"  Subject: {subject[:60]}")
    print(f"  From:    {from_email}")

    # Validate required fields
    missing = [f for f, v in [
        ('to', to), ('subject', subject), ('body', body),
        ('from_email', from_email), ('password', password)
    ] if not v]
    if missing:
        return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing)}'}), 400
    if '@' not in to:
        return jsonify({'success': False, 'error': f'Invalid recipient email: {to}'}), 400
    if '@' not in from_email:
        return jsonify({'success': False, 'error': f'Invalid sender email: {from_email}'}), 400

    try:
        msg            = MIMEMultipart()
        msg['From']    = f"{from_name} <{from_email}>"
        msg['To']      = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # File attachment (multipart form upload)
        file = request.files.get('attachment')
        if file and file.filename:
            data = file.read()
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(data)
            encoders.encode_base64(part)
            safe_name = re.sub(r'[^\w\s\-.]', '_', file.filename)
            part.add_header('Content-Disposition', f'attachment; filename="{safe_name}"')
            msg.attach(part)
            print(f"  File attachment: {file.filename} ({len(data)/1024:.0f} KB)")

        # Base64 attachment (from Settings mail template attachments)
        if att_b64:
            try:
                raw       = base64.b64decode(att_b64.split(',', 1)[-1] if ',' in att_b64 else att_b64)
                mt, ms    = (att_mime + '/octet-stream').split('/', 2)[:2]
                p2        = MIMEBase(mt, ms)
                p2.set_payload(raw)
                encoders.encode_base64(p2)
                safe_att  = re.sub(r'[^\w\s\-.]', '_', att_name)
                p2.add_header('Content-Disposition', f'attachment; filename="{safe_att}"')
                msg.attach(p2)
                print(f"  B64 attachment: {att_name} ({len(raw)/1024:.0f} KB)")
            except Exception as ex:
                print(f"  Attachment warning (skipped): {ex}")

        # CC recipients
        cc_list    = [e.strip() for e in cc.split(',') if e.strip()] if cc else []
        if cc_list:
            msg['Cc'] = ', '.join(cc_list)
        recipients = [to] + cc_list

        print(f"  Connecting to smtp.gmail.com:587 ...")
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as srv:
            srv.ehlo()
            srv.starttls()
            srv.ehlo()
            srv.login(from_email, password)
            srv.sendmail(from_email, recipients, msg.as_string())

        print(f"  ✓ Mail sent to {to}" + (f" + {len(cc_list)} CC" if cc_list else ""))
        return jsonify({'success': True})

    except smtplib.SMTPAuthenticationError:
        msg = (
            'Gmail authentication failed. '
            'You must use a Gmail App Password — not your regular Gmail password. '
            'Steps: Google Account → Security → 2-Step Verification → App passwords → Create new → copy the 16-char password.'
        )
        print("  ✗ Gmail auth error")
        return jsonify({'success': False, 'error': msg}), 401

    except smtplib.SMTPRecipientsRefused as e:
        print(f"  ✗ Recipient refused: {e}")
        return jsonify({'success': False, 'error': f'Recipient address rejected by Gmail: {to}'}), 400

    except smtplib.SMTPException as e:
        print(f"  ✗ SMTP error: {e}")
        return jsonify({'success': False, 'error': f'SMTP error: {e}'}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════
# GET /search — SerpAPI web search (Talent Discovery)
#   query params: q, api_key, num
# ══════════════════════════════════════════════════════════
@app.route('/search')
def search():
    q       = request.args.get('q', '')
    api_key = request.args.get('api_key', '')
    num     = request.args.get('num', '10')

    if not q or not api_key:
        return jsonify({'error': 'Missing q or api_key parameters'}), 400
    try:
        resp = requests.get(
            'https://serpapi.com/search.json',
            params={'q': q, 'api_key': api_key, 'num': num, 'engine': 'google'},
            timeout=30, verify=False
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════
# Apify proxy endpoints — avoids CORS issues from browser
# ══════════════════════════════════════════════════════════

@app.route('/apify/start', methods=['POST'])
def apify_start():
    """Start an Apify actor run."""
    actor = request.args.get('actor', '')
    token = request.args.get('token', '')
    if not actor or not token:
        return jsonify({'error': 'Missing actor or token'}), 400
    try:
        payload = request.get_json() or {}
        print(f"\n  ── /apify/start ──")
        print(f"  Actor:   {actor}")
        print(f"  Input:   {json.dumps(payload)[:120]}")
        r = requests.post(
            f'https://api.apify.com/v2/acts/{actor}/runs?token={token}',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60, verify=False
        )
        data = r.json()
        if r.status_code == 200 or r.status_code == 201:
            run_id = data.get('data', {}).get('id', '?')
            print(f"  ✓ Run started: {run_id}")
        else:
            print(f"  ✗ Apify error {r.status_code}: {str(data)[:200]}")
        return jsonify(data), r.status_code
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/apify/status')
def apify_status():
    """Poll the status of an Apify run."""
    run_id = request.args.get('run_id', '')
    token  = request.args.get('token', '')
    if not run_id or not token:
        return jsonify({'error': 'Missing run_id or token'}), 400
    try:
        r  = requests.get(
            f'https://api.apify.com/v2/actor-runs/{run_id}?token={token}',
            timeout=30, verify=False
        )
        data   = r.json()
        status = data.get('data', {}).get('status', '?')
        print(f"  Apify poll: {run_id[:12]}… → {status}")
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/apify/dataset')
def apify_dataset():
    """Fetch the result dataset from a completed Apify run."""
    ds_id = request.args.get('ds_id', '')
    token = request.args.get('token', '')
    if not ds_id or not token:
        return jsonify({'error': 'Missing ds_id or token'}), 400
    try:
        r = requests.get(
            f'https://api.apify.com/v2/datasets/{ds_id}/items?token={token}&format=json&limit=5000',
            timeout=60, verify=False
        )
        data = r.json()
        count = len(data) if isinstance(data, list) else '?'
        print(f"  ✓ Dataset fetched: {count} items")
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('\n' + '='*58)
    print('  InfluenceIQ Backend  v4  —  Cars24')
    print('='*58)
    print('  Endpoints ready:')
    print('    GET  /ping                     → health check')
    print('    GET  /whisper/check            → Whisper + ffmpeg status')
    print('    POST /upload-and-transcribe    → Whisper (file upload)')
    print('    POST /transcribe               → Whisper (Google Drive URL)')
    print('    POST /send-mail                → Gmail SMTP')
    print('    GET  /search                   → SerpAPI')
    print('    POST /apify/start              → start Apify actor run')
    print('    GET  /apify/status             → poll run status')
    print('    GET  /apify/dataset            → fetch results')
    print('='*58)
    print('\n  Open InfluenceIQ_v4.html in your browser')
    print('  Press Ctrl+C to stop\n')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
