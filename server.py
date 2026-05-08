# server.py (إصدار محسن ضد الأخطاء)
import asyncio
import json
import cv2
import random
import os
import secrets
import traceback
from datetime import datetime
from aiohttp import web
from ultralytics import YOLO
from geopy.geocoders import Nominatim

PORT = int(os.environ.get("PORT", 5000))
UPLOAD_DIR = "/tmp/geo_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_CLASSES = "person, car, weapon, explosive"
CLASS_EMOJI_MAP = {
    "person": "👤",
    "car": "🚗",
    "motorcycle": "🏍",
    "bus": "🚌",
    "truck": "🚚",
    "weapon": "🔫",
    "explosive": "💣"
}

print("تحميل YOLO-World...")
model = YOLO('yolov8s-world.pt')

sessions = {}

def safe_filename(name):
    # يحذف المسافات والنقاط الزائدة ويضيف طابع زمني
    base, ext = os.path.splitext(name)
    safe_name = f"{secrets.token_hex(8)}_{int(datetime.now().timestamp())}{ext}"
    return safe_name

def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="geo_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    # محاولة ثانية مع كتابة صحيحة لإنماء (اليمن) إذا كان الاسم يشبهها
    if "انماء" in city_name or "إنماء" in city_name:
        return 12.8000, 45.0333
    # افتراضي آخر: صنعاء
    return 15.3694, 44.1910

def simulate_movement(base_lat, base_lon, last_pos=None):
    if last_pos:
        lat, lon = last_pos
        lat += random.uniform(-0.0001, 0.0001)
        lon += random.uniform(-0.0001, 0.0001)
        return lat, lon
    return base_lat + random.uniform(-0.0005, 0.0005), base_lon + random.uniform(-0.0005, 0.0005)

# ========== الصفحة الرئيسية ==========
async def index(request):
    return web.Response(text=INDEX_HTML, content_type='text/html')

INDEX_HTML = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <title>نظام الاستخبارات الجغرافية</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e; color: #eee;
            display: flex; justify-content: center; align-items: center;
            height: 100vh; margin: 0; direction: rtl;
        }
        .container {
            background: #16213e; padding: 30px; border-radius: 15px;
            max-width: 500px; width: 90%;
        }
        h2 { color: #e94560; text-align: center; }
        label { font-weight: bold; display: block; margin: 10px 0 5px; }
        input[type="file"], input[type="text"] {
            width: 100%; padding: 10px; margin-bottom: 10px;
            border: 1px solid #0f3460; border-radius: 5px;
            background: #1a1a2e; color: #eee;
        }
        button {
            width: 100%; padding: 15px; background: #e94560;
            color: white; border: none; border-radius: 5px;
            font-size: 18px; cursor: pointer; margin-top: 20px;
        }
        .note { font-size: 12px; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <h2>رفع فيديو أو صورة للمعالجة</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label>اختر ملف (فيديو أو صورة):</label>
            <input type="file" name="file" accept="video/*,image/*" required>
            <div class="note">يدعم MP4, AVI, JPG, PNG (تجنب المسافات والنقاط المتكررة في اسم الملف)</div>
            <label>المدينة:</label>
            <input type="text" name="city" placeholder="مثال: إنماء, اليمن" required>
            <label>الفئات (بالإنجليزية بينها فاصلة):</label>
            <input type="text" name="classes" value="person, car, weapon, explosive" required>
            <button type="submit">رفع وتحليل</button>
        </form>
    </div>
</body>
</html>
"""

# ========== رفع الملف ==========
async def handle_upload(request):
    try:
        reader = await request.multipart()
        file_field = None
        city = ""
        classes_text = ""

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'file':
                file_field = part
            elif part.name == 'city':
                city = await part.text()
            elif part.name == 'classes':
                classes_text = await part.text()

        if file_field is None:
            return web.json_response({'error': 'لم يتم إرفاق ملف'}, status=400)

        original_name = file_field.filename
        safe_name = safe_filename(original_name)
        file_path = os.path.join(UPLOAD_DIR, safe_name)

        # حفظ الملف
        with open(file_path, 'wb') as f:
            while True:
                chunk = await file_field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        ext = os.path.splitext(original_name)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov']

        base_lat, base_lon = get_city_coordinates(city.strip())

        token = secrets.token_hex(16)
        sessions[token] = {
            'path': file_path,
            'type': 'video' if is_video else 'image',
            'city': city,
            'classes': classes_text,
            'base_lat': base_lat,
            'base_lon': base_lon,
            'track_positions': {} if is_video else None,
            'result': None
        }

        if is_video:
            raise web.HTTPFound(f'/live?token={token}')
        else:
            # تحليل الصورة
            classes_list = [c.strip() for c in classes_text.split(",") if c.strip()]
            if not classes_list:
                return web.json_response({'error': 'قائمة الفئات فارغة'}, status=400)
            model.set_classes(classes_list)
            img = cv2.imread(file_path)
            if img is None:
                return web.json_response({'error': 'فشل قراءة الصورة. تأكد من أن الملف صورة حقيقية واسمه بسيط (بدون مسافات طويلة).'}, status=400)

            results = model.predict(img, verbose=False)
            detected = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    class_name = classes_list[cls_idx] if cls_idx < len(classes_list) else "unknown"
                    lat = base_lat + random.uniform(-0.005, 0.005)
                    lon = base_lon + random.uniform(-0.005, 0.005)
                    detected.append({
                        "class": class_name,
                        "emoji": CLASS_EMOJI_MAP.get(class_name, "❓"),
                        "lat": lat,
                        "lon": lon
                    })
            sessions[token]['result'] = detected
            raise web.HTTPFound(f'/result?token={token}')

    except Exception as e:
        traceback.print_exc()
        return web.json_response({'error': f'خطأ داخلي: {str(e)}'}, status=500)

# ========== صفحة النتيجة للصورة ==========
async def result_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'image':
        return web.Response(text="رابط غير صالح أو منتهي الصلاحية", status=404)
    s = sessions[token]
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>نتائج التحليل</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>#map {{ height: 100vh; width: 100%; }}</style>
</head>
<body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var map = L.map('map').setView([{s['base_lat']}, {s['base_lon']}], 16);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{attribution: 'OSM'}}).addTo(map);
fetch('/api/result?token={token}')
  .then(r => r.json())
  .then(objects => {{
    if (!objects.length) alert('لم يتم اكتشاف أي كائن في الصورة.');
    objects.forEach(obj => {{
        L.marker([obj.lat, obj.lon], {{
            icon: L.divIcon({{className:'', html:'<div style="font-size:24px">'+obj.emoji+'</div>', iconSize:[30,30]}})
        }}).addTo(map).bindPopup('<b>'+obj.class+'</b>');
    }});
  }});
</script></body></html>"""
    return web.Response(text=html, content_type='text/html')

# ========== API نتيجة الصورة ==========
async def api_result(request):
    token = request.query.get('token')
    if not token or token not in sessions:
        return web.json_response({'error': 'Token غير صالح'}, status=404)
    return web.json_response(sessions[token].get('result', []))

# ========== الفيديو و WebSocket (تبقى كما السابق ولكن بنفس التحسين) ==========
async def live_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'video':
        return web.Response(text="رابط غير صالح", status=404)
    s = sessions[token]
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>مراقبة حية</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>#map{{height:100vh;width:100%}}</style>
</head>
<body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var map = L.map('map').setView([{s['base_lat']}, {s['base_lon']}], 17);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
var markers = {{}};
var ws = new WebSocket("wss://"+location.host+"/ws?token={token}");
ws.onmessage = function(event){{
    var objects = JSON.parse(event.data);
    for (var id in markers) markers[id]._alive = false;
    objects.forEach(function(obj){{
        var latlng = [obj.lat, obj.lon];
        if (markers[obj.id]) {{
            markers[obj.id].setLatLng(latlng);
            markers[obj.id]._alive = true;
        }} else {{
            var icon = L.divIcon({{className:'', html:'<div style="font-size:24px">'+obj.emoji+'</div>', iconSize:[30,30]}});
            var marker = L.marker(latlng, {{icon:icon}}).addTo(map).bindPopup('<b>'+obj.class+'</b>');
            marker._alive = true;
            markers[obj.id] = marker;
        }}
    }});
    for (var id in markers) if (!markers[id]._alive) {{ map.removeLayer(markers[id]); delete markers[id]; }}
}};
</script></body></html>"""
    return web.Response(text=html, content_type='text/html')

async def video_ws(request):
    token = request.query.get('token')
    if not token or token not in sessions:
        return web.Response(status=403)
    session = sessions[token]
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    cap = cv2.VideoCapture(session['path'])
    if not cap.isOpened():
        await ws.send_str(json.dumps({"error": "فشل فتح الفيديو"}))
        return ws

    classes_list = [c.strip() for c in session['classes'].split(",") if c.strip()]
    model.set_classes(classes_list)
    delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            results = model.track(frame, persist=True, verbose=False)
            objects = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for b, tid, cid in zip(boxes, ids, clss):
                    pid = int(tid)
                    cls_idx = int(cid)
                    cls_name = classes_list[cls_idx] if cls_idx < len(classes_list) else "unknown"
                    last = session['track_positions'].get(pid)
                    lat, lon = simulate_movement(session['base_lat'], session['base_lon'], last)
                    session['track_positions'][pid] = (lat, lon)
                    objects.append({
                        "id": pid, "lat": lat, "lon": lon,
                        "class": cls_name,
                        "emoji": CLASS_EMOJI_MAP.get(cls_name, "❓")
                    })
            # تنظيف
            cur_ids = {o['id'] for o in objects}
            for old in list(session['track_positions']):
                if old not in cur_ids:
                    del session['track_positions'][old]
            await ws.send_str(json.dumps(objects))
            await asyncio.sleep(delay)
    except Exception as e:
        traceback.print_exc()
    finally:
        cap.release()
    return ws

# ========== التطبيق ==========
app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/upload', handle_upload)
app.router.add_get('/live', live_page)
app.router.add_get('/result', result_page)
app.router.add_get('/ws', video_ws)
app.router.add_get('/api/result', api_result)

if __name__ == '__main__':
    print(f"السيرفر على المنفذ {PORT}")
    web.run_app(app, port=PORT)
