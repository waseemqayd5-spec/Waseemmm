# server.py
import asyncio
import json
import cv2
import random
import os
import secrets
from aiohttp import web
from ultralytics import YOLO
from geopy.geocoders import Nominatim

# ========== إعدادات المنفذ والمجلدات ==========
PORT = int(os.environ.get("PORT", 5000))
UPLOAD_DIR = "/tmp/geo_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ========== إعدادات الفئات الافتراضية والرموز ==========
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
COLOR_MAP = {
    "person": "#e74c3c",
    "car": "#3498db",
    "motorcycle": "#2ecc71",
    "bus": "#e67e22",
    "truck": "#9b59b6",
    "weapon": "#34495e",
    "explosive": "#f1c40f"
}

# ========== تحميل النموذج مرة واحدة عند بدء التشغيل ==========
print("تحميل YOLO-World... (قد يستغرق تحميل النموذج أول مرة بضع دقائق)")
model = YOLO('yolov8s-world.pt')

# ========== تخزين الجلسات المؤقتة ==========
sessions = {}  # token -> بيانات الجلسة

# ========== دالة تحديد إحداثيات المدينة ==========
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="geo_upload_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    # إحداثيات افتراضية لإنماء (اليمن)
    return 12.8000, 45.0333

# ========== دالة محاكاة حركة بسيطة للأهداف في الفيديو ==========
def simulate_movement(base_lat, base_lon, last_pos=None):
    if last_pos:
        lat, lon = last_pos
        lat += random.uniform(-0.0001, 0.0001)
        lon += random.uniform(-0.0001, 0.0001)
        return lat, lon
    return base_lat + random.uniform(-0.0005, 0.0005), base_lon + random.uniform(-0.0005, 0.0005)

# ========== الصفحة الرئيسية (رفع الملف) ==========
async def index(request):
    html = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام الاستخبارات الجغرافية</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            direction: rtl;
        }
        .container {
            background: #16213e;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            max-width: 500px;
            width: 90%;
        }
        h2 {
            color: #e94560;
            text-align: center;
            margin-bottom: 30px;
        }
        label {
            font-weight: bold;
            display: block;
            margin: 15px 0 5px;
        }
        input[type="file"], input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #0f3460;
            border-radius: 5px;
            background: #1a1a2e;
            color: #eee;
        }
        button {
            width: 100%;
            padding: 15px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover {
            background: #c23152;
        }
        .note {
            font-size: 12px;
            color: #888;
            margin-top: -5px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>رفع فيديو أو صورة للمعالجة</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">اختر ملف (فيديو أو صورة):</label>
            <input type="file" id="file" name="file" accept="video/*,image/*" required>
            <div class="note">يدعم MP4, AVI, JPG, PNG</div>

            <label for="city">المدينة:</label>
            <input type="text" id="city" name="city" placeholder="مثال: إنماء, اليمن" required>

            <label for="classes">الفئات المطلوب كشفها (بالإنجليزية، مفصولة بفواصل):</label>
            <input type="text" id="classes" name="classes" value="person, car, weapon, explosive" required>
            <div class="note">مثال: person, car, motorcycle, bus, truck, weapon, explosive</div>

            <button type="submit">رفع وتحليل</button>
        </form>
    </div>
</body>
</html>
"""
    return web.Response(text=html, content_type='text/html')

# ========== استقبال الملف ومعالجته ==========
async def handle_upload(request):
    reader = await request.multipart()
    file_field = None
    city = ""
    classes_text = ""

    # قراءة الحقول من النموذج
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

    filename = file_field.filename
    if not filename:
        return web.json_response({'error': 'اسم الملف غير صحيح'}, status=400)

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png']:
        return web.json_response({'error': 'نوع الملف غير مدعوم'}, status=400)

    is_video = ext in ['.mp4', '.avi', '.mov']
    token = secrets.token_hex(16)
    file_path = os.path.join(UPLOAD_DIR, f"{token}{ext}")

    # حفظ الملف على الخادم
    with open(file_path, 'wb') as f:
        while True:
            chunk = await file_field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    # تحديد الإحداثيات من اسم المدينة
    base_lat, base_lon = get_city_coordinates(city)

    # تجهيز الجلسة
    sessions[token] = {
        'path': file_path,
        'type': 'video' if is_video else 'image',
        'city': city,
        'classes': classes_text,
        'base_lat': base_lat,
        'base_lon': base_lon,
        'track_positions': {} if is_video else None,
        'result': None  # للصور
    }

    # إذا كان فيديو -> ننتقل لصفحة البث المباشر
    if is_video:
        raise web.HTTPFound(f'/live?token={token}')
    # إذا كانت صورة -> نحللها فورًا ونخزن النتيجة
    else:
        classes_list = [c.strip() for c in classes_text.split(",")]
        model.set_classes(classes_list)
        img = cv2.imread(file_path)
        if img is None:
            return web.json_response({'error': 'فشل قراءة الصورة'}, status=400)
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
                    "color": COLOR_MAP.get(class_name, "#95a5a6"),
                    "lat": lat,
                    "lon": lon
                })
        sessions[token]['result'] = detected
        raise web.HTTPFound(f'/result?token={token}')

# ========== صفحة عرض الفيديو الحي ==========
async def live_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'video':
        return web.Response(text="رابط غير صالح", status=404)
    s = sessions[token]
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>مراقبة حية بالفيديو</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>#map {{ height: 100vh; width: 100%; }}</style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{s['base_lat']}, {s['base_lon']}], 17);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
        }}).addTo(map);
        var markers = {{}};
        var ws = new WebSocket("wss://" + location.host + "/ws?token={token}");
        ws.onmessage = function(event) {{
            var objects = JSON.parse(event.data);
            for (var id in markers) {{ markers[id]._alive = false; }}
            objects.forEach(function(obj) {{
                var latlng = [obj.lat, obj.lon];
                if (markers[obj.id]) {{
                    markers[obj.id].setLatLng(latlng);
                    markers[obj.id]._alive = true;
                }} else {{
                    var icon = L.divIcon({{
                        className: 'emoji-marker',
                        html: '<div style="font-size:24px; text-shadow: 1px 1px 2px black;">' + obj.emoji + '</div>',
                        iconSize: [30, 30], iconAnchor: [15, 15], popupAnchor: [0, -15]
                    }});
                    var marker = L.marker(latlng, {{icon: icon}}).addTo(map)
                        .bindPopup("<b>" + obj.class + "</b><br>ID: " + obj.id);
                    marker._alive = true;
                    markers[obj.id] = marker;
                }}
            }});
            for (var id in markers) {{
                if (!markers[id]._alive) {{ map.removeLayer(markers[id]); delete markers[id]; }}
            }}
        }};
    </script>
</body>
</html>"""
    return web.Response(text=html, content_type='text/html')

# ========== WebSocket لتحليل الفيديو وإرسال التحديثات ==========
async def video_ws(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'video':
        return web.Response(text="غير مصرح", status=403)

    session = sessions[token]
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    cap = cv2.VideoCapture(session['path'])
    if not cap.isOpened():
        await ws.send_str(json.dumps({"error": "فشل فتح الفيديو"}))
        return ws

    classes_list = [c.strip() for c in session['classes'].split(",")]
    model.set_classes(classes_list)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 / fps

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # إعادة التشغيل من البداية للعرض المستمر
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = model.track(frame, persist=True, verbose=False)
            objects = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                for box, tid, cid in zip(boxes, ids, class_ids):
                    pid = int(tid)
                    cls_idx = int(cid)
                    class_name = classes_list[cls_idx] if cls_idx < len(classes_list) else "unknown"
                    last = session['track_positions'].get(pid)
                    lat, lon = simulate_movement(session['base_lat'], session['base_lon'], last)
                    session['track_positions'][pid] = (lat, lon)
                    objects.append({
                        "id": pid,
                        "lat": lat,
                        "lon": lon,
                        "class": class_name,
                        "emoji": CLASS_EMOJI_MAP.get(class_name, "❓"),
                        "color": COLOR_MAP.get(class_name, "#95a5a6")
                    })
            # إزالة الأهداف التي اختفت
            current_ids = [o['id'] for o in objects]
            for old_id in list(session['track_positions'].keys()):
                if old_id not in current_ids:
                    del session['track_positions'][old_id]

            await ws.send_str(json.dumps(objects))
            await asyncio.sleep(delay)
    except Exception as e:
        print(f"خطأ في WebSocket: {e}")
    finally:
        cap.release()
    return ws

# ========== صفحة نتيجة الصورة ==========
async def result_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'image':
        return web.Response(text="رابط غير صالح", status=404)
    s = sessions[token]
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>نتائج تحليل الصورة</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>#map {{ height: 100vh; width: 100%; }}</style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{s['base_lat']}, {s['base_lon']}], 16);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
        }}).addTo(map);
        fetch('/api/result?token={token}')
            .then(r => r.json())
            .then(objects => {{
                objects.forEach(obj => {{
                    var icon = L.divIcon({{
                        className: 'emoji-marker',
                        html: '<div style="font-size:24px; text-shadow: 1px 1px 2px black;">' + obj.emoji + '</div>',
                        iconSize: [30, 30], iconAnchor: [15, 15], popupAnchor: [0, -15]
                    }});
                    L.marker([obj.lat, obj.lon], {{icon: icon}}).addTo(map)
                        .bindPopup("<b>" + obj.class + "</b>");
                }});
            }});
    </script>
</body>
</html>"""
    return web.Response(text=html, content_type='text/html')

# ========== API لنتيجة الصورة (JSON) ==========
async def api_image_result(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'image':
        return web.json_response({'error': 'غير موجود'}, status=404)
    return web.json_response(sessions[token]['result'])

# ========== إعداد التطبيق والمسارات ==========
app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/upload', handle_upload)
app.router.add_get('/live', live_page)
app.router.add_get('/result', result_page)
app.router.add_get('/ws', video_ws)
app.router.add_get('/api/result', api_image_result)

if __name__ == '__main__':
    print(f"التطبيق يعمل على المنفذ {PORT}")
    web.run_app(app, port=PORT)
