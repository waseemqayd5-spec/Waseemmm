import asyncio
import json
import cv2
import random
import os
import secrets
import time
import traceback
import numpy as np
from datetime import datetime
from collections import defaultdict
from aiohttp import web
from ultralytics import YOLO
from geopy.geocoders import Nominatim

# ====================== إعدادات ======================
PORT = int(os.environ.get("PORT", 5000))
UPLOAD_DIR = "/tmp/geo_uploads" if os.path.exists("/tmp") else "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# المناطق الحساسة (مثال حول نقطة افتراضية)
SENSITIVE_ZONES = []  # يتم تعبئتها لاحقًا حسب المدينة

# ====================== تحميل النموذج ======================
MODEL_PATH = "models/best.pt"  # إن وُجد نموذج مخصص
if os.path.exists(MODEL_PATH):
    print(f"✅ تحميل نموذج مخصص: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print("ℹ️ لم يتم العثور على نموذج مخصص. تحميل yolov8n (خفيف)...")
    model = YOLO('yolov8n.pt')

# الفئات التي يمكن أن يكتشفها النموذج الحالي
CLASS_NAMES = list(model.names.values()) if hasattr(model, 'names') else [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'
]

# رموز وألوان للعرض
CLASS_EMOJI_MAP = {
    "person": "👤", "car": "🚗", "motorcycle": "🏍", "bus": "🚌", "truck": "🚚",
    "weapon": "🔫", "explosive": "💣", "rifle": "🔫", "knife": "🔪"
}
COLOR_MAP = {
    "person": "#e74c3c", "car": "#3498db", "motorcycle": "#2ecc71",
    "bus": "#e67e22", "truck": "#9b59b6", "weapon": "#34495e", "explosive": "#f1c40f"
}

# ====================== تخزين الجلسات ======================
sessions = {}

# ====================== أدوات مساعدة ======================
def get_city_coords(city_name):
    geolocator = Nominatim(user_agent="saqr_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    # احتياطي: مدن معروفة
    if "انماء" in city_name or "إنماء" in city_name:
        return 12.8000, 45.0333
    if "عدن" in city_name:
        return 12.8000, 45.0333
    # افتراضي صنعاء
    return 15.3694, 44.1910

def simulate_gps_movement(base_lat, base_lon, last_pos=None):
    if last_pos:
        lat, lon = last_pos
        lat += random.uniform(-0.0001, 0.0001)
        lon += random.uniform(-0.0001, 0.0001)
        return lat, lon
    return base_lat + random.uniform(-0.0005, 0.0005), base_lon + random.uniform(-0.0005, 0.0005)

# ====================== محرك تحليل السلوك ======================
class BehaviorAnalyzer:
    def __init__(self):
        self.tracks = defaultdict(list)  # track_id -> list of positions

    def update(self, track_id, position):
        self.tracks[track_id].append(position)
        if len(self.tracks[track_id]) > 30:
            self.tracks[track_id].pop(0)

    def get_behavior_score(self, track_id, class_name, sensitive_zones):
        if track_id not in self.tracks or len(self.tracks[track_id]) < 3:
            return 0.0
        path = np.array(self.tracks[track_id])
        score = 0.0

        # 1. سرعة مفاجئة
        if len(path) >= 2:
            speeds = np.linalg.norm(np.diff(path, axis=0), axis=1)
            threshold = {'person': 0.0002, 'car': 0.0005, 'motorcycle': 0.0007}.get(class_name, 0.0003)
            if np.max(speeds) > threshold:
                score += 0.3

        # 2. قرب من منطقة حساسة
        if sensitive_zones:
            current = path[-1]
            for zone in sensitive_zones:
                dist = np.linalg.norm(current - np.array(zone['center']))
                if dist < zone['radius']:
                    score += 0.4
                    break

        # 3. تسكع (ثبات طويل)
        if len(path) > 15:
            dispersion = np.std(path[-15:], axis=0).mean()
            if dispersion < 0.00005:
                score += 0.3

        return min(score, 1.0)

# ====================== مقياس التهديد ======================
class ThreatEvaluator:
    def __init__(self, behavior_analyzer):
        self.behavior = behavior_analyzer
        self.history = []  # أحداث سابقة للتأثير على القرار

    def evaluate(self, track_id, class_name, detection_conf, position, sensitive_zones):
        obj_score = detection_conf
        beh_score = self.behavior.get_behavior_score(track_id, class_name, sensitive_zones)

        # النمط التاريخي
        hist_score = 0.0
        for event in self.history[-50:]:
            if event['class'] == class_name and event['was_threat']:
                dist = np.linalg.norm(np.array(position) - np.array(event['position']))
                if dist < 0.001:
                    hist_score += 0.2
        hist_score = min(hist_score, 1.0)

        threat = (obj_score + beh_score + hist_score) / 3
        return min(threat, 1.0)

    def add_event(self, class_name, position, was_threat):
        self.history.append({
            'class': class_name,
            'position': position,
            'was_threat': was_threat,
            'time': time.time()
        })

# ====================== تعلم ذاتي (تغذية راجعة) ======================
class FeedbackCollector:
    def __init__(self):
        self.feedback = []

    def add(self, track_id, class_name, image_patch, ai_threat, human_label):
        self.feedback.append({
            'track_id': track_id,
            'class': class_name,
            'image': image_patch,  # يمكن حفظه كملف لاحقًا
            'ai_threat': ai_threat,
            'human_label': human_label  # 'threat' أو 'false_alarm'
        })

    def get_pending_retraining(self):
        # إذا تجاوز العدد حدًا، نبلغ المشغل بضرورة تنزيل البيانات وإعادة التدريب
        if len(self.feedback) >= 100:
            return True
        return False

# ====================== صفحة البداية ======================
async def index(request):
    return web.Response(text=INDEX_HTML, content_type='text/html')

INDEX_HTML = """<!DOCTYPE html>
<html lang="ar">
<head><meta charset="UTF-8"><title>الصقر - نظام الاستخبارات</title>
<style>
body { font-family:Arial; background:#1a1a2e; color:#eee; display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0; direction:rtl; }
.container { background:#16213e; padding:30px; border-radius:15px; max-width:500px; width:90%; }
h2 { color:#e94560; text-align:center; }
label { display:block; margin:10px 0 5px; }
input[type="file"], input[type="text"] { width:100%; padding:10px; background:#1a1a2e; border:1px solid #0f3460; color:#eee; }
button { width:100%; padding:15px; background:#e94560; color:white; border:none; border-radius:5px; font-size:18px; margin-top:20px; }
.note { font-size:12px; color:#888; }
</style>
</head>
<body><div class="container">
<h2>🦅 رفع فيديو أو صورة</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
<label>اختر ملف:</label>
<input type="file" name="file" accept="video/*,image/*" required>
<div class="note">يدعم MP4, AVI, JPG, PNG (اسم بسيط)</div>
<label>المدينة:</label>
<input type="text" name="city" placeholder="مثال: إنماء" required>
<label>الفئات (إنجليزية مفصولة بفاصلة):</label>
<input type="text" name="classes" value="person, car, truck" required>
<button type="submit">رفع وتحليل</button>
</form>
</div></body></html>"""

# ====================== معالجة الرفع ======================
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
                city = (await part.text()).strip()
            elif part.name == 'classes':
                classes_text = (await part.text()).strip()

        if file_field is None:
            return web.Response(text="<h2>خطأ: لم يتم إرفاق ملف</h2>", content_type='text/html', status=400)

        original_name = file_field.filename
        safe_name = f"{secrets.token_hex(8)}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(file_path, 'wb') as f:
            while True:
                chunk = await file_field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        ext = os.path.splitext(original_name)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov']
        if not is_video and ext not in ['.jpg', '.jpeg', '.png']:
            return web.Response(text="<h2>نوع الملف غير مدعوم</h2>", content_type='text/html', status=400)

        base_lat, base_lon = get_city_coords(city)
        # تعريف منطقة حساسة افتراضية (دائرة نصف قطرها درجة تقريباً)
        zones = [{'center': (base_lat, base_lon), 'radius': 0.002}]

        token = secrets.token_hex(16)
        sessions[token] = {
            'path': file_path,
            'type': 'video' if is_video else 'image',
            'city': city,
            'classes': classes_text,
            'base_lat': base_lat,
            'base_lon': base_lon,
            'zones': zones,
            'analyzer': BehaviorAnalyzer(),
            'evaluator': ThreatEvaluator(BehaviorAnalyzer()),
            'feedback': FeedbackCollector(),
            'track_positions': {},
            'result': None
        }

        if is_video:
            raise web.HTTPFound(f'/live?token={token}')
        else:
            # معالجة الصورة
            img = cv2.imread(file_path)
            if img is None:
                return web.Response(text="<h2>فشل قراءة الصورة</h2>", content_type='text/html', status=400)

            results = model(img, verbose=False)
            detected = []
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    conf = float(box.conf[0])
                    lat = base_lat + random.uniform(-0.005, 0.005)
                    lon = base_lon + random.uniform(-0.005, 0.005)
                    detected.append({
                        "class": class_name,
                        "emoji": CLASS_EMOJI_MAP.get(class_name, "❓"),
                        "lat": lat,
                        "lon": lon,
                        "conf": conf,
                        "threat": 0.0
                    })
            sessions[token]['result'] = detected
            raise web.HTTPFound(f'/result?token={token}')

    except Exception as e:
        traceback.print_exc()
        return web.Response(text=f"<h2>خطأ داخلي: {str(e)}</h2>", content_type='text/html', status=500)

# ====================== صفحة النتيجة (صورة) ======================
async def result_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'image':
        return web.Response(text="<h2>رابط غير صالح</h2>", content_type='text/html', status=404)
    s = sessions[token]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>نتائج التحليل</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>#map {{ height: 100vh; width: 100%; }}</style>
</head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var map = L.map('map').setView([{s['base_lat']}, {s['base_lon']}], 16);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
fetch('/api/result?token={token}')
  .then(r => r.json())
  .then(objects => {{
    if (!objects.length) alert('لم يتم اكتشاف أي كائن.');
    objects.forEach(obj => {{
        var icon = L.divIcon({{className:'', html:'<div style="font-size:24px">'+obj.emoji+'</div>', iconSize:[30,30]}});
        L.marker([obj.lat, obj.lon], {{icon: icon}}).addTo(map)
          .bindPopup('<b>'+obj.class+'</b><br>ثقة: '+(obj.conf*100).toFixed(1)+'%');
    }});
  }});
</script></body></html>"""
    return web.Response(text=html, content_type='text/html')

async def api_image_result(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'image':
        return web.json_response({'error': 'غير موجود'}, status=404)
    return web.json_response(sessions[token].get('result', []))

# ====================== صفحة الفيديو (Live) ======================
async def live_page(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'video':
        return web.Response(text="<h2>رابط غير صالح</h2>", content_type='text/html', status=404)
    s = sessions[token]
    return web.Response(text=LIVE_HTML.format(
        base_lat=s['base_lat'],
        base_lon=s['base_lon'],
        token=token
    ), content_type='text/html')

LIVE_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>الصقر - مراقبة حية</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
    #map {{ height: 70vh; width: 100%; }}
    #alerts {{ max-height: 30vh; overflow-y: auto; background: #16213e; padding: 10px; }}
    .alert {{ border-bottom: 1px solid #333; padding: 8px; }}
    .threat-high {{ color: red; font-weight: bold; }}
    .threat-medium {{ color: orange; }}
    .threat-low {{ color: green; }}
    button {{ margin: 2px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="alerts"><h3>🚨 التنبيهات</h3></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var map = L.map('map').setView([{base_lat}, {base_lon}], 17);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
var markers = {{}};
var alertsDiv = document.getElementById('alerts');

var ws = new WebSocket("wss://"+location.host+"/ws?token={token}");
ws.onmessage = function(event){{
    var data = JSON.parse(event.data);
    var objects = data.objects || [];
    var threats = data.threats || [];

    // تحديث العلامات
    for (var id in markers) markers[id]._alive = false;
    objects.forEach(function(obj){{
        var latlng = [obj.lat, obj.lon];
        var iconHtml = '<div style="font-size:24px">'+obj.emoji+'</div>';
        if (obj.threat > 0.7) iconHtml = '<div style="font-size:28px; color:red;">⚠️</div>';
        if (markers[obj.id]) {{
            markers[obj.id].setLatLng(latlng);
            markers[obj.id]._alive = true;
        }} else {{
            var icon = L.divIcon({{className:'', html: iconHtml, iconSize:[30,30]}});
            var marker = L.marker(latlng, {{icon:icon}}).addTo(map)
                .bindPopup('<b>'+obj.class+'</b><br>تهديد: '+(obj.threat*100).toFixed(0)+'%');
            marker._alive = true;
            markers[obj.id] = marker;
        }}
    }});
    for (var id in markers) if (!markers[id]._alive) {{ map.removeLayer(markers[id]); delete markers[id]; }}

    // عرض التنبيهات
    alertsDiv.innerHTML = '<h3>🚨 التنبيهات</h3>';
    threats.forEach(function(t){{
        var cls = t.threat > 0.7 ? 'threat-high' : (t.threat > 0.4 ? 'threat-medium' : 'threat-low');
        alertsDiv.innerHTML += '<div class="alert '+cls+'">' +
            '<b>'+t.class+'</b> تهديد: '+(t.threat*100).toFixed(1)+'% ' +
            '<button onclick="sendFeedback('+t.track_id+',\'threat\')">✔️ تأكيد</button>' +
            '<button onclick="sendFeedback('+t.track_id+',\'false_alarm\')">✖️ رفض</button>' +
            '</div>';
    }});
}};

function sendFeedback(trackId, label) {{
    ws.send(JSON.stringify({{feedback: {{track_id: trackId, label: label}}}}));
    alert("تم إرسال التغذية الراجعة");
}}
</script>
</body></html>"""

# ====================== WebSocket معالج الفيديو ======================
async def video_ws(request):
    token = request.query.get('token')
    if not token or token not in sessions or sessions[token]['type'] != 'video':
        return web.Response(status=403)

    sess = sessions[token]
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    cap = cv2.VideoCapture(sess['path'])
    if not cap.isOpened():
        await ws.send_str(json.dumps({"error": "فشل فتح الفيديو"}))
        return ws

    class_filter = [c.strip() for c in sess['classes'].split(",") if c.strip()]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 / fps

    # كائنات تحليلة
    analyzer = sess['analyzer']
    evaluator = sess['evaluator']
    feedback_collector = sess['feedback']

    async def send_data(objects, threats):
        await ws.send_str(json.dumps({"objects": objects, "threats": threats}))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = model.track(frame, persist=True, verbose=False)
            current_objects = []
            threats = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, tid, cls_id, conf in zip(boxes, track_ids, clss, confs):
                    tid = int(tid)
                    cls_id = int(cls_id)
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

                    # إن لم تكن الفئة ضمن المرشحات، تجاهل
                    if class_filter and class_name not in class_filter:
                        continue

                    # محاكاة الموقع (يمكن استبدالها ببيانات حقيقية)
                    lat, lon = simulate_gps_movement(sess['base_lat'], sess['base_lon'],
                                                     sess['track_positions'].get(tid))
                    sess['track_positions'][tid] = (lat, lon)

                    # تحديث المحلل السلوكي
                    analyzer.update(tid, (lat, lon))

                    # تقييم التهديد
                    threat = evaluator.evaluate(tid, class_name, float(conf), (lat, lon), sess['zones'])
                    emoji = CLASS_EMOJI_MAP.get(class_name, "❓")

                    obj = {
                        "id": tid,
                        "class": class_name,
                        "emoji": emoji,
                        "lat": lat,
                        "lon": lon,
                        "conf": float(conf),
                        "threat": round(threat, 2)
                    }
                    current_objects.append(obj)

                    if threat > 0.4:
                        threats.append(obj)

            # تنظيف المسارات القديمة
            current_ids = {o['id'] for o in current_objects}
            for old_id in list(analyzer.tracks.keys()):
                if old_id not in current_ids:
                    del analyzer.tracks[old_id]
            for old_id in list(sess['track_positions'].keys()):
                if old_id not in current_ids:
                    del sess['track_positions'][old_id]

            await send_data(current_objects, threats)

            # استقبال رسائل التغذية الراجعة من العميل
            try:
                msg = await ws.receive(timeout=0.01)
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if 'feedback' in data:
                        fb = data['feedback']
                        feedback_collector.add(
                            track_id=fb['track_id'],
                            class_name="غير معروف",
                            image_patch=None,
                            ai_threat=0.0,
                            human_label=fb['label']
                        )
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            await asyncio.sleep(delay)

    except Exception as e:
        traceback.print_exc()
    finally:
        cap.release()
    return ws

# ====================== التطبيق ======================
app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/upload', handle_upload)
app.router.add_get('/live', live_page)
app.router.add_get('/result', result_page)
app.router.add_get('/ws', video_ws)
app.router.add_get('/api/result', api_image_result)

if __name__ == '__main__':
    print(f"🦅 الصقر يعمل على المنفذ {PORT}")
    web.run_app(app, port=PORT)
