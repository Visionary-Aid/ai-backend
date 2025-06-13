from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
from gtts import gTTS
import torch
import os
import uuid

# === إعداد التطبيق ===
app = FastAPI()

# === الأصناف المستخدمة ===
class_names = ['person', 'car', 'dog', 'cat', 'bicycle']
label_translations = {
    'person': 'شخص',
    'car': 'سيارة',
    'dog': 'كلب',
    'cat': 'قطة',
    'bicycle': 'دراجة'
}

# === تحميل الموديل ===
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
model.eval()

# === التحويلات ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === مسار حفظ الملفات الصوتية المؤقتة ===
os.makedirs("audio", exist_ok=True)

# === نقطة التوقع ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_id = uuid.uuid4().hex
    temp_image_path = f"temp_{image_id}.jpg"
    audio_path = f"models/{image_id}.mp3"  # <-- هنا التغيير

    try:
        # حفظ الصورة المرفوعة مؤقتاً
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # تحميل الصورة وتحويلها
        image = Image.open(temp_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # التوقع
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]
            arabic_label = label_translations[label]

        # تحويل التوقع إلى صوت
        text = f"الصورة تحتوي على {arabic_label}"
        tts = gTTS(text, lang='ar')
        tts.save(audio_path)

        return {
            "predicted_label": label,
            "arabic_label": arabic_label,
            "audio_url": f"/audio/{image_id}.mp3"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# === نقطة تشغيل ملف الصوت ===
@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join("models", filename)  # <-- هنا التغيير
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg", filename=filename)
    return JSONResponse(status_code=404, content={"message": "Audio file not found"})