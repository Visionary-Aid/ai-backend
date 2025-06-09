from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
from gtts import gTTS
import os
import uuid

app = FastAPI()
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 5)  # 5 classes
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
model.eval()
class_names = ['person', 'car', 'dog', 'cat', 'bicycle']
label_translations = {
    'person': 'شخص',
    'car': 'سيارة',
    'dog': 'كلب',
    'cat': 'قطة',
    'bicycle': 'دراجة'
}
weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    temp_image_path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
    
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())

        
        image = Image.open(temp_image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        
        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            label = class_names[predicted_index]
            arabic_label = label_translations[label]

        
        speech_text = f"الصورة تحتوي على {arabic_label}"
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=speech_text, lang='ar')
        tts.save(audio_filename)

        return {
            "predicted_label": label,
            "arabic_label": arabic_label,
            "audio_file": f"/audio/{audio_filename}"
        }

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(".", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    return JSONResponse(status_code=404, content={"message": "Audio file not found"})
