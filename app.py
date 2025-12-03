import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------- CONFIG --------------------
IMG_SIZE = (224, 224)  # ye hum niche TFLite se auto-adjust bhi kar denge
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CONF_THRESHOLD = 0.55    # below this → "not sure / crop not supported"
GREEN_THRESHOLD = 0.15   # below this → "not a clear leaf"

TFLITE_MODEL_PATH = os.path.join("models", "plant_disease_model.tflite")
CLASS_INDICES_PATH = os.path.join("models", "class_indices.json")

# -------------------- LOAD TFLITE MODEL --------------------
print("Loading TFLite model from:", TFLITE_MODEL_PATH)
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input shape: [1, H, W, C]
input_shape = input_details[0]["shape"]
if len(input_shape) == 4:
    # override IMG_SIZE based on actual model input
    IMG_SIZE = (int(input_shape[1]), int(input_shape[2]))
print("Model input shape:", input_shape)
print("Using IMG_SIZE:", IMG_SIZE)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
print("Classes loaded:", len(idx_to_class))


# -------------------- SIMPLE LEAF CHECK --------------------
def estimate_green_ratio(path):
    """
    Rough check: how much of the image looks green-ish (leaf-like).
    This helps us reject smartwatch / random images.
    """
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.array(img)

    r = arr[:, :, 0].astype("float32")
    g = arr[:, :, 1].astype("float32")
    b = arr[:, :, 2].astype("float32")

    green_mask = (g > 80) & (g > r + 10) & (g > b + 10)
    green_ratio = green_mask.sum() / (128 * 128)

    return float(green_ratio)


# -------------------- AGRO KNOWLEDGE BASE --------------------
# (same as your existing code)
knowledge_base = {
    "Tomato___Late_blight": {
        "en": {
            "name": "Tomato Late Blight",
            "description": "Serious fungal disease causing dark, water-soaked lesions on leaves, stems and fruits. Often spreads rapidly in cool, wet weather.",
            "treatment": [
                "Remove and destroy heavily infected leaves and fruits.",
                "Avoid overhead irrigation; prefer drip irrigation.",
                "Use registered fungicides containing copper or mancozeb as per local guidelines.",
                "Do not save seed from infected plants."
            ],
            "fertilizer": [
                "Avoid excessive nitrogen — it makes plants more lush and susceptible.",
                "Maintain balanced NPK; add potash (K) to improve disease tolerance.",
                "Use well-decomposed compost, not fresh manure."
            ]
        },
        "hi": {
            "name": "टमाटर का लेट ब्लाइट",
            "description": "गंभीर फफूंदी रोग जो पत्तियों, तनों और फलों पर काले, पानी भरे धब्बे बनाता है। ठंडे और नम मौसम में बहुत तेज़ी से फैलता है।",
            "treatment": [
                "बहुत अधिक संक्रमित पत्तियाँ और फल तोड़कर नष्ट कर दें।",
                "ऊपर से पानी देने की बजाय ड्रिप सिंचाई का उपयोग करें।",
                "स्थानीय अनुशंसा के अनुसार कॉपर या मैनकोज़ेब युक्त फफूंदनाशी का छिड़काव करें।",
                "संक्रमित पौधों से बीज न रखें।"
            ],
            "fertilizer": [
                "बहुत अधिक नाइट्रोजन (Urea) देने से बचें।",
                "संतुलित NPK दें, विशेष रूप से पोटाश (K) बढ़ाएँ।",
                "कच्ची गोबर खाद की जगह अच्छी तरह सड़ी हुई गोबर/कम्पोस्ट का उपयोग करें।"
            ]
        }
    },
    "Tomato___Early_blight": {
        "en": {
            "name": "Tomato Early Blight",
            "description": "Causes brown concentric rings on older leaves, starting from lower canopy and moving upward.",
            "treatment": [
                "Remove lower infected leaves and keep the field clean.",
                "Use crop rotation; avoid planting tomato after potato or tomato.",
                "Fungicide sprays (chlorothalonil, mancozeb etc.) as per recommendation."
            ],
            "fertilizer": [
                "Maintain good potassium and calcium to strengthen leaves.",
                "Avoid continuous heavy nitrogen doses.",
                "Use FYM/compost to improve soil health."
            ]
        },
        "hi": {
            "name": "टमाटर का अर्ली ब्लाइट",
            "description": "पुरानी पत्तियों पर भूरे गोल-गोल छल्ले जैसे धब्बे दिखते हैं और रोग नीचे से ऊपर की ओर फैलता है।",
            "treatment": [
                "नीचे की संक्रमित पत्तियाँ तोड़कर खेत से बाहर फेंकें।",
                "लगातार टमाटर या आलू की फसल न लें, फसल चक्र अपनाएँ।",
                "अनुशंसित फफूंदनाशकों का छिड़काव करें (जैसे क्लोरोथालोनिल, मैनकोज़ेब)।"
            ],
            "fertilizer": [
                "पोटाश और कैल्शियम की पर्याप्त मात्रा दें।",
                "बार-बार केवल यूरिया डालने से बचें।",
                "जैविक खाद / कम्पोस्ट से मिट्टी की सेहत सुधारें।"
            ]
        }
    },
    "Tomato___healthy": {
        "en": {
            "name": "Healthy Tomato Leaf",
            "description": "Leaf appears healthy, uniformly green with no visible spots or blights.",
            "treatment": [
                "Maintain regular irrigation and avoid water stress.",
                "Monitor plants weekly for early signs of disease.",
                "Use mulch to reduce soil splash and weeds."
            ],
            "fertilizer": [
                "Follow balanced fertilizer schedule (NPK).",
                "Add compost or vermicompost to maintain soil organic matter.",
                "Apply micronutrient spray if mild yellowing appears."
            ]
        },
        "hi": {
            "name": "स्वस्थ टमाटर की पत्ती",
            "description": "पत्तियाँ हरी और समान रूप से विकसित दिखती हैं, उन पर कोई दाग या सड़न के लक्षण नहीं हैं।",
            "treatment": [
                "नियमित सिंचाई रखें, न बहुत अधिक न बहुत कम।",
                "हर हफ्ते पत्तियों की जाँच करें ताकि शुरुआती लक्षण जल्दी पकड़ में आएँ।",
                "मल्च (घास/पोलिथीन) का उपयोग करें ताकि मिट्टी के छींटे कम लगें।"
            ],
            "fertilizer": [
                "संतुलित NPK खाद का पालन करें।",
                "वर्मी-कम्पोस्ट/गोबर खाद डालें ताकि मिट्टी में कार्बनिक पदार्थ बने रहें।",
                "हल्की पीलापन होने पर सूक्ष्म पोषक तत्वों का स्प्रे करें।"
            ]
        }
    },
    "Potato___Late_blight": {
        "en": {
            "name": "Potato Late Blight",
            "description": "Highly destructive disease causing dark patches on leaves and tubers; can wipe out crop in cool, wet weather.",
            "treatment": [
                "Destroy infected haulm (foliage) before harvesting.",
                "Use resistant varieties when available.",
                "Apply protective and systemic fungicides as recommended."
            ],
            "fertilizer": [
                "Avoid waterlogging; ensure good drainage.",
                "Do not overuse nitrogen; maintain high potash.",
                "Apply gypsum if soil is low in calcium."
            ]
        },
        "hi": {
            "name": "आलू का लेट ब्लाइट",
            "description": "बहुत विनाशकारी रोग, जो पत्तियों और कंदों पर काले धब्बे बनाता है और ठंडे-नम मौसम में पूरी फसल नष्ट कर सकता है।",
            "treatment": [
                "कटाई से पहले संक्रमित हरी भाग (पत्तियाँ/तने) को नष्ट कर दें।",
                "जहाँ संभव हो, रोग सहनशील किस्मों का उपयोग करें।",
                "सुरक्षात्मक और सिस्टमिक दोनों प्रकार के फफूंदनाशी का छिड़काव करें।"
            ],
            "fertilizer": [
                "खेत में पानी भराव न होने दें, निकास व्यवस्था अच्छी रखें।",
                "नाइट्रोजन की अधिक मात्रा से बचें, पोटाश की मात्रा पर्याप्त रखें।",
                "जहाँ मिट्टी में कैल्शियम की कमी हो, वहाँ जिप्सम का प्रयोग करें।"
            ]
        }
    }
}


def get_advice(label, lang):
    kb = knowledge_base.get(label)
    if kb:
        data = kb.get(lang) or kb.get("en")
        return data

    # generic fallback
    if lang == "hi":
        return {
            "name": label,
            "description": "इस बीमारी के बारे में विस्तृत जानकारी अभी सिस्टम में नहीं जोड़ी गयी है।",
            "treatment": ["सही निदान और उपचार के लिए अपने नज़दीकी कृषि विशेषज्ञ से संपर्क करें।"],
            "fertilizer": ["संतुलित खाद योजना अपनाएँ और मिट्टी की जाँच रिपोर्ट के अनुसार खाद डालें।"]
        }
    else:
        return {
            "name": label,
            "description": "Detailed agronomy info for this disease is not yet added.",
            "treatment": ["Consult your local agriculture expert for exact diagnosis and treatment."],
            "fertilizer": ["Follow a balanced fertilizer program based on soil test reports."]
        }


# -------------------- PREDICTION (TFLITE) --------------------
def predict_image(path):
    # PIL + numpy se image load & resize
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0

    # batch dimension
    batch = np.expand_dims(arr, axis=0)

    # dtype align with tflite model
    batch = batch.astype(input_details[0]["dtype"])

    # set input
    interpreter.set_tensor(input_details[0]["index"], batch)

    # run inference
    interpreter.invoke()

    # get output
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = idx_to_class[idx]

    # debug
    print("\n--- PREDICTION DEBUG ---")
    print("file:", path)
    print("top:", label, confidence)
    for i in np.argsort(preds)[::-1][:3]:
        print("  ", idx_to_class[int(i)], float(preds[i]))
    print("------------------------\n")

    return label, confidence, preds



# -------------------- FLASK APP --------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    lang = request.form.get("lang", "en")

    if file.filename == "":
        return "Empty filename", 400

    safe_name = secure_filename(file.filename)
    filename = f"{len(os.listdir(UPLOAD_FOLDER))}_{safe_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 1) check whether this even looks like a leaf
    green_ratio = estimate_green_ratio(filepath)
    print("Green ratio:", green_ratio)

    if green_ratio < GREEN_THRESHOLD:
        # Not a leaf / very unclear
        if lang == "hi":
            info = {
                "name": "पत्ती की स्पष्ट पहचान नहीं",
                "description": "यह छवि पत्ती जैसी स्पष्ट नहीं दिख रही है। कृपया केवल पत्ती की नज़दीकी और साफ़ फोटो अपलोड करें।",
                "treatment": ["सही परिणाम के लिए एक पत्ती की साफ़, नज़दीकी फोटो लें।"],
                "fertilizer": []
            }
        else:
            info = {
                "name": "Leaf not clearly detected",
                "description": "This image does not look like a clear leaf photo. Please upload a close-up image of the leaf only.",
                "treatment": ["For accurate results, capture a clear, close-up photo of only the leaf."],
                "fertilizer": []
            }

        return render_template(
            "result.html",
            label="Unknown / Not a leaf",
            confidence=0.0,
            info=info,
            lang=lang,
            top_preds=[]
        )

    # 2) normal model prediction
    label, confidence, prob_vec = predict_image(filepath)

    if confidence < CONF_THRESHOLD:
        # low confidence → we admit we aren't sure
        if lang == "hi":
            info = {
                "name": "विश्वास कम है / फसल सपोर्टेड नहीं",
                "description": "मॉडल इस पत्ती के बारे में आश्वस्त नहीं है। संभव है कि यह फसल अभी सिस्टम में शामिल न हो।",
                "treatment": ["कृपया नज़दीकी कृषि विशेषज्ञ या किसान सलाह केंद्र से संपर्क करें।"],
                "fertilizer": []
            }
        else:
            info = {
                "name": "Low confidence / Crop not supported",
                "description": "The model is not confident about this leaf image. This crop may not be supported yet.",
                "treatment": ["Please consult your local agriculture expert for correct diagnosis and treatment."],
                "fertilizer": []
            }
    else:
        info = get_advice(label, lang)

    top_indices = np.argsort(prob_vec)[::-1][:3]
    top_preds = [
        {"label": idx_to_class[int(i)], "confidence": float(prob_vec[i])}
        for i in top_indices
    ]

    return render_template(
        "result.html",
        label=label,
        confidence=confidence,
        info=info,
        lang=lang,
        top_preds=top_preds
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)