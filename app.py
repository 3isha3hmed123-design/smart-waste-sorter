import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os
import google.generativeai as genai

# ==============================
#   إعدادات Google Gemini (API)
# ==============================

GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", "")

if GENAI_API_KEY == "":
    genai_configured = False
else:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        model_gemini = genai.GenerativeModel("gemini-1.0-pro")
        genai_configured = True
    except Exception:
        genai_configured = False

# ==============================
#   إعدادات Teachable Machine
# ==============================
TM_MODEL_PATH = "keras_model.h5"
TM_LABELS_PATH = "labels.txt"

tm_loaded = False
tm_model = None
tm_class_names = None

try:
    if os.path.exists(TM_MODEL_PATH) and os.path.exists(TM_LABELS_PATH):
        tm_model = tf.keras.models.load_model(TM_MODEL_PATH, compile=False)
        with open(TM_LABELS_PATH, "r", encoding="utf-8") as f:
            tm_class_names = [line.strip() for line in f.readlines()]
        tm_loaded = True
except Exception as e:
    tm_loaded = False

# ==============================
#       واجهة التطبيق - إعدادات
# ==============================
st.set_page_config(
    page_title="مساعد فرز النفايات الذكي",
    page_icon="♻️",
    layout="wide"
)

if "last_waste_type" not in st.session_state:
    st.session_state["last_waste_type"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ==============================
#   دالة تصنيف الصورة TM
# ==============================
def classify_waste_teachable_machine(image):
    if not tm_loaded:
        return "غير معروف", "⚠ نموذج Teachable Machine غير محمّل."

    size = (224, 224)
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    array = np.asarray(image).astype(np.float32)

    normalized = (array / 127.5) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    prediction = tm_model.predict(data)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])
    label_raw = tm_class_names[index]

    mapping = {
        "plastic": "بلاستيك",
        "paper": "ورق",
        "glass": "زجاج",
        "metal": "معدن",
        "organic": "نفايات عضوية",
        "mixed": "مختلطة",
    }

    waste_type = mapping.get(label_raw, label_raw)

    explain = {
        "بلاستيك": "اغسل البلاستيك وضعه في حاوية إعادة التدوير.",
        "ورق": "ضع الورق الجاف في حاوية الورق.",
        "زجاج": "اشطف الزجاج وضعه في حاوية الزجاج.",
        "معدن": "اغسل العلب المعدنية ثم اضغطها قليلاً.",
        "نفايات عضوية": "يمكن تحويل بقايا الطعام إلى سماد عضوي.",
        "مختلطة": "حاول فصل المكونات قبل التخلص منها."
    }

    return waste_type, f"{explain.get(waste_type, '')}\n\nنسبة الثقة: {confidence*100:.1f}%"

# ==============================
#   دالة الشاتبوت (Gemini)
# ==============================
def recycling_chatbot_ai(message, last_type):
    if not genai_configured:
        return "⚠ مفتاح Gemini غير مضاف. ضعي GENAI_API_KEY داخل أسرار Streamlit."

    prompt = f"""
أنت مساعد ذكي متخصص في إعادة التدوير والبيئة.
أجب بالعربية بشكل مختصر وواضح ومفيد.

آخر نوع نفاية تعرفنا عليه: {last_type}

سؤال المستخدم:
{message}
"""

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ حدث خطأ أثناء الاتصال بالنموذج:\n{e}"

# ==============================
#         واجهة التطبيق
# ==============================
def main():

    st.title("♻️ مساعد فرز النفايات الذكي")
    st.write("يعتمد على Teachable Machine لتصنيف الصور، وGemini للإجابة على الأسئلة.")

    tab1, tab2, tab3 = st.tabs(["📸 تصنيف النفايات", "💬 شاتبوت", "ℹ️ معلومات"])

    # ---------------------------------------------------
    # تبويب: تصنيف الصورة
    # ---------------------------------------------------
    with tab1:
        st.header("📸 ارفعي صورة للنفاية")
        uploaded = st.file_uploader("اختاري صورة:", type=["jpg", "jpeg", "png"])

        if uploaded:
            img = Image.open(uploaded)
            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption="الصورة المدخلة")

            with col2:
                label, info = classify_waste_teachable_machine(img)
                st.success(f"النتيجة: {label}")
                st.info(info)
                st.session_state["last_waste_type"] = label

    # ---------------------------------------------------
    # تبويب: شاتبوت
    # ---------------------------------------------------
    with tab2:
        st.header("💬 اسأل عن إعادة التدوير")

        if st.session_state["last_waste_type"]:
            st.write(f"آخر نوع نفايات: **{st.session_state['last_waste_type']}**")

        msg = st.text_input("اكتب سؤالك:")
        if st.button("إرسال"):
            answer = recycling_chatbot_ai(msg, st.session_state["last_waste_type"])
            st.session_state["chat_history"].append(("أنت", msg))
            st.session_state["chat_history"].append(("المساعد", answer))

        for sender, text in st.session_state["chat_history"]:
            if sender == "أنت":
                st.markdown(f"**🧑‍🎓 أنت:** {text}")
            else:
                st.markdown(f"**🤖 المساعد:** {text}")

    # ---------------------------------------------------
    # تبويب: معلومات
    # ---------------------------------------------------
    with tab3:
        st.header("ℹ️ عن المشروع")
        st.write("""
هذا التطبيق يساعد المستخدم على:
- تصنيف النفايات باستخدام نموذج Teachable Machine  
- طرح أسئلة متعلقة بإعادة التدوير  
- فهم علاقة إعادة التدوير بالتغير المناخي  
        """)

# تشغيل التطبيق
if __name__ == "__main__":
    main()
