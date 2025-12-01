import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai
import tensorflow as tf
import os

# ==============================
#   إعدادات Google Generative AI (Gemini)
# ==============================
import google.generativeai as genai
import streamlit as st

# نقرأ المفتاح من Secrets في Streamlit Cloud
GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", "")

if GENAI_API_KEY == "":
    genai_configured = False
else:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        genai_configured = True
        model_gemini = genai.GenerativeModel("gemini-pro")
    except Exception:
        genai_configured = False


# ==============================
#   إعدادات Teachable Machine
# ==============================
TM_MODEL_PATH = "keras_model.h5"
TM_LABELS_PATH = "labels.txt"

tm_model = None
tm_class_names = None
tm_loaded = False

if os.path.exists(TM_MODEL_PATH) and os.path.exists(TM_LABELS_PATH):
    try:
        tm_model = tf.keras.models.load_model(TM_MODEL_PATH, compile=False)
        with open(TM_LABELS_PATH, "r", encoding="utf-8") as f:
            tm_class_names = [line.strip() for line in f.readlines()]
        tm_loaded = True
    except Exception as e:
        tm_loaded = False
        tm_load_error = str(e)
else:
    tm_load_error = "keras_model.h5 أو labels.txt غير موجودين في المجلد."

# ==============================
#   إعدادات عامة للتطبيق
# ==============================
st.set_page_config(
    page_title="مساعد فرز النفايات الذكي",
    page_icon="♻️",
    layout="wide"
)

# تهيئة حالة الجلسة
if "last_waste_type" not in st.session_state:
    st.session_state["last_waste_type"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # لحفظ المحادثة مع الشاتبوت

# ==============================
#   تنسيقات CSS بسيطة (لشكل أجمل)
# ==============================
st.markdown(
    """
    <style>
    .main, .block-container {
        direction: rtl;
        text-align: right;
        font-family: "Tahoma", "Segoe UI", sans-serif;
    }

    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        background-color: #f0f4f8;
        border: 1px solid #d0d7de;
        margin-bottom: 0.8rem;
    }

    .tag {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        margin: 0.1rem;
        border-radius: 999px;
        background-color: #e0f2f1;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
#   دالة تصنيف النفايات باستخدام Teachable Machine
# ==============================
def classify_waste_teachable_machine(image):
    """
    تصنيف النفايات باستخدام نموذج Teachable Machine (keras_model.h5 + labels.txt)
    يجب أن يكون النموذج مدرَّبًا على أنواع النفايات التي تهمك.
    """
    if not tm_loaded or tm_model is None or tm_class_names is None:
        # في حال عدم توفر النموذج نرجع رسالة خطأ ودعم مبسط
        return (
            "غير معروف",
            "⚠ لم يتم تحميل نموذج Teachable Machine. "
            "تأكدي من رفع الملفات keras_model.h5 و labels.txt إلى كولاب."
        )

    # إعداد الصورة كما هو مقترح في كود Teachable Machine
    size = (224, 224)  # المقاس الافتراضي لمعظم نماذج TM
    image = image.convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)

    # التطبيع إلى [-1, 1] كما في Teachable Machine
    normalized_image_array = (image_array / 127.5) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # التنبؤ
    prediction = tm_model.predict(data)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    raw_label = tm_class_names[index]

    # إزالة أي أرقام أو رموز غريبة من البداية (أحيانًا TM يضيف أرقامًا)
    label_clean = raw_label.strip()
    # لو عندك أسماء عربية في labels.txt يكفي هذا السطر
    predicted_label = label_clean

    # خريطة لتوحيد المسميات، عدليها حسب تسمياتك في Teachable Machine
    mapping = {
        "plastic": "بلاستيك",
        "paper": "ورق",
        "glass": "زجاج",
        "metal": "معدن",
        "organic": "نفايات عضوية",
        "mixed": "مختلطة",
        "بلاستيك": "بلاستيك",
        "ورق": "ورق",
        "زجاج": "زجاج",
        "معدن": "معدن",
        "نفايات عضوية": "نفايات عضوية",
        "مختلطة": "مختلطة"
    }

    waste_type = mapping.get(predicted_label, predicted_label)

    tips = {
        "بلاستيك": "اغسل البلاستيك من بقايا الطعام وضعه في حاوية إعادة تدوير البلاستيك.",
        "ورق": "ضع الورق الجاف والنظيف في حاوية الورق، وتجنب الورق المبلل أو المتسخ بالطعام.",
        "زجاج": "اشطف الزجاج وضعه في حاوية الزجاج، وتجنب رميه في الطبيعة.",
        "معدن": "اغسل العلب المعدنية واضغطها قليلاً ثم ضعها في حاوية المعادن.",
        "نفايات عضوية": "يمكن استخدام بقايا الطعام في صنع الكمبوست لتقليل انبعاثات الميثان.",
        "مختلطة": "حاول فصل مكونات النفايات (ورق، بلاستيك، زجاج...) قبل رميها."
    }

    explanation = tips.get(
        waste_type,
        "فرز النفايات يساعد في حماية البيئة وتقليل التلوث."
    )

    explanation += f"\n\nناتج النموذج: **{predicted_label}** بنسبة ثقة تقريبية: {confidence*100:.1f}%."

    return waste_type, explanation

# ==============================
#   دالة الشاتبوت الذكي (Gemini)
# ==============================
def recycling_chatbot_ai(message, last_waste_type):
    """
    شاتبوت ذكاء اصطناعي حقيقي يعتمد على Google Gemini
    مع تزويده بسياق آخر نوع نفايات تم تصنيفه.
    """

    if not genai_configured:
        return (
            "⚠ لا يمكن استخدام نموذج الذكاء الاصطناعي الآن.\n"
            "تحققي من إضافة مفتاح GENAI_API_KEY الصحيح في الكود داخل app.py."
        )

    context = ""
    if last_waste_type:
        context = f"آخر نوع نفايات تعرف عليه المستخدم هو: {last_waste_type}.\n"

    prompt = (
        "أنت مساعد ذكي مختص في إعادة التدوير، إدارة النفايات، وحماية البيئة. "
        "أجب بالعربية الفصحى بشكل مبسط وواضح، مع نصائح عملية للفرز وتقليل النفايات، "
        "ووضّح ارتباط ذلك بالتغير المناخي عندما يكون مناسباً.\n\n"
        f"{context}"
        "سؤال المستخدم:\n"
        f"{message}"
    )

    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return (
            "حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.\n"
            f"تفاصيل (للمطور): {e}"
        )

# ==============================
#   الدالة الرئيسية لواجهة التطبيق
# ==============================
def main():
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("## ♻️ مساعد فرز النفايات")
        st.markdown(
            """
            <div class="info-card">
            هذا التطبيق يساعدك على:
            <ul>
                <li>تصنيف نوع النفايات من صورة باستخدام Teachable Machine.</li>
                <li>الحصول على نصائح لإعادة التدوير.</li>
                <li>فهم ارتباط إدارة النفايات بالتغير المناخي.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### خطوات الاستخدام")
        st.markdown(
            "- انتقل إلى تبويب **تصنيف النفايات** وارفع صورة.\n"
            "- شاهد النوع المقترح للنفاية مع نصائح مناسبة.\n"
            "- اسأل في تبويب **شاتبوت إعادة التدوير** عن أي شيء يخص الفرز."
        )

        st.markdown("### أنواع النفايات (مثال):")
        st.markdown(
            '<span class="tag">بلاستيك</span>'
            '<span class="tag">ورق</span>'
            '<span class="tag">زجاج</span>'
            '<span class="tag">معدن</span>'
            '<span class="tag">نفايات عضوية</span>'
            '<span class="tag">مختلطة</span>',
            unsafe_allow_html=True
        )

        if not tm_loaded:
            st.markdown(
                "<div class='info-card' style='background-color:#ffecec;'>"
                "⚠ نموذج Teachable Machine لم يتم تحميله بشكل صحيح.<br>"
                "تأكدي من رفع الملفات <b>keras_model.h5</b> و <b>labels.txt</b> "
                "في نفس مجلد app.py."
                "</div>",
                unsafe_allow_html=True
            )

        if not genai_configured:
            st.markdown(
                "<div class='info-card' style='background-color:#fff4e5;'>"
                "⚠ لم يتم تفعيل نموذج Google Gemini بعد. "
                "رجاءً أضيفي مفتاح GENAI_API_KEY في الكود."
                "</div>",
                unsafe_allow_html=True
            )

    # عنوان رئيسي
    st.title("♻️ مساعد فرز النفايات الذكي")
    st.write(
        "تطبيق تجريبي يساعدك في فرز النفايات وتقليل التلوث والتغير المناخي "
        "باستخدام Teachable Machine لتصنيف الصور وGoogle Gemini للشاتبوت."
    )

    # التبويبات الرئيسية
    tab1, tab2, tab3 = st.tabs(
        ["📸 تصنيف النفايات", "💬 شاتبوت إعادة التدوير", "ℹ️ عن المشروع"]
    )

    # ------------------------------------------------------
    # تبويب 1: تصنيف من صورة
    # ------------------------------------------------------
    with tab1:
        st.header("📸 ارفعي صورة للنفاية")

        col_upload, col_info = st.columns([1.2, 1.0])

        with col_upload:
            uploaded_file = st.file_uploader(
                "ارفع صورة (JPG أو PNG):",
                type=["jpg", "jpeg", "png"]
            )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="الصورة المدخلة", use_column_width=True)

            with col2:
                waste_type, explanation = classify_waste_teachable_machine(image)
                if waste_type == "غير معروف":
                    st.error(waste_type)
                    st.info(explanation)
                else:
                    st.success(f"نوع النفاية حسب النموذج: **{waste_type}**")
                    st.info(explanation)
                    st.session_state["last_waste_type"] = waste_type

        else:
            with col_info:
                st.markdown(
                    """
                    <div class="info-card">
                    🔍 <b>طريقة العمل</b><br>
                    - ارفعي صورة لزجاجة، ورقة، عبوة بلاستيكية، أو بقايا طعام... حسب ما دربتي النموذج.<br>
                    - سيقوم نموذج <b>Teachable Machine</b> بتصنيفها وإعطائك التسمية المناسبة.<br>
                    - التطبيق يعرض لك نصائح للفرز للمساعدة في حماية البيئة.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ------------------------------------------------------
    # تبويب 2: شاتبوت إعادة التدوير
    # ------------------------------------------------------
    with tab2:
        st.header("💬 اسأل عن إعادة التدوير")

        if st.session_state["last_waste_type"]:
            st.write(
                f"🔎 آخر نوع نفايات تم التعرف عليه: "
                f"**{st.session_state['last_waste_type']}**"
            )

        with st.expander("أمثلة على أسئلة يمكنك طرحها"):
            st.markdown(
                "- كيف أعيد تدوير البلاستيك؟\n"
                "- ماذا أفعل بالنفايات العضوية؟\n"
                "- ما علاقة إعادة التدوير بالتغير المناخي؟\n"
                "- أعطني أفكارًا لإعادة استخدام الزجاج أو الكرتون."
            )

        user_msg = st.text_input("✏️ اكتب سؤالك هنا:")

        send_col, _ = st.columns([1, 3])
        with send_col:
            send_clicked = st.button("إرسال السؤال")

        if send_clicked:
            if user_msg.strip() == "":
                st.warning("من فضلك اكتب سؤالاً أولاً.")
            else:
                reply = recycling_chatbot_ai(
                    user_msg,
                    st.session_state["last_waste_type"]
                )
                # حفظ في تاريخ المحادثة
                st.session_state["chat_history"].append(("أنت", user_msg))
                st.session_state["chat_history"].append(("المساعد", reply))

        # عرض المحادثة
        if st.session_state["chat_history"]:
            st.subheader("المحادثة")
            for sender, text in st.session_state["chat_history"]:
                if sender == "أنت":
                    st.markdown(f"**🧑‍🎓 {sender}:** {text}")
                else:
                    st.markdown(f"**🤖 {sender}:** {text}")

    # ------------------------------------------------------
    # تبويب 3: عن المشروع
    # ------------------------------------------------------
    with tab3:
        st.header("ℹ️ عن المشروع")

        st.subheader("المشكلة البيئية")
        st.write(
            "يعاني العالم من زيادة كبيرة في النفايات غير المفرزة، مما يؤدي إلى حرق وطمر كميات ضخمة "
            "من النفايات، وهذا يسبب انبعاث غازات دفيئة مثل ثاني أكسيد الكربون (CO₂) والميثان (CH₄)، "
            "ويزيد من آثار التغير المناخي."
        )

        st.subheader("فكرة الحل")
        st.write(
            "هذا التطبيق يقترح حلاً يعتمد على الذكاء الاصطناعي لمساعدة الأفراد على فرز النفايات "
            "من خلال التعرف على نوع النفايات من الصورة باستخدام Teachable Machine، "
            "وتقديم نصائح لإعادة التدوير وإعادة الاستخدام بمساعدة شاتبوت ذكي."
        )

        st.subheader("التقنيات المستخدمة")
        st.markdown(
            "- لغة البرمجة: **Python**  \n"
            "- مكتبة الواجهات: **Streamlit**  \n"
            "- بيئة التنفيذ: **Google Colab**  \n"
            "- تصنيف الصور: **Teachable Machine (TensorFlow keras_model.h5)**  \n"
            "- الشاتبوت: **Google Gemini (google-generativeai)**  \n"
            "- (يمكن مستقبلاً إضافة لوحة إحصائيات وتأثيرات مناخية)"
        )

        st.subheader("الارتباط بالتغير المناخي")
        st.write(
            "من خلال تحسين فرز النفايات، نقلل الكمية التي تُحرق أو تُدفن، وبالتالي نقلل انبعاث الغازات "
            "المسببة للاحتباس الحراري، مما يساهم في حماية البيئة والحد من التغير المناخي."
        )

        st.subheader("أفكار للتطوير المستقبلي")
        st.markdown(
            "- تحسين نموذج Teachable Machine ببيانات أكثر تنوعًا.  \n"
            "- إضافة عدّاد تقريبي لكمية الانبعاثات التي تم تجنبها بفضل الفرز الصحيح.  \n"
            "- ربط التطبيق بنظام نقاط ومكافآت عند الالتزام بإعادة التدوير.  \n"
            "- إضافة واجهة عرض للطلاب تشرح خطوات بناء النموذج واستخدامه."
        )

# تشغيل التطبيق
if __name__ == "__main__":
    main()
