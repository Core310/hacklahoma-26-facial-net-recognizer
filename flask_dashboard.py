import streamlit as st
import pymongo
import base64
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import defaultdict

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_db"
COLLECTION_NAME = "identities"
# UPDATE THIS with your IP! (e.g. 10.204.191.90 or Tailscale URL)
STREAM_URL = "http://10.204.191.90:5001/video_feed"

st.set_page_config(page_title="Face Database Manager", layout="wide")
st.title("🤖 Face Recognition Manager")


# Connect to DB
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI)


# Initialize AI for Uploads (Runs on CPU)
@st.cache_resource
def init_app():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


client = init_connection()
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
app = init_app()

# --- LIVE CAMERA VIEW ---
st.header("Live Camera Feed")
st.markdown(
    f'<img src="{STREAM_URL}" width="100%" style="border-radius: 10px; border: 2px solid #333;" />',
    unsafe_allow_html=True
)
st.divider()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📤 Upload & Enroll")

# 1. Mode Selection
upload_mode = st.sidebar.radio(
    "Naming Strategy:",
    ("Single Identity (All same person)",
     "Use Filenames",
     "Auto-Increment")
)

uploaded_files = st.sidebar.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# 2. Dynamic Input
target_name = ""
linkedin_input = ""

if upload_mode.startswith("Single"):
    target_name = st.sidebar.text_input("Name:", placeholder="e.g. Arika")
    linkedin_input = st.sidebar.text_input("Link (Optional):", placeholder="https://linkedin.com/in/...")

if st.sidebar.button("Process & Upload"):
    if not uploaded_files:
        st.sidebar.error("Please upload images.")
    elif upload_mode.startswith("Single") and not target_name:
        st.sidebar.error("Please enter a name.")
    else:
        progress = st.sidebar.progress(0)
        success_count = 0
        total = len(uploaded_files)

        # Auto-Increment Logic
        next_id = 1
        if upload_mode.startswith("Auto"):
            for doc in collection.find():
                if doc['name'].startswith("Person_"):
                    try:
                        curr = int(doc['name'].split("_")[1])
                        if curr >= next_id: next_id = curr + 1
                    except:
                        pass

        for idx, up_file in enumerate(uploaded_files):
            # Decode
            file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            faces = app.get(img)

            if len(faces) == 1:
                face = faces[0]

                # Naming Logic
                f_name = "Unknown"
                f_link = ""

                if upload_mode.startswith("Single"):
                    f_name = target_name
                    f_link = linkedin_input
                elif upload_mode.startswith("Use"):
                    f_name = os.path.splitext(up_file.name)[0]
                elif upload_mode.startswith("Auto"):
                    f_name = f"Person_{next_id}"
                    next_id += 1

                # Thumbnail
                box = face.bbox.astype(int)
                h, w, _ = img.shape
                pad = 20
                x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
                x2, y2 = min(w, box[2] + pad), min(h, box[3] + pad)
                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    crop = cv2.resize(crop, (100, 100))
                    _, buf = cv2.imencode('.jpg', crop)
                    b64_str = base64.b64encode(buf).decode('utf-8')

                    collection.insert_one({
                        "name": f_name,
                        "embedding": face.embedding.tolist(),
                        "age": int(face.age),
                        "gender": "M" if face.gender == 1 else "F",
                        "thumbnail": b64_str,
                        "linkedin": f_link,
                        "created_at": "batch"
                    })
                    success_count += 1

            progress.progress((idx + 1) / total)
        st.sidebar.success(f"Added {success_count} faces!")
        st.rerun()

st.sidebar.divider()

# --- MERGE TOOL ---
st.sidebar.header("🛠️ Merge Tool")
all_docs = list(collection.find())
unique_names = sorted(list(set(d['name'] for d in all_docs)))

m_targets = st.sidebar.multiselect("Select identities:", unique_names)
m_name = st.sidebar.text_input("Merge into Name:", placeholder="e.g. Arika")
m_link = st.sidebar.text_input("Merge Link (Optional):")

if st.sidebar.button("Merge Selected"):
    if not m_name or not m_targets:
        st.sidebar.error("Select names/target!")
    else:
        update = {"name": m_name}
        if m_link: update["linkedin"] = m_link

        res = collection.update_many(
            {"name": {"$in": m_targets}},
            {"$set": update}
        )
        st.sidebar.success(f"Merged {res.modified_count} docs!")
        st.rerun()

st.sidebar.divider()

# --- DELETE / REFRESH ---
col_ref, col_del = st.sidebar.columns(2)
if col_ref.button("🔄 Refresh"):
    st.rerun()

if col_del.button("🗑️ CLEAR ALL"):
    collection.drop()
    st.warning("Database Wiped!")
    st.rerun()

# ==========================================
# MAIN GALLERY
# ==========================================
st.header("Identities Gallery")

if not all_docs:
    st.info("Database is empty.")
else:
    # Group by Name
    grouped = defaultdict(list)
    for doc in all_docs:
        grouped[doc['name']].append(doc)

    sorted_names = sorted(grouped.keys())

    for name in sorted_names:
        vectors = grouped[name]

        # Find first non-empty link in this group
        group_link = next((d.get('linkedin') for d in vectors if d.get('linkedin')), "")

        with st.expander(f"👤 **{name}** ({len(vectors)} vectors)", expanded=True):

            # Show Link at top of group
            if group_link:
                st.markdown(f"🔗 **Link:** [{group_link}]({group_link})")

            # Photos Grid
            cols = st.columns(6)
            for idx, doc in enumerate(vectors):
                with cols[idx % 6]:
                    if "thumbnail" in doc and doc['thumbnail']:
                        try:
                            bts = base64.b64decode(doc['thumbnail'])
                            st.image(bts, width=80)
                        except:
                            st.text("Err")
                    else:
                        st.caption("No Img")

                    st.caption(f"{doc.get('age', '?')}yo")

                    # Delete Single Vector
                    if st.button("x", key=f"del_{doc['_id']}"):
                        collection.delete_one({"_id": doc["_id"]})
                        st.rerun()

            st.divider()

            # Edit Group Info
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                new_n = st.text_input("Rename:", value=name, key=f"ren_{name}")
            with c2:
                new_l = st.text_input("Edit Link:", value=group_link, key=f"lnk_{name}")
            with c3:
                if st.button("Update", key=f"upd_{name}"):
                    collection.update_many(
                        {"name": name},
                        {"$set": {"name": new_n, "linkedin": new_l}}
                    )
                    st.rerun()