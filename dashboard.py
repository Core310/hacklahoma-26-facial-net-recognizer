import streamlit as st
import pymongo
import base64
import numpy as np

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_db"
COLLECTION_NAME = "identities"

st.set_page_config(page_title="Face Database Manager", layout="wide")
st.title("🤖 Face Recognition Manager")


# Connect to DB
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI)


client = init_connection()
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- Sidebar Actions ---
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.error("Danger Zone")
if st.sidebar.button("🗑️ DELETE ALL DATA"):
    collection.drop()
    st.warning("Database Wiped!")
    st.rerun()

# --- Main Interface ---
# Fetch all faces
faces = list(collection.find())

if not faces:
    st.info("Database is empty. Run main.py and register some faces!")
else:
    st.write(f"Found {len(faces)} identities in database.")

    # Display in a grid
    cols = st.columns(4)  # 4 items per row

    for idx, face in enumerate(faces):
        with cols[idx % 4]:
            # Container for each face
            with st.container(border=True):
                # 1. Show Image
                if "thumbnail" in face and face['thumbnail']:
                    try:
                        img_bytes = base64.b64decode(face['thumbnail'])
                        st.image(img_bytes, width=100)
                    except:
                        st.text("Bad Image")
                else:
                    st.text("No Image")

                # 2. Edit Name
                current_name = face['name']
                # Create a unique key for each input box
                new_name = st.text_input(f"ID: {str(face['_id'])[-5:]}", value=current_name, key=f"name_{idx}")

                # 3. Save Button
                if new_name != current_name:
                    collection.update_one({"_id": face["_id"]}, {"$set": {"name": new_name}})
                    st.toast(f"Renamed to {new_name}!")
                    # We don't rerun immediately to let you edit others,
                    # but next refresh will show it.

                # 4. Delete Individual
                st.caption(f"Age: {face.get('age', '?')} | {face.get('gender', '?')}")

                if st.button("Delete", key=f"del_{idx}"):
                    collection.delete_one({"_id": face["_id"]})
                    st.rerun()

# --- Batch Merge Tool ---
st.divider()
st.header("Batch Merge Tool")
st.write("Select multiple IDs to merge them into a single person.")

# Create list for dropdown: "Name (ID: ...)"
# We store the FULL _id in a hidden way to make matching robust
id_map = {f"{f['name']} (ID: {str(f['_id'])[-5:]})": f['_id'] for f in faces}
options = list(id_map.keys())

selected_labels = st.multiselect("Select duplicates:", options)
target_name = st.text_input("Merge all selected into this Name:", placeholder="e.g. Arika")

if st.button("Merge All Selected"):
    if not target_name:
        st.error("Please enter a target name.")
    else:
        count = 0
        for label in selected_labels:
            real_id = id_map[label]
            collection.update_one({"_id": real_id}, {"$set": {"name": target_name}})
            count += 1

        st.success(f"Merged {count} faces into '{target_name}'!")
        st.rerun()