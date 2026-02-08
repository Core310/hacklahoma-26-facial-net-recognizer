import streamlit as st
import pymongo
import base64
from collections import defaultdict

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

# --- Sidebar: Merge Tool ---
st.sidebar.header("🛠️ Merge Tool")
st.sidebar.info("Combine multiple 'Person_X' entries into one Identity.")

# Fetch all unique names
all_docs = list(collection.find())
unique_names = sorted(list(set(d['name'] for d in all_docs)))

# Multi-Select for Merge
merge_targets = st.sidebar.multiselect("Select identities to merge:", unique_names)
target_name = st.sidebar.text_input("New Name for all:", placeholder="e.g. Arika")

if st.sidebar.button("Merge Selected"):
    if not target_name or not merge_targets:
        st.sidebar.error("Select names and enter a target!")
    else:
        # Update MongoDB
        result = collection.update_many(
            {"name": {"$in": merge_targets}},
            {"$set": {"name": target_name}}
        )
        st.sidebar.success(f"✅ Merged {result.modified_count} vectors into '{target_name}'!")
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

if st.sidebar.button("🗑️ DELETE ALL DATA"):
    collection.drop()
    st.warning("Database Wiped!")
    st.rerun()

# --- Main Interface: Grouped View ---
st.header("Identities Gallery")

if not all_docs:
    st.info("Database is empty. Run main.py and look at the camera!")
else:
    # Group docs by Name
    grouped = defaultdict(list)
    for doc in all_docs:
        grouped[doc['name']].append(doc)

    # Sort groups by name (Arika first, Person_X last)
    sorted_names = sorted(grouped.keys())

    for name in sorted_names:
        vectors = grouped[name]

        # Identity Header
        with st.expander(f"👤 **{name}** ({len(vectors)} vectors/photos)", expanded=True):

            # Show Gallery of Faces for this Person
            cols = st.columns(6)  # 6 images per row
            for idx, doc in enumerate(vectors):
                with cols[idx % 6]:
                    # Display Image
                    if "thumbnail" in doc and doc['thumbnail']:
                        try:
                            img_bytes = base64.b64decode(doc['thumbnail'])
                            st.image(img_bytes, width=80)
                        except:
                            st.text("Err")
                    else:
                        st.caption("No Img")

                    # Individual Stats
                    st.caption(f"{doc.get('gender', '?')} | {doc.get('age', '?')}yo")

                    # Delete Single Vector button
                    if st.button("x", key=f"del_{doc['_id']}", help="Delete this specific photo/vector"):
                        collection.delete_one({"_id": doc["_id"]})
                        st.rerun()

            # Rename Entire Group
            col1, col2 = st.columns([3, 1])
            with col1:
                new_group_name = st.text_input("Rename this Identity:", value=name, key=f"ren_{name}")
            with col2:
                if st.button("Update Name", key=f"btn_{name}"):
                    collection.update_many({"name": name}, {"$set": {"name": new_group_name}})
                    st.success("Renamed!")
                    st.rerun()