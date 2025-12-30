import re
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st

# Initialize page
st.set_page_config(page_title="Quran Contest Report", layout="wide")

# local_hour = (datetime.now().hour - 8) % 24 
is_daytime = 6 <= local_hour < 18

HIGHLIGHT_BG = "#dfe8d2" if is_daytime else "#4b5a4a"
HIGHLIGHT_TEXT = "#2c2a26" if is_daytime else "#f4f1e8"
theme_css = f"""
<style>
  @import url("https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;600;700&display=swap");
  :root {{
    --app-bg: {"#efe6d7" if is_daytime else "#222821"};
    --app-bg-2: {"#ebe1d1" if is_daytime else "#2a3129"};
    --app-bg-3: {"#e5dacb" if is_daytime else "#1f2520"};
    --sidebar-bg: {"#e9f2e9" if is_daytime else "#1f2721"};
    --sidebar-bg-2: {"#f3f8f1" if is_daytime else "#242d26"};
    --sidebar-border: {"#d7e2d6" if is_daytime else "#2f3a30"};
    
    /* --text-color: {"#2a2723" if is_daytime else "#e9e6df"}; */
    
    /* --text-color: {"#000000" if is_daytime else "#e9e6df"};*/
    --text-color: {"#000000" if is_daytime else "#ffffff"};
    --uploader-bg: {"#f7f4ee" if is_daytime else "#1f2721"};
    --uploader-text: {"#2a2723" if is_daytime else "#e9e6df"};
    --input-bg: {"#f6f0e6" if is_daytime else "#2b322b"};
    --input-text: {"#2a2723" if is_daytime else "#e9e6df"};
    --button-bg: {"#ece4d6" if is_daytime else "#303830"};
    --button-text: {"#2a2723" if is_daytime else "#e9e6df"};
    --expander-text: {"#2a2723" if is_daytime else "#f3efe4"};
  }}
  .stApp {{
    background: radial-gradient(circle at 25% 8%, var(--app-bg) 0%, var(--app-bg-2) 50%, var(--app-bg-3) 100%);
    color: var(--text-color);
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }}
  .stApp p,
  .stApp label,
  .stApp .stMarkdown,
  .stApp .stCaption,
  .stApp .stText,
  .stApp .stSubheader,
  .stApp .stHeader,
  .stApp .stTitle {{
    color: var(--text-color) !important;
  }}
  .stApp h1, .stApp h2, .stApp h3 {{
    font-family: "Playfair Display", "Georgia", serif;
    letter-spacing: 0.2px;
  }}
  [data-testid="stHeader"] {{
    background: var(--app-bg) !important;
    border-bottom: 1px solid var(--sidebar-border) !important;
  }}
  [data-testid="stToolbar"] {{
    background: transparent !important;
  }}
  [data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg) !important;
    background-image: linear-gradient(180deg, var(--sidebar-bg-2) 0%, var(--sidebar-bg) 100%) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    color: var(--text-color) !important;
  }}
  [data-testid="stSidebar"] * {{
    color: var(--text-color) !important;
  }}
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] textarea,
  [data-testid="stSidebar"] select,
  [data-testid="stSidebar"] [data-baseweb="select"] div {{
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border-color: var(--sidebar-border) !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: var(--input-text) !important;
  }}
  [data-testid="stSidebar"] button {{
    background-color: var(--button-bg) !important;
    color: var(--button-text) !important;
    border-color: var(--sidebar-border) !important;
    border-radius: 10px !important;
  }}
  .hero-subtitle {{
    font-size: 1.05rem;
    opacity: 0.85;
    margin-top: -0.4rem;
    margin-bottom: 1.2rem;
  }}
  .card {{
    background: rgba(255, 255, 255, 0.24);
    border: 1px solid var(--sidebar-border);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 16px rgba(19, 21, 18, 0.12);
  }}
  [data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(19, 21, 18, 0.12);
  }}
  [data-testid="stTabs"] {{
    margin-top: 0.5rem;
  }}
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    padding: 6px;
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
    font-weight: 600;
  }}
  [data-testid="stExpander"] {{
    background-color: transparent !important;
  }}
  /* Collapsed expander header: dark text on light bar */
  [data-testid="stExpander"] summary {{
    background: #e9dece !important;
    border-radius: 8px !important;
  }}
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary * {{
    color: #2a2723 !important;
  }}
  /* Expanded expander header: white text on dark bar */
  [data-testid="stExpander"] details[open] > summary {{
    background: #1b2028 !important;
  }}
  [data-testid="stExpander"] details[open] > summary,
  [data-testid="stExpander"] details[open] > summary * {{
    color: #ffffff !important;
  }}
  [data-testid="stFileUploader"],
  [data-testid="stFileUploader"] > div,
  [data-testid="stFileUploaderDropzone"] {{
    background-color: var(--uploader-bg) !important;
    color: var(--uploader-text) !important;
    border-color: var(--sidebar-border) !important;
  }}
  [data-testid="stFileUploader"] * {{
    color: var(--uploader-text) !important;
  }}

  /* Insert this right here before </style> */
  [data-testid="stMetricValue"] {{
    color: var(--text-color) !important;
  }}
  [data-testid="stMetricLabel"] p {{
    color: var(--text-color) !important;
  }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- Helper Functions ---

def map_gender(val: object) -> str:
    val = str(val).strip().upper()
    if val in {"M", "MALE", "BOY", "BOYS", "M"}: return "Boys"
    if val in {"F", "FEMALE", "GIRL", "GIRLS", "F"}: return "Girls"
    return "Unknown"

def infer_gender(teacher_name: object) -> str:
    t = str(teacher_name).lower() if pd.notna(teacher_name) else ""
    if any(k in t for k in ["sister", "sr.", "mrs", "ms", "duraiya", "durriyah", "female"]):
        return "Girls"
    return "Boys"

def get_sort_key(grade: object):
    g = str(grade)
    if "yrs" in g:
        num = re.search(r"(\d+)", g)
        return (0, int(num.group(1)) if num else 0, g)
    if "Grade" in g:
        num = re.search(r"(\d+)", g)
        return (1, int(num.group(1)) if num else 0, g)
    if "Surat" in g: return (2, 0, g)
    if "Juz" in g: return (3, 0, g)
    return (4, 0, g)

def should_split(grade: str) -> bool:
    if "yrs" in grade: return False
    grade_match = re.search(r"Grade\s+(\d+)", grade)
    if grade_match and int(grade_match.group(1)) <= 4: return False
    return True

# --- Main Data Extraction Logic (Updated for New Tracking File) ---

def extract_participants(
    df: pd.DataFrame, location_col: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    participants = []
    missing_gender_names = []
    def normalize_col(name: object) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"\\s*\\.\\s*", ".", s)
        s = re.sub(r"\\s+", " ", s)
        return s

    normalized_cols = {normalize_col(c): c for c in df.columns}
    available_gender_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"gender(\.\d+)?", normalize_col(c))],
        key=lambda c: int(m.group(1)) if (m := re.search(r"gender\.(\d+)", normalize_col(c))) else 0,
    )
    last_name_cols = [
        c
        for c in df.columns
        if re.search(r"name \\(last\\)$", normalize_col(c))
    ]

    def find_col(*candidates: str) -> str | None:
        for cand in candidates:
            key = normalize_col(cand)
            if key in normalized_cols:
                return normalized_cols[key]
        return None

    def gender_col_from_child(child_col: object) -> str | None:
        key = normalize_col(child_col)
        m = re.match(r"^(\\d+)_child contest options$", key)
        if m:
            return find_col(f"{m.group(1)}_Gender")
        m = re.match(r"^(\\d+)(st|nd|rd|th) child contest options$", key)
        if m:
            num = int(m.group(1))
            if num == 1:
                return find_col("Gender")
            # Some files use Gender.1 for 2nd, others use Gender.2 for 2nd, etc.
            return find_col(f"Gender.{num-1}") or find_col(f"Gender.{num}")
        return None

    def gender_value_from_child(row: pd.Series, child_col: object) -> str | None:
        if not has_value(row.get(child_col)):
            return None
        key = normalize_col(child_col)
        num = None
        m = re.match(r"^(\\d+)_child contest options$", key)
        if m:
            num = int(m.group(1))
        m = re.match(r"^(\\d+)(st|nd|rd|th) child contest options$", key)
        if m:
            num = int(m.group(1))
        if not num:
            return None
        candidates = [f"{num}_Gender"]
        if num == 1:
            candidates += ["Gender", "Gender.1"]
        else:
            candidates += [f"Gender.{num-1}", f"Gender.{num}"]
        for cand in candidates:
            col = find_col(cand)
            if col in df.columns and has_value(row.get(col)):
                return map_gender(row.get(col))
        return None

    # Prefer unprefixed Name (First)/(Last) columns when present.
    prefixes = sorted(
        {
            int(m.group(1))
            for c in df.columns
            if (m := re.match(r"^(\\d+)_Name \\(First\\)$", str(c)))
        }
    )
    has_unprefixed = any(
        re.fullmatch(r"name \\(first\\)(\\.\\d+)?", normalize_col(c))
        for c in df.columns
    )
    slots = []
    if has_unprefixed:
        suffixes = []
        for c in df.columns:
            m = re.search(r"name \\(first\\)(?:\\.(\\d+))?$", normalize_col(c))
            if m:
                suffixes.append(int(m.group(1)) if m.group(1) else 0)
        max_idx = max(suffixes) if suffixes else 0
        ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
        for idx in range(0, max_idx + 1):
            ord_label = ordinals[idx] if idx < len(ordinals) else f"{idx+1}th"
            first = "Name (First)" if idx == 0 else f"Name (First).{idx}"
            last = "Name (Last)" if idx == 0 else f"Name (Last).{idx}"
            child_cat = f"{ord_label} Child Contest Options"
            gender = "Gender" if idx == 0 else f"Gender.{idx}"
            slots.append(
                {
                    "first": find_col(first) or first,
                    "last": find_col(last) or last,
                    "child_cat": find_col(child_cat) or child_cat,
                    "adult_cat": find_col("Adult Contest Options"),
                    "gender": find_col(gender) or gender,
                }
            )
    elif prefixes:
        for p in prefixes:
            slots.append(
                {
                    "first": f"{p}_Name (First)",
                    "last": f"{p}_Name (Last)",
                    "child_cat": f"{p}_Child Contest Options",
                    "adult_cat": f"{p}_Adult Contest Options",
                    "gender": f"{p}_Gender",
                }
            )
    else:
        ordinals = ["1st", "2nd", "3rd", "4th", "5th"]
        for idx, ord_label in enumerate(ordinals):
            first = "Name (First)" if idx == 0 else f"Name (First).{idx}"
            last = "Name (Last)" if idx == 0 else f"Name (Last).{idx}"
            child_cat = f"{ord_label} Child Contest Options"
            gender = "Gender" if idx == 0 else f"Gender.{idx}"
            slots.append(
                {
                    "first": find_col(first) or first,
                    "last": find_col(last) or last,
                    "child_cat": find_col(child_cat) or child_cat,
                    "adult_cat": find_col("Adult Contest Options"),
                    "gender": find_col(gender) or gender,
                }
            )
    adult_cat_col = "Adult Contest Options"

    for idx, slot in enumerate(slots):
        for _, row in df.iterrows():
            f_name = row.get(slot["first"])
            if pd.isna(f_name) or str(f_name).strip() == "":
                continue

            def has_value(val: object) -> bool:
                return pd.notna(val) and str(val).strip() != ""

            l_name = row.get(slot["last"], "")
            if not has_value(l_name):
                last_candidates = [
                    v for v in (row.get(c) for c in last_name_cols) if has_value(v)
                ]
                if len(last_candidates) == 1:
                    l_name = last_candidates[0]
            email = row.get("Email", "N/A")

            # Check Child options for this specific slot
            c_cat = row.get(slot["child_cat"])
            # Adult options are usually only in the primary slot (Slot 1)
            a_cat = row.get(slot.get("adult_cat")) if slot.get("adult_cat") else (row.get(adult_cat_col) if idx == 0 else None)

            c_has = has_value(c_cat)
            a_has = has_value(a_cat)
            has_entry = c_has or a_has
            if not has_entry:
                continue
                
            cat = c_cat if c_has else a_cat
            entry_type = "Child" if c_has else "Adult"

            # Only check gender for actual participant entries with full names and a real gender column.
            has_full_name = str(f_name).strip() != "" and str(l_name).strip() != ""
            gender_col = slot.get("gender")
            if not gender_col or gender_col not in df.columns:
                gender_col = find_col("Gender" if idx == 0 else f"Gender.{idx}")
            gender_col_exists = bool(gender_col) and gender_col in df.columns
            gender_raw = row.get(gender_col) if gender_col_exists else None
            if has_entry and gender_col_exists and has_value(gender_raw):
                gender = map_gender(gender_raw)
            else:
                gender = "Unknown"
                if has_entry and has_full_name and gender_col_exists:
                    missing_gender_names.append(f"{f_name} {l_name}".strip())

            cat_str = str(cat).replace("|0", "").strip()
            
            # Location extraction
            location_val = "Unknown"
            if location_col:
                location_val = row.get(location_col, "Unknown")
                if pd.isna(location_val) or str(location_val).strip() == "":
                    location_val = "Unknown"
            
            participants.append({
                "First Name": f_name, "Last Name": l_name, "Grade": cat_str,
                "Email": email, "Gender": gender, "Location": location_val, "Entry Type": entry_type,
            })

    return pd.DataFrame(participants), missing_gender_names

# --- Report and UI Logic ---

def build_reports(results_df: pd.DataFrame):
    tables = {}
    counts = (
        results_df.groupby("Grade").size().rename("Count")
        .sort_index(key=lambda idx: idx.map(get_sort_key))
    )
    counts_df = counts.reset_index().rename(columns={"Grade": "Category"})
    counts_df.loc[len(counts_df)] = ["Total", int(counts_df["Count"].sum())]

    unique_grades = sorted(results_df["Grade"].unique(), key=get_sort_key)
    for grade in unique_grades:
        subset = results_df[results_df["Grade"] == grade].copy()
        if not should_split(grade):
            # Combined Logic
            child_sub = subset[subset["Entry Type"] != "Adult"].copy()
            adult_sub = subset[subset["Entry Type"] == "Adult"].copy()

            if not child_sub.empty:
                child_sub.insert(0, "Count", range(1, len(child_sub) + 1))
                tables[grade] = child_sub[["Count", "First Name", "Last Name", "Email"]]
            if not adult_sub.empty:
                adult_sub.insert(0, "Count", range(1, len(adult_sub) + 1))
                tables[f"{grade} - ADULT"] = adult_sub[["Count", "First Name", "Last Name", "Email"]]
        else:
            # Gender Split Logic
            for gender in ["Boys", "Girls"]:
                gen_sub = subset[subset["Gender"] == gender].copy()
                if gen_sub.empty: continue
                
                child_sub = gen_sub[gen_sub["Entry Type"] != "Adult"].copy()
                adult_sub = gen_sub[gen_sub["Entry Type"] == "Adult"].copy()

                if not child_sub.empty:
                    child_sub.insert(0, "Count", range(1, len(child_sub) + 1))
                    tables[f"{grade} - {gender.upper()}"] = child_sub[["Count", "First Name", "Last Name", "Email"]]
                if not adult_sub.empty:
                    adult_sub.insert(0, "Count", range(1, len(adult_sub) + 1))
                    tables[f"{grade} - {gender.upper()} - ADULT"] = adult_sub[["Count", "First Name", "Last Name", "Email"]]

    tables["Category_Counts"] = counts_df
    return tables

def tables_to_excel_bytes(tables: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 24, "font_color": "#1F4E78", "align": "left"})
        used_names = set()

        def safe_name(title):
            name = re.sub(r"[:*?/\\\[\]]", "", title)[:31]
            if name not in used_names:
                used_names.add(name)
                return name
            i = 1
            while f"{name[:28]}_{i}" in used_names: i += 1
            res = f"{name[:28]}_{i}"
            used_names.add(res)
            return res

        for sheet_title, data in tables.items():
            name = safe_name(sheet_title)
            is_count = sheet_title == "Category_Counts"
            export_data = add_checkin_column(data) if not is_count else data
            export_data.to_excel(writer, sheet_name=name, startrow=3 if not is_count else 0, index=False)
            if not is_count:
                ws = writer.sheets[name]
                ws.write(0, 0, sheet_title, title_fmt)
                ws.set_column("B:F", 25)
    return output.getvalue()

def tables_to_single_sheet_bytes(tables: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        from xlsxwriter.utility import xl_rowcol_to_cell
        title_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#1F4E78", "align": "left"})
        sheet_name = "All Categories"
        row_cursor = 0
        ws = None
        for sheet_title, data in tables.items():
            if sheet_title == "Category_Counts":
                continue
            export_data = data.copy()
            if "Email" in export_data.columns:
                export_data = export_data.drop(columns=["Email"])
            export_data["Admitted"] = ""
            export_data["Seated"] = ""
            export_data["Tested"] = ""
            export_data.to_excel(writer, sheet_name=sheet_name, startrow=row_cursor + 1, index=False)
            ws = writer.sheets[sheet_name]
            ws.write(row_cursor, 0, sheet_title, title_fmt)
            door_col = export_data.columns.get_loc("Admitted")
            seated_col = export_data.columns.get_loc("Seated")
            test_col = export_data.columns.get_loc("Tested")
            data_start = row_cursor + 2
            for idx in range(len(export_data)):
                row = data_start + idx
                ws.insert_checkbox(row, door_col, False)
                ws.insert_checkbox(row, seated_col, False)
                ws.insert_checkbox(row, test_col, False)
            row_cursor += len(export_data) + 3
        if ws:
            ws.set_column("A:G", 25)
    return output.getvalue()

def labels_to_pdf_bytes(df: pd.DataFrame) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin_x = 36
    margin_y = 36
    cols = 3
    rows = 10
    gap_x = 8
    gap_y = 10
    label_w = (width - 2 * margin_x - (cols - 1) * gap_x) / cols
    label_h = (height - 2 * margin_y - (rows - 1) * gap_y) / rows

    def wrap_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines = []
        line = []
        for w in words:
            if sum(len(x) for x in line) + len(line) + len(w) <= max_chars:
                line.append(w)
            else:
                lines.append(" ".join(line))
                line = [w]
        if line:
            lines.append(" ".join(line))
        return lines or [""]

    x = margin_x
    y = height - margin_y - label_h
    c.setFont("Helvetica", 10)

    for idx, row in df.iterrows():
        full_name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
        category = str(row.get("Grade", "")).strip()

        text = c.beginText(x + 8, y + label_h - 16)
        text.textLine(full_name)
        for line in wrap_text(category, 26)[:2]:
            text.textLine(line)
        c.drawText(text)

        col = (idx + 1) % cols
        if col == 0:
            x = margin_x
            y -= label_h + gap_y
            if y < margin_y:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin_y - label_h
        else:
            x += label_w + gap_x

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def filter_matches(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().lower()
    if not q: return df
    mask = df.apply(lambda row: q in " ".join(str(row[c]).lower() for c in ["First Name", "Last Name"] if c in df.columns), axis=1)
    return df[mask]

def highlight_matches(df: pd.DataFrame, query: str):
    q = query.strip().lower()
    if not q: return df
    mask = df.apply(lambda row: q in " ".join(str(row[c]).lower() for c in ["First Name", "Last Name"] if c in df.columns), axis=1)
    def style_row(row):
        if not mask.loc[row.name]:
            return [""] * len(row)
        return [f"background-color: {HIGHLIGHT_BG}; color: {HIGHLIGHT_TEXT}"] * len(row)
    return df.style.apply(style_row, axis=1)

def add_checkin_column(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    if "Checked In" not in view.columns:
        view["Checked In"] = False
    return view

# --- UI Layout ---

st.image("mac-logo350.png", width=200)
st.title("Quran Contest Report Builder")
st.markdown(
    '<div class="hero-subtitle">Upload the registration Excel file to generate clean tables and summary charts.</div>',
    unsafe_allow_html=True,
)
uploaded = st.sidebar.file_uploader("Upload tracking file", type=["xlsx", "xls", "csv"])

if not uploaded:
    st.info("Upload a file to get started.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# Auto-detect location column
loc_cands = [c for c in raw_df.columns if re.search(r"(city|location|address)", str(c), re.I)]
selected_loc_col = None
if loc_cands:
    default_loc_idx = 0
    for i, c in enumerate(loc_cands):
        if re.search(r"city", str(c), re.I):
            default_loc_idx = i
            break
    selected_loc_col = loc_cands[default_loc_idx]
results_df, missing_gender_names = extract_participants(raw_df, location_col=selected_loc_col)
dup_subset = ["First Name", "Last Name", "Grade", "Email", "Gender"]
dup_mask = results_df.duplicated(subset=dup_subset, keep=False)
duplicates_df = (
    results_df[dup_mask]
    .sort_values(dup_subset)
    .reset_index(drop=True)
)
results_df = results_df.drop_duplicates(subset=dup_subset).reset_index(drop=True)

if results_df.empty:
    st.warning("No data extracted. Verify your headers.")
    st.stop()
if missing_gender_names:
    st.warning("Missing gender for: " + ", ".join(sorted(set(missing_gender_names))))

# Initialize filter state before using it
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = {"cats": [], "query": ""}

# KPI chips
st.markdown('<div class="card">', unsafe_allow_html=True)
total_count = len(results_df)

def normalize_col_name(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s*\.\s*", ".", s)
    s = re.sub(r"\s+", " ", s)
    return s

boys_count = int((results_df["Gender"] == "Boys").sum()) if "Gender" in results_df.columns else 0
girls_count = int((results_df["Gender"] == "Girls").sum()) if "Gender" in results_df.columns else 0

adult_count = int((results_df["Entry Type"] == "Adult").sum()) if "Entry Type" in results_df.columns else 0
child_count = int((results_df["Entry Type"] == "Child").sum()) if "Entry Type" in results_df.columns else 0
donation_total = 0.0
donation_col = "Total Donation (Price)"
if donation_col in raw_df.columns:
    donation_total = (
        raw_df[donation_col]
        .astype(str)
        .str.replace(r"[^0-9.\\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
        .sum()
    )
payment_total = 0.0
payment_col = "Payment Amount"
if payment_col in raw_df.columns:
    payment_total = (
        raw_df[payment_col]
        .astype(str)
        .str.replace(r"[^0-9.\\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
        .sum()
    )
reg_fees_total = total_count * 35
payment_diff = payment_total - (donation_total + reg_fees_total)
kpis = [
    ("Total", total_count),
    ("Boys", boys_count),
    ("Girls", girls_count),
    ("Adults", adult_count),
    ("Children", child_count),
    ("Donations", f"${donation_total:,.2f}"),
    ("Total Payment", f"${payment_total:,.2f}"),
]
top_kpis = kpis[:5]
bottom_kpis = [
    ("Donations", f"${donation_total:,.2f}"),
    ("Reg Fees", f"${reg_fees_total:,.2f}"),
    ("Total Payment", f"${payment_total:,.2f}"),
    ("Diff", f"${payment_diff:,.2f}"),
]
top_cols = st.columns(5)
for col, (label, value) in zip(top_cols, top_kpis):
    col.metric(label, value)
bottom_cols = st.columns(4)
for col, (label, value) in zip(bottom_cols, bottom_kpis):
    col.metric(label, value)
st.markdown("</div>", unsafe_allow_html=True)

# Filtering and Search
all_cats = sorted(results_df["Grade"].unique(), key=get_sort_key)
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = {"cats": [], "query": ""}
if "category_filter" not in st.session_state:
    st.session_state.category_filter = st.session_state.filters_applied["cats"]
if "name_filter" not in st.session_state:
    st.session_state.name_filter = st.session_state.filters_applied["query"]

def apply_category():
    st.session_state.filters_applied["cats"] = st.session_state.category_filter

def clear_category():
    st.session_state.category_filter = []
    st.session_state.filters_applied["cats"] = []

def apply_name():
    st.session_state.filters_applied["query"] = st.session_state.name_filter

def clear_name():
    st.session_state.name_filter = ""
    st.session_state.filters_applied["query"] = ""


sel_cats = st.sidebar.multiselect(
    "Filter by Category",
    options=all_cats,
    key="category_filter",
)
cat_btns = st.sidebar.columns(2)
cat_btns[0].button("Apply Category", on_click=apply_category)
cat_btns[1].button("Clear Category", on_click=clear_category)

search_query = st.sidebar.text_input(
    "Search name",
    key="name_filter",
)
name_btns = st.sidebar.columns(2)
name_btns[0].button("Apply Name", on_click=apply_name)
name_btns[1].button("Clear Name", on_click=clear_name)

filtered_results = results_df.copy()
applied = st.session_state.filters_applied
if applied["cats"]:
    filtered_results = filtered_results[filtered_results["Grade"].isin(applied["cats"])]

reports = build_reports(results_df)

# Display Tabs
tab_preview, tab_summary, tab_labels = st.tabs(["Preview Tables", "Summary Chart", "Label Preview"])

with tab_preview:
    st.subheader("All Participants")
    applied_query = applied["query"]
    view = filter_matches(filtered_results, applied_query)
    st.caption(f"Showing {len(view)} entries")
    st.dataframe(highlight_matches(view, applied_query), width="stretch")

    if not duplicates_df.empty:
        with st.expander("Duplicate entries (removed)"):
            st.dataframe(duplicates_df, width="stretch")

    for title, data in reports.items():
        with st.expander(title):
            v = filter_matches(data, applied_query)
            st.dataframe(highlight_matches(v, applied_query), width="stretch")

with tab_summary:
    st.subheader("Category Counts")
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
        bar_data = reports["Category_Counts"].iloc[:-1].copy()
        bar = (
            alt.Chart(bar_data)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Category:N",
                    sort="-x",
                    axis=alt.Axis(
                        labelFontSize=12,
                        labelLimit=260,
                        title="Category",
                    ),
                ),
                x=alt.X("Count:Q", axis=alt.Axis(title="Count")),
                tooltip=["Category", "Count"],
            )
        )
        st.altair_chart(bar, width="stretch")
    except Exception:
        st.bar_chart(reports["Category_Counts"].iloc[:-1], x="Category", y="Count")
    try:
        import altair as alt
        alt.data_transformers.disable_max_rows()
        pie_data = reports["Category_Counts"].iloc[:-1].copy()
        pie = (
            alt.Chart(pie_data)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Category",
                    type="nominal",
                    legend=alt.Legend(labelLimit=300, columns=2),
                ),
                tooltip=["Category", "Count"],
            )
        )
        st.altair_chart(pie, width="stretch")
    except Exception:
        st.info("Pie chart requires Altair. Install with `pip install altair`.")
    
    if selected_loc_col:
        st.subheader("City Distribution")
        city_series = (
            results_df["Location"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
        city_norm = city_series.str.casefold()
        city_counts = city_norm.value_counts().reset_index()
        city_counts.columns = ["City", "Count"]
        city_counts["City"] = city_counts["City"].str.title()
        try:
            import altair as alt
            city_pie = (
                alt.Chart(city_counts)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="City", type="nominal"),
                    tooltip=["City", "Count"],
                )
            )
            st.altair_chart(city_pie, width="stretch")
        except Exception:
            st.info("Pie chart requires Altair. Install with `pip install altair`.")
        st.dataframe(city_counts, width="stretch")
    else:
        st.info("No city/location column found in the uploaded file.")

with tab_labels:
    st.subheader("Label Preview")
    label_view = results_df[["First Name", "Last Name", "Grade"]].copy()
    label_view["Label"] = (
        label_view["First Name"].fillna("").astype(str).str.strip()
        + " "
        + label_view["Last Name"].fillna("").astype(str).str.strip()
    ).str.strip() + " | " + label_view["Grade"].fillna("").astype(str).str.strip()
    st.dataframe(label_view[["Label"]], width="stretch")

# Download Buttons
st.sidebar.download_button(
    "Download Excel Report",
    data=tables_to_excel_bytes(reports),
    file_name=f"Quran_Contest_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.sidebar.download_button(
    "Download All Tabs (Single Sheet)",
    data=tables_to_single_sheet_bytes(reports),
    file_name=f"Quran_Contest_All_Categories_{datetime.now().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
filtered_reports = build_reports(filtered_results)
applied_cats = st.session_state.filters_applied.get("cats", [])
applied_query = st.session_state.filters_applied.get("query", "").strip()
filter_bits = []
if applied_cats:
    filter_bits.append("_".join(str(c) for c in applied_cats))
if applied_query:
    filter_bits.append(f"Name_{applied_query}")
filter_label = "_".join(filter_bits) if filter_bits else "All"
filter_label = re.sub(r"[^A-Za-z0-9_-]+", "_", filter_label).strip("_")
st.sidebar.download_button(
    "Download Filtered Report",
    data=tables_to_excel_bytes(filtered_reports),
    file_name=f"Quran_Contest_Report_{filter_label}_{datetime.now().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
try:
    st.sidebar.download_button(
        "Download Labels (PDF)",
        data=labels_to_pdf_bytes(results_df),
        file_name=f"Quran_Contest_Labels_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )
except Exception:
    st.sidebar.info("Label PDF requires `reportlab`. Install with `pip install reportlab`.")






# . .venv/bin/activate
# streamlit run streamlit_app.py
