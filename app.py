import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
import requests
import json
import time
import numpy as np
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Konfigurasi Sistem
NEO4J_URI      = st.secrets["NEO4J_URI"]
NEO4J_USER     = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
GROQ_API_KEY   = st.secrets["GROQ_API_KEY"]
GROQ_URL       = 'https://api.groq.com/openai/v1/chat/completions'
MODEL_LLM      = 'llama-3.1-8b-instant'
EMBEDDING_MODEL= 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

TOP_K_RETRIEVAL   = 20
TOP_K_KONTEKS_LLM = 10

# Konfigurasi Tampilan Halaman (UI)
st.set_page_config(page_title="GraphRAG PTA UTM", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp{background:#f4f7f6}
  .stChatMessage{border-radius:15px;padding:10px;margin-bottom:10px;
                  box-shadow:0 2px 5px rgba(0,0,0,.05)}
  .stChatMessage.user{background:#e3f2fd}
  .stChatMessage.assistant{background:#fff;border:1px solid #e0e0e0}
  .streamlit-expanderHeader{font-weight:bold;color:#1565c0;border-radius:5px}
  .block-container{padding-top:2rem;padding-bottom:5rem}
  h1,h2,h3{color:#2c3e50;font-family:'Segoe UI',sans-serif}
</style>
""", unsafe_allow_html=True)

# Inisialisasi Model dan Database
@st.cache_resource
def init_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=200,
        connection_acquisition_timeout=60)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')



try:
    driver      = init_neo4j_driver()
    embed_model = load_embedding_model()
except Exception as e:
    st.error(f"Gagal memuat sistem: {str(e)}")
    st.stop()

# Fungsi Bantuan untuk Eksekusi Query Neo4j
def _retry(driver, fn, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (SessionExpired, ServiceUnavailable):
            if attempt < max_retries:
                time.sleep(1); continue
            raise

def _run(driver, cypher, **params):
    def _ex():
        with driver.session() as s:
            return s.run(cypher, **params).data()
    try:
        return _retry(driver, _ex) or []
    except Exception:
        return []

def _run1(driver, cypher, **params):
    def _ex():
        with driver.session() as s:
            return s.run(cypher, **params).single()
    try:
        return _retry(driver, _ex)
    except Exception:
        return None


# Fungsi Bantuan Pemrosesan Teks & Gelar
_GELAR_RE = re.compile(
    r',?\s*(?:s\.t\.?|s\.kom\.?|m\.kom\.?|m\.t\.?|m\.mt\.?|m\.eng\.?|'
    r's\.si\.?|m\.sc\.?|dr\.?|prof\.?|ir\.?|s\.pd\.?|m\.pd\.?|'
    r'ph\.d\.?|s\.sos\.?|m\.sos\.?|apt\.?|drs\.?|dra\.?|'
    r'se\.?|sh\.?|sp\.?|m\.it\.?|dipl\.?)(?:\s*,\s*|(?=\s|$))',
    re.IGNORECASE)

# Prefix informal yang harus dibuang di AWAL nama
_PREFIX_INFORMAL_RE = re.compile(
    r'^(?:pak|bu|bapak|ibu|prof\.?\s*dr\.?|dr\.?|prof\.?)\s+',
    re.IGNORECASE)


def _bersihkan_nama(nama: str) -> str:
    """Strip academic titles and informal prefixes."""
    # Strip lecturer/supervisor prefix
    nama = re.sub(
        r'^(?:dosen\s+)?(?:pembimbing\s+(?:pertama|kedua|1|2|i|ii)?\s*)',
        '', nama, flags=re.IGNORECASE).strip()
    # Strip informal prefixes for accurate searching
    nama = _PREFIX_INFORMAL_RE.sub('', nama).strip()
    # Strip titles iteratively until clean
    sebelum = None
    while nama != sebelum:
        sebelum = nama
        nama = _GELAR_RE.sub('', nama).strip()
    # If comma remains, keep only the front part
    if ',' in nama:
        nama = nama.split(',')[0].strip()
    return nama.strip('?.!, ')


def fix_encoding(teks: str) -> str:
    """Fix mojibake/encoding issues from database."""
    if not isinstance(teks, str):
        return teks
    
    # Fix specific common mojibake words
    teks = teks.replace("naã ve", "naive").replace("naïve", "naive").replace("naã\xafve", "naive")
    teks = teks.replace("k â medoids", "k-medoids").replace("k â€“ medoids", "k-medoids")
    
    # Replace non-standard quotes with standard ones
    reps = {
        "â€œ": '"', "â€\x9d": '"', "â€": '"',
        "â€˜": "'", "â€™": "'",
        "â€“": "-", "â€”": "-", 
        "Ã¢": "a", "Ã©": "e", "Ã": "i"
    }
    for k, v in reps.items():
        teks = teks.replace(k, v)
    return teks

def apply_fix_encoding(data):
    if isinstance(data, str):
        return fix_encoding(data)
    elif isinstance(data, list):
        return [apply_fix_encoding(x) for x in data]
    elif isinstance(data, dict):
        return {k: apply_fix_encoding(v) for k, v in data.items()}
    return data


def _ekstrak_nama_dosen_dari_q(q: str) -> str:
    """Ambil nama dosen dari pertanyaan user, abaikan embel-embel kalimatnya."""
    pola_list = [
        r'(?:dibimbing\s+(?:oleh\s+)?|bimbingan\s+)(?:dosen\s+)?(?:pak|bu|bapak|ibu|prof\.?\s*dr\.?|dr\.?)?\s*(.+?)(?:\?|$|,|\s+dan\s)',
        r'(?:pak|bu|bapak|ibu)\s+([a-z][a-z\'\s]+?)(?:\?|$|,|\s+itu|\s+dan\s)',
        r'dosen\s+(?:pembimbing\s+)?(?:pak|bu|bapak|ibu|dr\.?|prof\.?)?\s*([a-z][a-z\'\.\s]+?)(?:\?|$|,)',
    ]
    for p in pola_list:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            n = _bersihkan_nama(m.group(1).strip().rstrip('?.,! '))
            if len(n) > 2:
                return n
    return ''

# Penanganan Pola Pertanyaan Langsung (Cypher)
def query_agregat_neo4j(query: str, driver) -> str:
    q = query.lower().strip()

    def _run(cypher, **params):
        cypher = cypher.replace("toLower(dos.nama) CONTAINS", "replace(toLower(dos.nama), \"'\", \"\") CONTAINS")
        cypher = cypher.replace("toLower(mhs.nama) CONTAINS", "replace(toLower(mhs.nama), \"'\", \"\") CONTAINS")
        cypher = cypher.replace("toLower(p) CONTAINS", "replace(toLower(p), \"'\", \"\") CONTAINS")
        if params:
            params = {k: v.replace("'", "") if isinstance(v, str) else v for k, v in params.items()}
        def _ex():
            with driver.session() as s:
                data = s.run(cypher, **params).data()
                if data and isinstance(data, list) and 'judul' in data[0]:
                    seen = set()
                    dedup = []
                    for r in data:
                        j = r.get('judul')
                        if j not in seen:
                            seen.add(j)
                            dedup.append(r)
                    return apply_fix_encoding(dedup)
                return apply_fix_encoding(data)
        try:
            return _retry(driver, _ex) or []
        except Exception:
            return []

    def _run1(cypher, **params):
        cypher = cypher.replace("toLower(dos.nama) CONTAINS", "replace(toLower(dos.nama), \"'\", \"\") CONTAINS")
        cypher = cypher.replace("toLower(mhs.nama) CONTAINS", "replace(toLower(mhs.nama), \"'\", \"\") CONTAINS")
        cypher = cypher.replace("toLower(p) CONTAINS", "replace(toLower(p), \"'\", \"\") CONTAINS")
        if params:
            params = {k: v.replace("'", "") if isinstance(v, str) else v for k, v in params.items()}
        def _ex():
            with driver.session() as s:
                return apply_fix_encoding(s.run(cypher, **params).single())
        try:
            return _retry(driver, _ex)
        except Exception:
            return None

    def _bersihkan_q(teks: str) -> str:
        """Bersihkan nama dari string query."""
        return _bersihkan_nama(teks.strip().rstrip('?.,! '))

    # ------------------------------------------------------------------
    # POLA 0: Deteksi bahasa informal / slang → normalisasi dulu
    # ------------------------------------------------------------------
    q_norm = q  # Use normalized query

    # Lecturer Alias Mapping (Ground Truth)
    ALIAS_DOSEN = {
        "pak dwi": "dwi kuswanto",
        "bu dwi": "andharini dwi cahyani",
        "bu andharini": "andharini dwi cahyani",
        "pak budi": "budi dwi satoto",
        "pak yoga": "yoga dwitya",
        "pak husni": "husni",
        "pak mulaab": "mula'ab",
        "pak mula'ab": "mula'ab",
        "pak firdaus": "firdaus solihin",
        "pak heri": "heri awalul",
        "pak sigit": "sigit susanto",
        "bu bain": "bain khusnul",
        "bu fika": "fika hastarita",
        "bu rima": "rima tri",
        "bu rika": "rika yunitarini",
        "pak yusuf": "muhammad yusuf",
        "pak fuad": "muhammad fuad",
        "pak aeri": "aeri rachmad",
        "pak arif": "arif muntasa",
        "bu eka": "eka mala",
        "pak iwan": "iwan santosa",
        "pak yonathan": "yonathan ferry",
        "pak syarief": "mohammad syarief",
        "pak khozaimi": "ach khozaimi",
        "bu indah": "indah agustien",
        "bu novi": "novi prastiti",
        "bu imamah": "imamah",
        "pak ubaidillah": "achmad ubaidillah",
        "bu sri": "sri herawati",
        "pak jauhari": "achmad jauhari",
        "pak yasid": "achmad yasid",
        "bu ari": "ari kusumaningsih",
        "bu arik": "arik kurniawati",
        "pak cucun": "cucun very",
        "bu devie": "devie rosa",
        "bu diana": "diana rahmawati",
        "bu ifada": "noor ifada",
        "pak rachmad": "rachmad hidayat",
        "bu yeni": "yeni kustiyahningsih",
        "pak umam": "faikul umam",
        "pak firli": "firli irhamni",
        "pak firman": "firman farid",
        "pak haryanto": "haryanto",
        "pak hermawan": "hermawan",
        "bu ika": "ika oktavia",
        "pak khamdi": "khamdi mubarok",
        "pak koko": "koko joni",
        "pak kurniawan": "kurniawan eka",
        "pak meidya": "meidya koeshardianto",
        "pak sophan": "kautsar sophan",
        "pak ali": "ali syakur",
        "bu rosida": "rosida vivin",
        "pak wahyudi": "wahyudi agustiono"
    }
    
    for k, v in ALIAS_DOSEN.items():
        q_norm = re.sub(rf'\b{k}\b', v, q_norm)

    # Normalize informal greetings and slang
    q_norm = re.sub(r'\bgasih\b|\bgak sih\b|\btidak ya\b|\bnggak\b|\bngak\b|\bgak\b', '', q_norm)
    q_norm = re.sub(r'\bada (?:gak|tidak|nggak|ngak)?\s*', 'ada ', q_norm)
    q_norm = re.sub(r'\bapakah ada\b', 'ada', q_norm)
    q_norm = re.sub(r'\b(?:bisa\s+)?(?:tolong(?:in)?\s+)?(?:bantu(?:in)?\s+)?cari(?:in|kan)?\b', 'carikan', q_norm)
    q_norm = re.sub(r'\b(?:mau\s+)?(?:lihat|liat|tunjukin)\b', 'tampilkan', q_norm)
    q_norm = re.sub(r'\b(?:sebutkan|list|daftar|kasih tau|kasih info)\b', 'tampilkan', q_norm)
    
    # Standardize campus terminology
    q_norm = re.sub(r'\b(?:ta|tugas akhir)\b', 'skripsi', q_norm)
    q_norm = re.sub(r'\b(?:dospem|dosen pembimbing|pembimbing satu|pembimbing 1|pembimbing utama)\b', 'dosen', q_norm)
    
    # Normalize lecturer queries
    q_norm = re.sub(r'\b(?:siapa|siapakah)\s+(?:sih\s+)?(?:yang\s+)?(?:membimbing|pembimbingnya?|dosennya?)\b', 'dibimbing oleh', q_norm)
    q_norm = re.sub(r'\b(?:siapa|siapakah)\s+(?:dosen|pembimbing)\s+(?:dari\s+)?(?:skripsi\s+)?\b', 'dibimbing oleh', q_norm)
    
    # Normalize title queries
    q_norm = re.sub(r'\b(?:apa\s+judulnya?|judulnya?\s+apa)\b', 'tampilkan judul', q_norm)

    # Strip conversation suffixes
    q_norm = re.sub(r'\bskripsinya\b', 'skripsi', q_norm)
    q_norm = re.sub(r'\bpenelitiannya\b', 'penelitian', q_norm)
    q_norm = re.sub(r'\bjudulnya\b', 'judul', q_norm)
    q_norm = re.sub(r'\bdosennya\b', 'dosen', q_norm)
    q_norm = re.sub(r'\bpembimbingnya\b', 'pembimbing', q_norm)
    q_norm = re.sub(r'\bpunyanya\b', 'milik', q_norm)

    q_norm = re.sub(r'\b(?:anak-anak\s+|anak\s+)?(?:yang\s+)?(?:dibimbing|bimbingannya)\b', 'bimbingan', q_norm)
    q_norm = re.sub(r'\bngebimbing\b', 'membimbing', q_norm)
    q_norm = re.sub(r'\bneliti\b', 'meneliti', q_norm)
    q_norm = re.sub(r'\bbuat\b', 'untuk', q_norm)
    q_norm = re.sub(r'\bsoal\b|\bbahas\b', 'tentang', q_norm)
    
    # Hapus sisa-sisa kata gaul di akhir kalimat
    q_norm = re.sub(r'\s+(?:ya|dong|sih|nih|deh)(?:\?|\.|!)*$', '', q_norm)
    q_norm = re.sub(r'\s+(?:ya|dong|sih|nih|deh)\s+', ' ', q_norm)
    q_norm = re.sub(r'\s+', ' ', q_norm).strip()

    # Cek apakah user tanya jumlah skripsi bimbingan dosen tertentu
    m = re.search(
        r'berapa\s+(?:jumlah\s+|banyak\s+)?skripsi\s+(?:yang\s+)?'
        r'(?:dibimbing|bimbingan|diawasi)\s+(?:oleh\s+)?(?:dosen\s+)?(.+)',
        q_norm)
    if m and ' dan ' not in m.group(1):
        nama_bersih = _bersihkan_q(m.group(1))
        if len(nama_bersih) >= 2:
            res = _run1('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                WHERE toLower(dos.nama) CONTAINS $nama
                RETURN dos.nama AS nama_dosen, count(d) AS jumlah
            ''', nama=nama_bersih.lower())
            if res:
                return (f"Terdapat **{res['jumlah']} skripsi** yang dibimbing oleh "
                        f"**{res['nama_dosen']}** dalam basis data PTA UTM.")
            else:
                return (f"Tidak ditemukan dosen dengan nama mengandung "
                        f"**'{nama_bersih}'** dalam basis data PTA UTM.")

    # Cek apakah user cuma tanya keberadaan skripsi (ada/nggak)
    m_ada_mhs = re.search(
        r'ada\s+(?:skripsi|dokumen|penelitian)\s+'
        r'(?:yang\s+)?(?:disusun|ditulis|dibuat|milik|punya|dari|oleh)\s+'
        r'(?:mahasiswa\s+)?(?:bernama\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m_ada_mhs:
        nama_bersih = _bersihkan_q(m_ada_mhs.group(1))
        if len(nama_bersih) >= 2:
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                RETURN DISTINCT mhs.nama AS mahasiswa, d.judul AS judul,
                       collect(DISTINCT dos.nama) AS pembimbing,
                       collect(DISTINCT kk.nama)[..5] AS kata_kunci
            ''', mhs=nama_bersih.lower())
            if res:
                r = res[0]
                pemb_str = ', '.join(r['pembimbing']) if r['pembimbing'] else '-'
                kk_str   = ', '.join(r['kata_kunci'])  if r['kata_kunci']  else '-'
                return (f"Ya, terdapat skripsi milik **{r['mahasiswa']}**:\n\n"
                        f"**{r['judul']}**\n\n"
                        f"Pembimbing: {pemb_str}\nKata kunci: {kk_str}")
            else:
                return (f"Tidak ditemukan skripsi milik mahasiswa bernama "
                        f"**'{nama_bersih}'** dalam basis data PTA UTM.")

    m_ada_topik = re.search(
        r'ada\s+(?:skripsi|dokumen|penelitian)\s+'
        r'(?:yang\s+)?(?:membahas|tentang|menggunakan\s+(?:metode\s+)?)\s+(.+?)(?:\?|$)',
        q_norm)
    if m_ada_topik:
        topik = _bersihkan_q(m_ada_topik.group(1))
        if len(topik) >= 2:
            res = _run('''
                MATCH (d:Dokumen)
                WHERE toLower(d.judul) CONTAINS $t OR toLower(d.abstrak) CONTAINS $t
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis,
                       collect(DISTINCT dos.nama) AS pembimbing
                ORDER BY d.judul LIMIT 10
            ''', t=topik.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(res)])
                return (f"Ya, ditemukan **{len(res)} skripsi** yang membahas "
                        f"**'{topik}'**:\n\n{daftar}")
            else:
                return (f"Tidak ditemukan skripsi yang membahas **'{topik}'** "
                        f"dalam basis data PTA UTM.")

    m_ada_dos = re.search(
        r'ada\s+(?:skripsi|dokumen|penelitian)\s+'
        r'(?:yang\s+)?dibimbing\s+(?:oleh\s+)?(?:dosen\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m_ada_dos and ' dan ' not in m_ada_dos.group(1):
        nama_bersih = _bersihkan_q(m_ada_dos.group(1))
        if len(nama_bersih) >= 2:
            res = _run1('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                WHERE toLower(dos.nama) CONTAINS $nama
                RETURN dos.nama AS nama_dosen, count(d) AS jumlah
            ''', nama=nama_bersih.lower())
            if res:
                return (f"Ya, terdapat **{res['jumlah']} skripsi** yang dibimbing "
                        f"oleh **{res['nama_dosen']}**.")
            else:
                return (f"Tidak ditemukan dosen dengan nama **'{nama_bersih}'** "
                        f"dalam basis data PTA UTM.")

    # Cek pencarian skripsi dengan dua dosen pembimbing
    m2dos = re.search(
        r'(?:carikan?|tampilkan|daftar|list|ada(?:kah)?|berapa(?: banyak| jumlah)?)\s+(?:skripsi\s+)?'
        r'(?:yang\s+)?(?:dibimbing|bimbingan)\s+(?:oleh\s+)?'
        r'(?:dosen\s+)?(.+?)\s+dan\s+(?:dosen\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m2dos:
        nama1 = _bersihkan_q(m2dos.group(1))
        nama2 = _bersihkan_q(m2dos.group(2))
        if len(nama1) >= 2 and len(nama2) >= 2:
            res = _run('''
                MATCH (d:Dokumen)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                WITH d, mhs, collect(DISTINCT dos.nama) AS pembimbing
                WHERE any(p IN pembimbing WHERE toLower(p) CONTAINS $n1)
                  AND any(p IN pembimbing WHERE toLower(p) CONTAINS $n2)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis,
                       pembimbing
                ORDER BY d.judul
            ''', n1=nama1.lower(), n2=nama2.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})\n"
                    f"     Pembimbing: {', '.join(r['pembimbing'])}"
                    for i, r in enumerate(res)])
                return (f"Ditemukan **{len(res)} skripsi** yang dibimbing bersama "
                        f"**{nama1.title()}** dan **{nama2.title()}**:\n\n{daftar}")
            else:
                # Kalau barengan nggak ketemu, coba cari datanya satu-satu
                r1 = _run1('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                    WHERE toLower(dos.nama) CONTAINS $n RETURN dos.nama AS nama, count(d) AS jml
                ''', n=nama1.lower())
                r2 = _run1('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                    WHERE toLower(dos.nama) CONTAINS $n RETURN dos.nama AS nama, count(d) AS jml
                ''', n=nama2.lower())
                i1 = f"**{r1['nama']}** ({r1['jml']} skripsi)" if r1 else f"'{nama1}' tidak ditemukan"
                i2 = f"**{r2['nama']}** ({r2['jml']} skripsi)" if r2 else f"'{nama2}' tidak ditemukan"
                return (f"Tidak ditemukan skripsi yang dibimbing **bersama-sama** "
                        f"oleh **{nama1.title()}** dan **{nama2.title()}**.\n\n"
                        f"Data masing-masing:\n- {i1}\n- {i2}\n\n"
                        f"Coba cari skripsi masing-masing dosen secara terpisah.")

    # Cari skripsi berdasarkan topik yang dibimbing dosen tertentu
    m = re.search(
        r'(?:tampilkan|carikan?|cari)\s+(?:referensi\s+penelitian|skripsi|dokumen)\s+'
        r'(?:dalam\s+bidang\s+(?:keilmuan\s+)?)?tentang\s+(.+?)\s+'
        r'(?:yang\s+)?dibimbing\s+(?:oleh\s+)?(?:dosen\s+)?(.+?)(?:\?|$)',
        q_norm)
    if not m:
        m = re.search(
            r'(?:tampilkan|carikan?)\s+skripsi\s+dalam\s+bidang\s+(?:keilmuan\s+)?(.+?)\s+'
            r'(?:yang\s+)?(?:disusun\s+di\s+bawah\s+bimbingan|dibimbing)\s+(?:dosen\s+)?(.+?)(?:\?|$)',
            q_norm)
    if m:
        topik_raw = m.group(1).strip().rstrip('?.,! ')
        dosen_raw = _bersihkan_q(m.group(2))
        if len(topik_raw) > 2 and len(dosen_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                WHERE toLower(dos.nama) CONTAINS $dosen
                  AND (toLower(d.judul) CONTAINS $topik
                    OR toLower(d.abstrak) CONTAINS $topik)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis,
                       dos.nama AS nama_dosen
                ORDER BY d.judul
            ''', dosen=dosen_raw.lower(), topik=topik_raw.lower())
            if res:
                nama_resmi = res[0]['nama_dosen']
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(res)])
                return (f"Ditemukan **{len(res)} skripsi** tentang **'{topik_raw}'** "
                        f"yang dibimbing oleh **{nama_resmi}**:\n\n{daftar}")

    # Cari skripsi mahasiswa tertentu dengan pembimbing tertentu
    m = re.search(
        r'(?:tampilkan|carikan?|cari)\s+(?:dokumen\s+)?skripsi\s+'
        r'(?:yang\s+)?disusun\s+oleh\s+(.+?)\s+'
        r'(?:dengan\s+)?dosen\s+pembimbing\s+(.+?)(?:\?|$)',
        q_norm)
    if m:
        mhs_raw   = m.group(1).strip().rstrip('?.,! ')
        dosen_raw = _bersihkan_q(m.group(2))
        if len(mhs_raw) > 2 and len(dosen_raw) > 2:
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)-[:DIBIMBING_OLEH]->(dos:Dosen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                  AND toLower(dos.nama) CONTAINS $dosen
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS mahasiswa,
                       dos.nama AS nama_dosen,
                       collect(DISTINCT kk.nama)[..5] AS kata_kunci
            ''', mhs=mhs_raw.lower(), dosen=dosen_raw.lower())
            if res:
                r = res[0]
                kk_str = ', '.join(r['kata_kunci']) if r['kata_kunci'] else '-'
                return (f"Ditemukan skripsi milik **{r['mahasiswa']}** "
                        f"yang dibimbing **{r['nama_dosen']}**:\n\n"
                        f"**{r['judul']}**\n\nKata kunci: {kk_str}")

    # Cari skripsi mahasiswa tertentu yang bahas topik tertentu
    m = re.search(
        r'(?:tampilkan|carikan?)\s+(?:dokumen\s+)?skripsi\s+milik\s+(.+?)\s+'
        r'yang\s+(?:menerapkan\s+metode|menggunakan|membahas)\s+(.+?)(?:\?|$)',
        q_norm)
    if m:
        mhs_raw   = m.group(1).strip().rstrip('?.,! ')
        topik_raw = m.group(2).strip().rstrip('?.,! ')
        if len(mhs_raw) > 2 and len(topik_raw) > 2:
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                  AND (toLower(d.judul) CONTAINS $topik
                    OR toLower(d.abstrak) CONTAINS $topik)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS mahasiswa,
                       collect(DISTINCT dos.nama) AS pembimbing,
                       collect(DISTINCT kk.nama)[..5] AS kata_kunci
            ''', mhs=mhs_raw.lower(), topik=topik_raw.lower())
            if res:
                r = res[0]
                return (f"Skripsi milik **{r['mahasiswa']}**:\n\n**{r['judul']}**\n\n"
                        f"Pembimbing: {', '.join(r['pembimbing']) or '-'}\n"
                        f"Kata kunci: {', '.join(r['kata_kunci']) or '-'}")

    # Cari skripsi berdasarkan nama mahasiswa saja
    m = re.search(
        r'(?:tampilkan|carikan?)\s+(?:dokumen\s+)?(?:skripsi|penelitian)\s+'
        r'(?:milik\s+|disusun\s+oleh\s+(?:mahasiswa\s+)?(?:bernama\s+)?)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        mhs_raw = m.group(1).strip().rstrip('?.,! ')
        # Pastikan ini benar-benar nyari nama orang, bukan nanya dosen atau topik
        if (len(mhs_raw) > 2
                and 'dosen' not in mhs_raw
                and 'pembimbing' not in mhs_raw
                and 'berfokus' not in mhs_raw
                and 'kategori' not in mhs_raw
                and 'metode' not in mhs_raw
                and 'membahas' not in mhs_raw
                and 'tentang' not in mhs_raw):
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                OPTIONAL MATCH (d)-[:TERMASUK_KATEGORI]->(k:Kategori)
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS mahasiswa,
                       collect(DISTINCT dos.nama) AS pembimbing,
                       k.nama AS kategori,
                       collect(DISTINCT kk.nama)[..5] AS kata_kunci
            ''', mhs=mhs_raw.lower())
            if res:
                r = res[0]
                return (f"Ditemukan skripsi milik **{r['mahasiswa']}**:\n\n"
                        f"**{r['judul']}**\n\n"
                        f"Pembimbing: {', '.join(r['pembimbing']) or '-'}\n"
                        f"Kategori: {r['kategori'] or '-'}\n"
                        f"Kata kunci: {', '.join(r['kata_kunci']) or '-'}")

    # Cari skripsi berdasarkan kategori dan topiknya
    m = re.search(
        r'(?:tampilkan|carikan?)\s+(?:dokumen\s+)?skripsi\s+(?:pada|dalam)\s+'
        r'kategori\s+(.+?)\s+yang\s+membahas\s+(?:tentang\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        kat_raw   = m.group(1).strip().rstrip('?.,! ')
        topik_raw = m.group(2).strip().rstrip('?.,! ')
        if len(kat_raw) > 2 and len(topik_raw) > 2:
            res = _run('''
                MATCH (d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                WHERE toLower(k.nama) CONTAINS $kat
                  AND (toLower(d.judul) CONTAINS $topik
                    OR toLower(d.abstrak) CONTAINS $topik)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis,
                       k.nama AS kategori
                ORDER BY d.judul LIMIT 20
            ''', kat=kat_raw.lower(), topik=topik_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(res)])
                return (f"Ditemukan **{len(res)} skripsi** dalam kategori "
                        f"**{res[0]['kategori']}** yang membahas **'{topik_raw}'**:\n\n{daftar}")

    # Cari skripsi cuma berdasarkan nama kategorinya
    m = re.search(
        r'(?:tampilkan|carikan?)\s+(?:dokumen\s+)?skripsi\s+'
        r'(?:yang\s+)?termasuk\s+(?:dalam\s+)?kategori\s+(.+?)(?:\?|$)',
        q_norm)
    if m:
        kat_raw = m.group(1).strip().rstrip('?.,! ')
        if len(kat_raw) > 2:
            total_res = _run1('''
                MATCH (d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                WHERE toLower(k.nama) CONTAINS $kat
                RETURN k.nama AS kat_resmi, count(DISTINCT d) AS jml
            ''', kat=kat_raw.lower())
            if total_res:
                sample = _run('''
                    MATCH (d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                    WHERE toLower(k.nama) CONTAINS $kat
                    OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                    RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis
                    ORDER BY d.judul LIMIT 20
                ''', kat=kat_raw.lower())
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(sample)])
                catatan = (f"\n\n*Menampilkan 20 dari {total_res['jml']} total skripsi.*"
                           if total_res['jml'] > 20 else '')
                return (f"Kategori **{total_res['kat_resmi']}** memiliki "
                        f"**{total_res['jml']} skripsi**:\n\n{daftar}{catatan}")

    # Cari judul skripsi berdasarkan topik atau metode
    m = re.search(
        r'(?:carikan?|tampilkan)\s+(?:judul\s+|dokumen\s+)?skripsi\s+'
        r'(?:yang\s+)?(?:menggunakan\s+(?:metode\s+(?:atau\s+membahas\s+)?)?|'
        r'membahas\s+(?:tentang\s+)?|menerapkan\s+(?:metode\s+)?|tentang\s+)(.+?)(?:\?|$)',
        q_norm)
    if m:
        topik_raw = m.group(1).strip().rstrip('?.,! ')
        if len(topik_raw) > 2:
            res = _run('''
                MATCH (d:Dokumen)
                WHERE toLower(d.judul) CONTAINS $t
                   OR toLower(d.abstrak) CONTAINS $t
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                WITH d, collect(DISTINCT kk.nama) AS kata_kunci
                WHERE toLower(d.judul) CONTAINS $t
                   OR toLower(d.abstrak) CONTAINS $t
                   OR any(k IN kata_kunci WHERE toLower(k) CONTAINS $t)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis
                ORDER BY d.judul LIMIT 20
            ''', t=topik_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(res)])
                return (f"Ditemukan **{len(res)} skripsi** yang membahas "
                        f"**'{topik_raw}'**:\n\n{daftar}")
            else:
                return (f"Tidak ditemukan skripsi dengan topik **'{topik_raw}'**. "
                        f"Coba kata kunci yang berbeda atau lebih umum.")

    # Cari semua daftar skripsi bimbingan dosen tertentu
    m = re.search(
        r'(?:carikan?|tampilkan|sebutkan|list|daftar)\s+(?:daftar\s+|semua\s+)?'
        r'(?:dokumen\s+)?skripsi\s+(?:yang\s+)?'
        r'(?:dibimbing|bimbingan|diawasi)\s+(?:oleh\s+)?'
        r'(?:dosen\s+)?(?:pembimbing\s+)?(.+?)(?:\?|$)',
        q_norm)
    if not m:
        m = re.search(
            r'(?:carikan?|tampilkan|sebutkan)\s+(?:semua\s+)?skripsi\s+'
            r'(?:yang\s+)?(?:dibimbing|bimbingan)\s+(?:oleh\s+)?'
            r'(?:dosen\s+)?(?:pak|bu|bapak|ibu)?\s*(.+?)(?:\?|$)',
            q_norm)
    if not m:
        # Gaya bahasa sehari-hari buat nanya bimbingan dosen
        m = re.search(
            r'(?:daftar|list|semua|tampilkan|carikan)\s+skripsi\s+bimbingan\s+(?:pak|bu|bapak|ibu)?\s*(.+?)(?:\?|$)',
            q_norm)
    if m:
        nama_raw    = m.group(1).strip().rstrip('?.,! ')
        nama_bersih = _bersihkan_q(nama_raw)
        if len(nama_bersih) >= 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                WHERE toLower(dos.nama) CONTAINS $nama
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT dos.nama AS nama_dosen,
                       d.judul AS judul, mhs.nama AS penulis
                ORDER BY d.judul
            ''', nama=nama_bersih.lower())
            if res:
                nama_resmi = res[0]['nama_dosen']
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** (Penulis: {r['penulis'] or '-'})"
                    for i, r in enumerate(res)])
                return (f"Terdapat **{len(res)} skripsi** yang dibimbing oleh "
                        f"**{nama_resmi}**:\n\n{daftar}")
            else:
                return (f"Tidak ditemukan dosen dengan nama mengandung "
                        f"**'{nama_bersih}'** dalam basis data PTA UTM.")

    # Penanganan pola pertanyaan kepo/spesifik (siapa, apa, di mana)

    # Nanyain topik atau kata kunci dari skripsi seseorang
    m = re.search(
        r'apa\s+(?:topik\s+utama\s+(?:atau\s+)?kata\s+kunci|kata\s+kunci|topik)\s+'
        r'(?:dari\s+|utama\s+)?(?:penelitian\s+)?(?:skripsi\s+)?(?:milik\s+|dari\s+)?(.+?)(?:\?|$)',
        q_norm)
    if not m:
        m = re.search(
            r'(?:topik|kata\s+kunci)\s+.{0,15}(?:skripsi|penelitian)\s+'
            r'(?:milik|dari|punya|oleh)\s+(.+?)(?:\?|$)',
            q_norm)
    if m:
        mhs_raw = _bersihkan_q(m.group(1))
        if len(mhs_raw) > 2:
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                RETURN DISTINCT mhs.nama AS mahasiswa, d.judul AS judul,
                       collect(DISTINCT kk.nama) AS kata_kunci,
                       collect(DISTINCT dos.nama) AS pembimbing
            ''', mhs=mhs_raw.lower())
            if res:
                r = res[0]
                kk_str   = ', '.join(r['kata_kunci'])  if r['kata_kunci']  else '-'
                pemb_str = ', '.join(r['pembimbing']) if r['pembimbing'] else '-'
                return (f"Topik utama dari skripsi milik **{r['mahasiswa']}** berjudul:\n"
                        f"*\"{r['judul']}\"*\n\n"
                        f"Kata kunci: **{kk_str}**\nPembimbing: {pemb_str}")

    # Cari tahu kategori dari topik yang dibimbing dosen tertentu
    m = re.search(
        r'masuk\s+(?:ke\s+dalam\s+)?kategori\s+(?:apakah|apa)\s+'
        r'penelitian\s+skripsi\s+tentang\s+(.+?)\s+'
        r'yang\s+dibimbing\s+(?:oleh\s+)?(?:dosen\s+)?(?:pembimbing\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        topik_raw = m.group(1).strip().rstrip('?.,! ')
        dosen_raw = _bersihkan_q(m.group(2))
        if len(topik_raw) > 2 and len(dosen_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                WHERE toLower(dos.nama) CONTAINS $dosen
                  AND (toLower(d.judul) CONTAINS $topik
                    OR toLower(d.abstrak) CONTAINS $topik)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT k.nama AS kategori, d.judul AS judul,
                       mhs.nama AS penulis, dos.nama AS nama_dosen
            ''', dosen=dosen_raw.lower(), topik=topik_raw.lower())
            if res:
                r = res[0]
                return (f"Penelitian skripsi tentang **'{topik_raw}'** yang "
                        f"dibimbing **{r['nama_dosen']}** berjudul:\n"
                        f"*\"{r['judul']}\"*\n\n"
                        f"Termasuk dalam kategori: **{r['kategori']}**\n"
                        f"Penulis: {r['penulis'] or '-'}")

    # Nanyain kategori skripsi yang diambil mahasiswa
    m = re.search(
        r'kategori\s+(?:keilmuan\s+)?apa\s+.{0,30}'
        r'mahasiswa\s+(?:bernama\s+)?(.+?)(?:\?|$)',
        q_norm)
    if not m:
        m = re.search(
            r'kategori\s+(?:keilmuan\s+)?apa\s+(?:yang\s+)?(?:diambil|dipilih)\s+oleh\s+'
            r'(?:mahasiswa\s+)?(?:bernama\s+)?(.+?)(?:\?|$)',
            q_norm)
    if m:
        mhs_raw = _bersihkan_q(m.group(1))
        if len(mhs_raw) > 2:
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                RETURN DISTINCT mhs.nama AS mahasiswa, k.nama AS kategori,
                       d.judul AS judul, collect(DISTINCT dos.nama) AS pembimbing
            ''', mhs=mhs_raw.lower())
            if res:
                r = res[0]
                return (f"Mahasiswa **{r['mahasiswa']}** mengambil kategori "
                        f"**{r['kategori']}** dengan skripsi:\n"
                        f"*\"{r['judul']}\"*\n\n"
                        f"Pembimbing: {', '.join(r['pembimbing']) or '-'}")

    # Kepo siapa aja mahasiswa yang bahas topik tertentu
    m = re.search(
        r'siapa\s+(?:saja\s+)?mahasiswa\s+yang\s+(?:menulis|menyusun|membuat)\s+'
        r'skripsi\s+(?:dengan\s+)?(?:topik\s+)?(?:pembahasan\s+|tentang\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        topik_raw = m.group(1).strip().rstrip('?.,! ')
        if len(topik_raw) > 2:
            res = _run('''
                MATCH (d:Dokumen)
                WHERE toLower(d.judul) CONTAINS $t OR toLower(d.abstrak) CONTAINS $t
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                RETURN DISTINCT mhs.nama AS penulis, d.judul AS judul,
                       collect(DISTINCT dos.nama) AS pembimbing
                ORDER BY d.judul LIMIT 10
            ''', t=topik_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['penulis'] or '-'}** — *{r['judul']}*"
                    for i, r in enumerate(res)])
                return (f"Mahasiswa yang menulis skripsi terkait **'{topik_raw}'**:\n\n{daftar}")

    # Nyari tahu dosen pembimbing dari skripsi atau topik tertentu
    m = re.search(
        r'siapa\s+(?:nama\s+)?dosen\s+pembimbing\s+(?:dari\s+|untuk\s+)?'
        r'(?:penelitian\s+)?skripsi\s+(?:yang\s+)?(?:disusun\s+oleh\s+mahasiswa\s+bernama\s+|'
        r'tentang\s+|milik\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        subjek = _bersihkan_q(m.group(1))
        if len(subjek) > 2:
            # Coba cek apakah kata kuncinya berupa topik
            res = _run('''
                MATCH (d:Dokumen)-[:DIBIMBING_OLEH]->(dos:Dosen)
                WHERE toLower(d.judul) CONTAINS $s OR toLower(d.abstrak) CONTAINS $s
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT dos.nama AS nama_dosen, d.judul AS judul,
                       mhs.nama AS penulis
                ORDER BY d.judul LIMIT 10
            ''', s=subjek.lower())
            if not res:
                # Kalau bukan topik, coba cek sebagai nama mahasiswa
                res = _run('''
                    MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)-[:DIBIMBING_OLEH]->(dos:Dosen)
                    WHERE toLower(mhs.nama) CONTAINS $s
                    RETURN DISTINCT dos.nama AS nama_dosen, d.judul AS judul,
                           mhs.nama AS penulis
                ''', s=subjek.lower())
            if res:
                if len(res) == 1:
                    r = res[0]
                    return (f"Pembimbing skripsi **'{r['judul']}'** "
                            f"(Penulis: {r['penulis'] or '-'}) adalah:\n**{r['nama_dosen']}**")
                else:
                    daftar = '\n'.join([
                        f"  {i+1}. **{r['judul']}** — Pembimbing: {r['nama_dosen']}"
                        for i, r in enumerate(res)])
                    return (f"Ditemukan **{len(res)} skripsi** terkait **'{subjek}'**:\n\n{daftar}")

    # Daftar mahasiswa bimbingan dosen di bidang tertentu
    m = re.search(
        r'siapa\s+mahasiswa\s+(?:yang\s+)?dibimbing\s+(?:oleh\s+)?(.+?)\s+'
        r'(?:pada\s+)?bidang\s+(?:keilmuan\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        dosen_raw = _bersihkan_q(m.group(1))
        kat_raw   = m.group(2).strip().rstrip('?.,! ')
        if len(dosen_raw) > 2 and len(kat_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                WHERE toLower(dos.nama) CONTAINS $dosen
                  AND toLower(k.nama) CONTAINS $kat
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                RETURN DISTINCT mhs.nama AS mahasiswa, d.judul AS judul,
                       dos.nama AS nama_dosen, k.nama AS kategori
                ORDER BY mhs.nama LIMIT 30
            ''', dosen=dosen_raw.lower(), kat=kat_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['mahasiswa'] or '-'}** — *{r['judul']}*"
                    for i, r in enumerate(res)])
                return (f"Mahasiswa bimbingan **{res[0]['nama_dosen']}** "
                        f"pada bidang **{res[0]['kategori']}**:\n\n{daftar}")

    # Daftar mahasiswa bimbingan dosen dengan topik tertentu
    m = re.search(
        r'siapa\s+(?:saja\s+)?(?:mahasiswa\s+)?bimbingan\s+(.+?)\s+'
        r'yang\s+(?:melakukan\s+penelitian\s+terkait|meneliti\s+tentang|membahas)\s+(.+?)(?:\?|$)',
        q_norm)
    if m:
        dosen_raw = _bersihkan_q(m.group(1))
        topik_raw = m.group(2).strip().rstrip('?.,! ')
        if len(dosen_raw) > 2 and len(topik_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                WHERE toLower(dos.nama) CONTAINS $dosen
                  AND (toLower(d.judul) CONTAINS $topik
                    OR toLower(d.abstrak) CONTAINS $topik)
                RETURN DISTINCT dos.nama AS nama_dosen,
                       mhs.nama AS mahasiswa, d.judul AS judul
                ORDER BY mhs.nama
            ''', dosen=dosen_raw.lower(), topik=topik_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['mahasiswa'] or '-'}** — *{r['judul']}*"
                    for i, r in enumerate(res)])
                return (f"Mahasiswa bimbingan **{res[0]['nama_dosen']}** "
                        f"yang meneliti **'{topik_raw}'**:\n\n{daftar}")

    # Kepo dosen tertentu seringnya ngebimbing topik apa aja
    m = re.search(
        r'apa\s+saja\s+(?:topik|kata\s+kunci)\s+'
        r'(?:yang\s+)?(?:sering\s+)?(?:dibimbing|diawasi)\s+(?:oleh\s+)?'
        r'(?:dosen\s+)?(?:pembimbing\s+)?(.+?)(?:\?|$)',
        q_norm)
    if m:
        dosen_raw = _bersihkan_q(m.group(1))
        if len(dosen_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                WHERE toLower(dos.nama) CONTAINS $dosen
                RETURN dos.nama AS nama_dosen, kk.nama AS kata_kunci,
                       count(*) AS frekuensi
                ORDER BY frekuensi DESC LIMIT 15
            ''', dosen=dosen_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['kata_kunci']}** ({r['frekuensi']} skripsi)"
                    for i, r in enumerate(res)])
                return (f"Topik/kata kunci yang sering muncul pada skripsi "
                        f"bimbingan **{res[0]['nama_dosen']}**:\n\n{daftar}")

    # Lihat semua mahasiswa bimbingannya
    m = re.search(
        r'siapa\s+(?:saja\s+)?mahasiswa\s+(?:bimbingan|yang\s+dibimbing(?:\s+oleh)?)\s+(.+?)(?:\?|$)',
        q_norm)
    if m:
        dosen_raw = _bersihkan_q(m.group(1))
        if len(dosen_raw) > 2:
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                WHERE toLower(dos.nama) CONTAINS $dosen
                RETURN DISTINCT dos.nama AS nama_dosen,
                       mhs.nama AS mahasiswa, d.judul AS judul
                ORDER BY mhs.nama
            ''', dosen=dosen_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['mahasiswa'] or '-'}** — *{r['judul']}*"
                    for i, r in enumerate(res)])
                return (f"Terdapat **{len(res)} mahasiswa** bimbingan "
                        f"**{res[0]['nama_dosen']}**:\n\n{daftar}")

    # Penanganan gaya bahasa ngobrol santai

    # Nanya dosen biasa ngebimbing apa pakai bahasa santai
    pola_n1 = [
        re.search(r'(?:topik|tema|kata\s+kunci)\s+(?:apa|yang)\s+.{0,30}'
                  r'(?:dibimbing|bimbingan)\s+(?:oleh\s+)?(.+?)(?:\?|$)', q_norm),
        re.search(r'(?:pak|bu)\s+(\w+)\s+.{0,30}(?:membimbing|topik)\s+apa', q_norm),
        re.search(r'(?:pak|bu)\s+(\w+)\s+(?:biasanya|sering)\s+(?:membimbing|meneliti)', q_norm),
        re.search(r'(?:topik|tema)\s+.{0,20}(?:pak|bu)\s+(\w+)', q_norm),
    ]
    for m in pola_n1:
        if m:
            nama_raw = _bersihkan_q(m.group(1))
            nama_raw = re.sub(r'\s+(itu|ya|dong|apa|kan|sih)$', '', nama_raw).strip()
            if len(nama_raw) < 2: continue
            res = _run('''
                MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                WHERE toLower(dos.nama) CONTAINS $nama
                RETURN dos.nama AS nama_dosen, kk.nama AS kata_kunci,
                       count(*) AS frekuensi
                ORDER BY frekuensi DESC LIMIT 15
            ''', nama=nama_raw.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['kata_kunci']}** ({r['frekuensi']} skripsi)"
                    for i, r in enumerate(res)])
                return (f"Topik yang sering muncul pada skripsi bimbingan "
                        f"**{res[0]['nama_dosen']}**:\n\n{daftar}")

    # Nanya judul skripsi teman atau orang lain
    pola_n2 = [
        re.search(r'(?:apa\s+topik|judul|kata\s+kunci)\s+skripsi\s+'
                  r'(?:milik\s+|dari\s+)?(.+?)(?:\?|$)', q_norm),
        re.search(r'(.+?)\s+(?:itu\s+)?(?:skripsi|penelitian)\s+'
                  r'(?:tentang|topik|judul)\s+apa', q_norm),
        re.search(r'skripsi\s+(.+?)\s+(?:itu\s+)?judul(?:nya)?\s+apa', q_norm),
    ]
    for m in pola_n2:
        if m:
            nama_mhs = _bersihkan_q(m.group(1))
            nama_mhs = re.sub(r'\s+(itu|ya|dong|kan)$', '', nama_mhs).strip()
            if len(nama_mhs) < 2 or 'dosen' in nama_mhs: continue
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                RETURN DISTINCT mhs.nama AS mahasiswa, d.judul AS judul,
                       collect(DISTINCT kk.nama) AS kata_kunci,
                       collect(DISTINCT dos.nama) AS pembimbing
            ''', mhs=nama_mhs.lower())
            if res:
                r = res[0]
                return (f"Skripsi milik **{r['mahasiswa']}**:\n"
                        f"*\"{r['judul']}\"*\n\n"
                        f"Kata kunci: {', '.join(r['kata_kunci']) or '-'}\n"
                        f"Pembimbing: {', '.join(r['pembimbing']) or '-'}")

    # Kepo pembimbingnya mahasiswa tertentu
    pola_n3 = [
        re.search(r'(?:siapa|siapakah)\s+(?:dosen\s+)?pembimbing\s+'
                  r'(?:skripsi\s+)?(?:milik\s+|dari\s+|punya\s+)?(.+?)(?:\?|$)', q_norm),
        re.search(r'(.+?)\s+dibimbing\s+(?:oleh\s+)?siapa', q_norm),
        re.search(r'siapa\s+yang\s+membimbing\s+(?:skripsi\s+)?(.+?)(?:\?|$)', q_norm),
    ]
    for m in pola_n3:
        if m:
            nama_mhs = _bersihkan_q(m.group(1))
            nama_mhs = re.sub(r'\s+(itu|ya|dong|kan)$', '', nama_mhs).strip()
            if len(nama_mhs) < 2: continue
            res = _run('''
                MATCH (mhs:Mahasiswa)<-[:DITULIS_OLEH]-(d:Dokumen)-[:DIBIMBING_OLEH]->(dos:Dosen)
                WHERE toLower(mhs.nama) CONTAINS $mhs
                RETURN DISTINCT mhs.nama AS mahasiswa, d.judul AS judul,
                       collect(DISTINCT dos.nama) AS pembimbing
            ''', mhs=nama_mhs.lower())
            if res:
                r = res[0]
                return (f"Skripsi **{r['mahasiswa']}**: *\"{r['judul']}\"*\n\n"
                        f"Dibimbing oleh: **{', '.join(r['pembimbing'])}**")

    # Cek kategori sebuah topik skripsi
    pola_n5 = [
        re.search(r'(?:ada\s+)?(?:referensi\s+)?skripsi\s+(.+?)\s+yang\s+(?:tentang|membahas|meneliti)\s+(.+?)(?:\?|$)', q_norm),
        re.search(r'skripsi\s+(?:tentang|membahas|meneliti)\s+(.+?)\s+(?:masuknya\s+(?:di|ke)\s+|termasuk\s+(?:kategori\s+)?)(.+?)(?:\?|$)', q_norm)
    ]
    for m_n5 in pola_n5:
        if m_n5:
            if "masuknya" in m_n5.group(0) or "termasuk" in m_n5.group(0):
                topik_raw = m_n5.group(1).strip()
                kat_raw = m_n5.group(2).strip()
                is_tanya_kategori = True
            else:
                kat_raw = m_n5.group(1).strip()
                topik_raw = m_n5.group(2).strip()
                is_tanya_kategori = False
            
            if len(kat_raw) > 2 and len(topik_raw) > 2:
                res = _run('''
                    MATCH (d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                    WHERE toLower(k.nama) CONTAINS $kat
                      AND (toLower(d.judul) CONTAINS $topik
                        OR toLower(d.abstrak) CONTAINS $topik)
                    RETURN DISTINCT d.judul AS judul, k.nama AS kategori
                    ORDER BY d.judul LIMIT 10
                ''', kat=kat_raw.lower(), topik=topik_raw.lower())
                if res:
                    if is_tanya_kategori:
                        return f"Ya, skripsi tentang **'{topik_raw}'** masuk dalam kategori **{res[0]['kategori']}**.\nContoh: *{res[0]['judul']}*"
                    else:
                        daftar = '\n'.join([f"  {i+1}. **{r['judul']}**" for i, r in enumerate(res)])
                        return f"Ditemukan skripsi kategori **{res[0]['kategori']}** yang membahas **'{topik_raw}'**:\n\n{daftar}"

    # Nanya skripsi bimbingan dosen di kategori tertentu
    pola_n6 = [
        re.search(r'(?:ada\s+)?skripsi\s+(.+?)\s+yang\s+(?:bimbingan|dibimbing\s+oleh)\s+(.+?)(?:\s+ada\s+apa\s+saja|\s+apa\s+saja)?(?:\?|$)', q_norm)
    ]
    for m_n6 in pola_n6:
        if m_n6:
            kat_raw = m_n6.group(1).strip()
            dos_raw = _bersihkan_q(m_n6.group(2))
            if len(kat_raw) > 2 and len(dos_raw) > 2:
                res = _run('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                    WHERE toLower(dos.nama) CONTAINS $dosen
                      AND toLower(k.nama) CONTAINS $kat
                    RETURN DISTINCT d.judul AS judul, k.nama AS kategori, dos.nama AS nama_dosen
                    ORDER BY d.judul LIMIT 20
                ''', dosen=dos_raw.lower(), kat=kat_raw.lower())
                if res:
                    daftar = '\n'.join([f"  {i+1}. **{r['judul']}**" for i, r in enumerate(res)])
                    return (f"Skripsi kategori **{res[0]['kategori']}** yang dibimbing **{res[0]['nama_dosen']}**:\n\n{daftar}")

    # Nanya kategori topik dari bimbingan dosen tertentu
    pola_n7 = [
        re.search(r'skripsi\s+(?:tentang|membahas|meneliti)\s+(.+?)\s+yang\s+(?:dibimbing|bimbingan)\s+(.+?)\s+(?:itu\s+)?masuk(?:nya)?\s+(?:ke\s+)?kategori\s+(?:minat\s+)?apa', q_norm)
    ]
    for m_n7 in pola_n7:
        if m_n7:
            topik_raw = m_n7.group(1).strip()
            dos_raw = _bersihkan_q(m_n7.group(2))
            if len(topik_raw) > 2 and len(dos_raw) > 2:
                res = _run('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)-[:TERMASUK_KATEGORI]->(k:Kategori)
                    WHERE toLower(dos.nama) CONTAINS $dosen
                      AND (toLower(d.judul) CONTAINS $topik OR toLower(d.abstrak) CONTAINS $topik)
                    RETURN DISTINCT d.judul AS judul, k.nama AS kategori, dos.nama AS nama_dosen
                ''', dosen=dos_raw.lower(), topik=topik_raw.lower())
                if res:
                    return f"Skripsi tentang **'{topik_raw}'** yang dibimbing **{res[0]['nama_dosen']}** masuk ke kategori **{res[0]['kategori']}**.\n\nContoh: *\"{res[0]['judul']}\"*"

    # Nanya kaitan bimbingan bareng antara dua dosen
    m = re.search(
        r'(?:hubungan|kaitan|relasi|keterkaitan)\s+.{0,10}'
        r'(?:pak|bu|dr\.?|prof\.?)?\s*(\w[\w\s]+?)\s+'
        r'(?:dengan|dan|ke)\s+'
        r'(?:pak|bu|dr\.?|prof\.?)?\s*(\w[\w\s]+?)(?:\?|$|,)',
        q_norm)
    if m:
        nama1 = _bersihkan_q(m.group(1))
        nama2 = _bersihkan_q(m.group(2))
        if len(nama1) > 1 and len(nama2) > 1:
            res = _run('''
                MATCH (d:Dokumen)
                OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(mhs:Mahasiswa)
                OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                WITH d, mhs, collect(DISTINCT dos.nama) AS pembimbing
                WHERE any(p IN pembimbing WHERE toLower(p) CONTAINS $n1)
                  AND any(p IN pembimbing WHERE toLower(p) CONTAINS $n2)
                RETURN DISTINCT d.judul AS judul, mhs.nama AS penulis, pembimbing
                ORDER BY d.judul LIMIT 20
            ''', n1=nama1.lower(), n2=nama2.lower())
            if res:
                daftar = '\n'.join([
                    f"  {i+1}. **{r['judul']}** — Penulis: {r['penulis'] or '-'}"
                    for i, r in enumerate(res)])
                return (f"Ditemukan **{len(res)} skripsi** yang dibimbing bersama "
                        f"**{nama1.title()}** dan **{nama2.title()}**:\n\n{daftar}")
            else:
                r1 = _run1('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                    WHERE toLower(dos.nama) CONTAINS $n
                    RETURN dos.nama AS nama, count(d) AS jml
                ''', n=nama1.lower())
                r2 = _run1('''
                    MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
                    WHERE toLower(dos.nama) CONTAINS $n
                    RETURN dos.nama AS nama, count(d) AS jml
                ''', n=nama2.lower())
                i1 = f"**{r1['nama']}** ({r1['jml']} skripsi)" if r1 else f"'{nama1}' tidak ditemukan"
                i2 = f"**{r2['nama']}** ({r2['jml']} skripsi)" if r2 else f"'{nama2}' tidak ditemukan"
                return (f"Tidak ada skripsi yang dibimbing bersama.\n\n"
                        f"Data masing-masing:\n- {i1}\n- {i2}")

    # Pertanyaan statistik (Ditaruh di akhir biar nggak numpuk sama yang spesifik)
    if re.search(r'berapa\s+(?:jumlah\s+)?dosen\s+pembimbing', q_norm):
        r = _run1('MATCH (dos:Dosen) RETURN count(dos) AS jml', driver=driver)
        return f"Terdapat **{r['jml'] if r else 0} dosen pembimbing** dalam basis data PTA UTM."

    # Pastikan pola buat ngitung total skripsi dicek paling belakang
    if re.search(r'berapa\s+(?:total\s+|jumlah\s+)?(?:semua\s+)?(?:dokumen\s+)?skripsi'
                 r'(?!\s+(?:yang|dibimbing|milik|tentang|dalam|pada))', q_norm):
        r = _run1('MATCH (d:Dokumen) RETURN count(d) AS jml')
        return f"Terdapat **{r['jml'] if r else 0} skripsi** dalam basis data PTA UTM."

    if re.search(r'(?:daftar|sebutkan|apa\s+saja)\s+(?:semua\s+)?kategori', q_norm):
        res = _run('''
            MATCH (k:Kategori)<-[:TERMASUK_KATEGORI]-(d:Dokumen)
            RETURN k.nama AS kategori, count(d) AS jml ORDER BY jml DESC''')
        if res:
            daftar = '\n'.join([f"  {i+1}. **{r['kategori']}** ({r['jml']} skripsi)"
                                for i,r in enumerate(res)])
            return f"Kategori penelitian di PTA UTM:\n\n{daftar}"

    if re.search(r'(?:siapa|dosen)\s+.{0,20}(?:paling\s+banyak|terbanyak)\s+'
                 r'(?:membimbing|bimbingan)', q_norm):
        res = _run('''
            MATCH (dos:Dosen)<-[:DIBIMBING_OLEH]-(d:Dokumen)
            RETURN dos.nama AS nama_dosen, count(d) AS jml ORDER BY jml DESC LIMIT 10''')
        if res:
            daftar = '\n'.join([f"  {i+1}. **{r['nama_dosen']}** — {r['jml']} skripsi"
                                for i,r in enumerate(res)])
            return f"Dosen dengan bimbingan terbanyak:\n\n{daftar}"

    if re.search(r'(?:kategori|bidang)\s+(?:apa|mana)\s+.{0,10}(?:paling\s+banyak|populer)', q_norm):
        res = _run('''
            MATCH (k:Kategori)<-[:TERMASUK_KATEGORI]-(d:Dokumen)
            RETURN k.nama AS kategori, count(d) AS jml ORDER BY jml DESC LIMIT 5''')
        if res:
            daftar = '\n'.join([f"  {i+1}. **{r['kategori']}** — {r['jml']} skripsi"
                                for i,r in enumerate(res)])
            return f"Kategori terpopuler:\n\n{daftar}"

    return None  # Kalau nggak ada yang cocok, lempar ke LLM aja

# Pencarian Semantic & Graph Retrieval
def preprocess_query(query: str, embedding_model) -> dict:
    raw    = query.lower().strip()
    clean  = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', query)).strip()
    folded = clean.lower()
    emb    = embedding_model.encode([folded], convert_to_numpy=True)[0]
    return {'query_folded': folded, 'query_raw_folded': raw, 'query_embedding': emb}

def semantic_search(query_embedding, driver, top_k=5):
    def _ambil():
        with driver.session() as s:
            return s.run('''
                MATCH (d:Dokumen) WHERE d.embedding IS NOT NULL
                RETURN d.id AS doc_id, d.judul AS judul,
                       d.abstrak AS abstrak, d.embedding AS embedding
            ''').data()
    try:
        records = _retry(driver, _ambil)
    except Exception:
        return []
    if not records:
        return []

    doc_ids  = [r['doc_id']          for r in records]
    juduls   = [r['judul']           for r in records]
    abstraks = [r.get('abstrak','')  for r in records]
    embs     = np.array([r['embedding'] for r in records], dtype=np.float32)

    q_n    = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    e_n    = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    scores = (e_n @ q_n).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        doc_id = doc_ids[idx]
        def _meta(did=doc_id):
            with driver.session() as s:
                return s.run('''
                    MATCH (d:Dokumen {id: $did})
                    OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(m:Mahasiswa)
                    OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
                    OPTIONAL MATCH (d)-[:TERMASUK_KATEGORI]->(k:Kategori)
                    OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
                    RETURN m.nama AS penulis,
                           collect(DISTINCT dos.nama) AS pembimbing,
                           k.nama AS kategori,
                           collect(DISTINCT kk.nama) AS kata_kunci
                ''', did=did).single()
        try:
            meta = _retry(driver, _meta)
        except Exception:
            meta = None
        results.append({
            'doc_id'        : doc_id,
            'judul'         : juduls[idx]  or '',
            'abstrak'       : abstraks[idx] or '',
            'penulis'       : meta['penulis']    if meta else '',
            'pembimbing'    : meta['pembimbing'] if meta else [],
            'kategori'      : meta['kategori']   if meta else '',
            'kata_kunci'    : meta['kata_kunci'] if meta else [],
            'score_semantic': float(scores[idx]),
            'rank_semantic' : rank + 1,
            'sumber'        : 'semantic'
        })
    return apply_fix_encoding(results)

# Penelusuran Database Graph
def graph_traversal(query_folded, query_raw_folded, driver, top_k=5):
    SW = {
        'tampilkan', 'tampilkanlah', 'cari', 'carikan', 'carilah', 'skripsi', 'yang',
        'dengan', 'oleh', 'dalam', 'pada', 'untuk', 'atau', 'dan', 'siapa', 'apa', 'saja',
        'ada', 'apakah', 'bagaimana', 'berapa', 'mana', 'dibimbing', 'membahas',
        'menggunakan', 'topik', 'kategori', 'kata', 'kunci', 'metode', 'sekaligus',
        'satu', 'penelitian', 'tentang', 'termasuk', 'dosen', 'pembimbing', 'pertama',
        'kedua', 'daftar', 'disusun', 'milik', 'bernama', 'mahasiswa', 'referensi',
        'dokumen', 'keilmuan', 'bidang', 'diambil', 'diawasi', 'sering', 'masuk',
        'kategorikah', 'kategorinya', 'penulis', 'ditulis', 'utama', 'dari', 'adalah',
        'ini', 'itu', 'akan', 'dapat', 'juga', 'serta', 'yaitu', 'tersebut', 'berdasarkan',
        'beserta', 'judul', 'mau', 'ingin', 'kalau', 'jika', 'gimana', 'seperti',
        'pak', 'bu', 'bapak', 'ibu', 'prof',
        'dong', 'sih', 'deh', 'yuk', 'nih', 'loh', 'lah', 'kan', 'emang', 'memang',
        'kayak', 'gitu', 'gini', 'gak', 'tidak', 'bukan', 'gasih', 'gasik',
        'biasa', 'biasanya', 'umumnya', 'banyak', 'sedikit', 'lebih', 'kurang',
        'paling', 'sangat', 'punya', 'miliki', 'dimiliki', 'cocok', 'pas', 'sesuai',
        'rekomendasi', 'sarankan', 'saran', 'tolong', 'bisa', 'boleh', 'coba',
        'kasih', 'tahu', 'info', 'infokan', 'mengenai', 'soal', 'hal', 'ada',
        'ya', 'buat', 'neliti', 'ngebimbing', 'bahas', 'cariin', 'lihat',
        'ta', 'tugas', 'akhir', 'dospem', 'siapakah', 'bantuin', 'bantu',
        'mencarikan', 'tunjukin', 'liat', 'dimana', 'kapan', 'mengapa', 'kenapa',
        'judulnya', 'skripsinya', 'dosennya', 'pembimbingnya', 'punyanya'
    }
    GELAR = {
        's.t', 's.t.', 'st', 's.kom', 's.kom.', 'skom', 'm.kom', 'm.kom.', 'mkom',
        'm.t', 'm.t.', 'mt', 'm.mt', 'm.mt.', 'mmt', 'm.eng', 'm.eng.', 'meng',
        's.si', 's.si.', 'ssi', 'm.sc', 'm.sc.', 'msc', 'm.it', 'm.it.',
        'dr', 'dr.', 'prof', 'prof.', 'ir', 'ir.', 'drs', 'dra', 'dipl',
        's.kom.', 'm.kom.', 's.t.', 'm.t.'
    }

    keywords = [w for w in query_folded.split()
                if w not in SW and len(w) > 2]
    raw = query_raw_folded

    penanda_mhs = ['disusun oleh', 'milik', 'bernama',
                   'dari mahasiswa', 'punya mahasiswa']
    penanda_dos = ['dosen pembimbing', 'dibimbing oleh', 'bimbingan',
                   'diawasi oleh', 'pembimbing', 'oleh dosen']

    def ekstrak(teks, p_mulai, p_akhir):
        for p in p_mulai:
            idx = teks.find(p)
            if idx == -1:
                continue
            sisa = teks[idx+len(p):].strip()
            i_akhir = len(sisa)
            for pa in p_akhir:
                pos = sisa.find(pa)
                if pos != -1:
                    i_akhir = min(i_akhir, pos)
            return sisa[:i_akhir].strip()
        return ''

    seg_mhs = ekstrak(raw, penanda_mhs, penanda_dos)
    seg_dos = ekstrak(raw, penanda_dos, penanda_mhs)

    def kws_dari_segmen(seg):
        if not seg:
            return []
        return [w.strip(',.?!:;') for w in seg.split()
                if w.strip(',.?!:;') not in SW
                and w.strip(',.?!:;').lower().rstrip('.') not in GELAR
                and len(w.strip(',.?!:;')) > 2]

    kw_mhs = kws_dari_segmen(seg_mhs)
    kw_dos = kws_dari_segmen(seg_dos)

    ind = ['s.t.', 's.kom', 'm.kom', 'm.t.', 'm.mt', 'm.eng', 's.si', 'm.sc',
           'dr.', 'prof.', 'dibimbing oleh', 'milik', 'bernama', 'disusun oleh',
           'bimbingan', 'diawasi oleh', 'mulaab', 'firdaus', 'sigit',
           'hermawan', 'jauhari', 'mula', 'm.it']
    if any(p in raw for p in ind) and not kw_mhs and not kw_dos:
        fb = [w.strip(',.?!:;') for w in raw.split()
              if w.strip(',.?!:;') not in SW
              and w.strip(',.?!:;').lower().rstrip('.') not in GELAR
              and len(w.strip(',.?!:;')) > 2]
        kw_mhs = kw_dos = fb

    if not keywords and not kw_mhs and not kw_dos:
        return []

    semua_nama = list(set(kw_mhs + kw_dos))

    def _wh(field, kws):
        if not kws:
            return 'false'
        if 'nama' in field:
            return ' OR '.join(["replace(toLower(%s), \"'\", \"\") CONTAINS \"%s\"" % (field, kw.replace("'", "")) for kw in kws])
        return ' OR '.join(['toLower(%s) CONTAINS "%s"' % (field, kw) for kw in kws])

    cypher = f'''
        MATCH (d:Dokumen)
        WHERE ({_wh("d.judul", keywords)})
           OR ({_wh("d.abstrak", keywords)})
        WITH d
        OPTIONAL MATCH (d)-[:DITULIS_OLEH]->(m:Mahasiswa)
        OPTIONAL MATCH (d)-[:DIBIMBING_OLEH]->(dos:Dosen)
        OPTIONAL MATCH (d)-[:TERMASUK_KATEGORI]->(k:Kategori)
        OPTIONAL MATCH (d)-[:MEMILIKI_KATA_KUNCI]->(kk:Kata_Kunci)
        WITH d, m,
             collect(DISTINCT dos.nama)  AS pembimbing,
             collect(DISTINCT k.nama)    AS kategori_list,
             collect(DISTINCT kk.nama)   AS kk_list,
             collect(DISTINCT dos)       AS dos_nodes,
             collect(DISTINCT k)         AS k_nodes,
             collect(DISTINCT kk)        AS kk_nodes
        WHERE any(dos IN dos_nodes WHERE {_wh("dos.nama", semua_nama)})
           OR ({_wh("m.nama", semua_nama)})
           OR any(k  IN k_nodes  WHERE {_wh("k.nama",  keywords)})
           OR any(kk IN kk_nodes WHERE {_wh("kk.nama", keywords)})
           OR ({_wh("d.judul",   keywords)})
           OR ({_wh("d.abstrak", keywords)})
        RETURN DISTINCT d.id AS doc_id, d.judul AS judul, d.abstrak AS abstrak,
               m.nama AS penulis, pembimbing,
               kategori_list[0] AS kategori, kk_list AS kata_kunci
    '''
    try:
        raw_records = _retry(driver, lambda: driver.session().run(cypher).data())
        records = []
        if raw_records:
            seen_j = set()
            for r in raw_records:
                j = r.get('judul', '')
                if j not in seen_j:
                    seen_j.add(j)
                    records.append(r)
        records = apply_fix_encoding(records)
    except Exception:
        return []

    scored = []
    for r in records:
        dos_txt = ' '.join(r.get('pembimbing', [])).lower()
        mhs_txt = str(r.get('penulis', '')).lower()
        lain = ' '.join([str(r.get('judul', '')), str(r.get('kategori', '')),
                         ' '.join(r.get('kata_kunci', []))]).lower()

        skor_t = (sum(1 for kw in keywords if f' {kw} ' in f' {lain} ')
                  / max(len(keywords), 1)) if keywords else 0.0
        if kw_mhs or kw_dos:
            sm = (sum(1 for kw in kw_mhs if kw in mhs_txt) /
                  len(kw_mhs)) if kw_mhs else 0.0
            sd = (sum(1 for kw in kw_dos if kw in dos_txt) /
                  len(kw_dos)) if kw_dos else 0.0
            b = sum([1 for x in [kw_mhs, kw_dos] if x])
            sn = (sm+sd)/b if b else 0.0
            sf = (sn*2+skor_t)/3
        else:
            sf = skor_t
        if sf > 0:
            scored.append({**r, 'score_graph': sf})

    scored.sort(key=lambda x: x['score_graph'], reverse=True)
    return [{'doc_id': r.get('doc_id', ''), 'judul': r.get('judul', ''),
             'abstrak': r.get('abstrak', ''), 'penulis': r.get('penulis', ''),
             'pembimbing': r.get('pembimbing', []), 'kategori': r.get('kategori', ''),
             'kata_kunci': r.get('kata_kunci', []), 'score_graph': r['score_graph'],
             'rank_graph': rank+1, 'sumber': 'graph'}
            for rank, r in enumerate(scored[:top_k])]

def reciprocal_rank_fusion(sem, graph, k=60, top_k=5):
    scores = defaultdict(float)
    data   = {}
    for rank, doc in enumerate(sem, 1):
        scores[doc['doc_id']] += 1/(k+rank); data[doc['doc_id']] = doc
    for rank, doc in enumerate(graph, 1):
        scores[doc['doc_id']] += 1/(k+rank)
        if doc['doc_id'] not in data: data[doc['doc_id']] = doc
    sorted_items = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    result = []
    seen_judul = set()
    for did, rrf in sorted_items:
        d = data[did].copy()
        j = str(d.get('judul', '')).strip().lower()
        if j and j not in seen_judul:
            seen_judul.add(j)
            d['rrf_score'] = round(rrf,6)
            d['final_rank'] = len(result) + 1
            result.append(d)
        elif not j:
            # Tetap masukkan jika anehnya judul kosong
            d['rrf_score'] = round(rrf,6)
            d['final_rank'] = len(result) + 1
            result.append(d)
            
        if len(result) >= top_k:
            break
    return result

def assemble_context(docs, max_chars=400):
    parts = []
    for doc in docs:
        doc = apply_fix_encoding(doc)
        pemb = doc.get('pembimbing',[])
        pemb_str = (', '.join(p for p in pemb if p) if isinstance(pemb,list) else str(pemb))
        kk = doc.get('kata_kunci',[])
        kk_str = (', '.join(k for k in kk if k) if isinstance(kk,list) else str(kk))
        abstrak = str(doc.get('abstrak','') or '')[:max_chars]
        parts.append(
            f"[Dokumen {doc.get('final_rank','?')} | ID: {doc.get('doc_id','?')}]\n"
            f"Judul      : {str(doc.get('judul','') or '').strip()}\n"
            f"Penulis    : {str(doc.get('penulis','') or '').strip()}\n"
            f"Pembimbing : {pemb_str}\n"
            f"Kategori   : {str(doc.get('kategori','') or '').strip()}\n"
            f"Kata Kunci : {kk_str}\n"
            f"Abstrak    : {abstrak}")
    return '\n\n' + ('─'*60+'\n\n').join(parts)

# Aturan Ketat LLM Biar Nggak Halusinasi
SYSTEM_PROMPT = """Anda adalah asisten pencarian akademik Portal Tugas Akhir (PTA) UTM.

ATURAN MUTLAK — ZERO TOLERANCE UNTUK HALUSINASI:
1. Jawab HANYA berdasarkan informasi yang ada di KONTEKS. DILARANG KERAS mengarang, menebak, atau menyebutkan nama/judul/angka yang tidak tercantum dalam konteks yang diberikan.
2. DILARANG menyimpulkan informasi di luar dari konteks. Meskipun pertanyaan terkesan santai atau gaul (informal), tetap gunakan HANYA data dari konteks.
3. DILARANG menyebut nama dosen yang tidak ada di konteks meskipun Anda merasa "tahu" nama tersebut dari luar konteks.
4. DILARANG mengarang angka atau statistik yang tidak tertera di konteks.
5. Jika pertanyaan sama sekali tidak dapat dijawab berdasarkan konteks yang diberikan, Anda WAJIB menjawab dengan:
   "Maaf, informasi tersebut tidak ditemukan dalam basis data pencarian saat ini."
6. Sebutkan SEMUA dokumen dalam konteks yang relevan dengan pertanyaan, jangan hanya 1 atau 2.
7. Selalu sebutkan jumlah dokumen relevan secara eksplisit di awal kalimat.
8. Gunakan format daftar bernomor jika terdapat lebih dari satu dokumen.

CONTOH BENAR — jika data ada di konteks:
Pertanyaan: "Carikan dong skripsi tentang Identifikasi ALL"
[Konteks berisi 5 dokumen tentang ALL]
Jawaban: "Ditemukan 5 skripsi yang membahas Identifikasi ALL:
1. Identifikasi ALL Berdasarkan Fitur Bentuk — Anwar Fuadi
2. Identifikasi ALL Menggunakan Naive Bayes — Girindra Bimantara Putra
..."

CONTOH BENAR — jika data tidak ada di konteks:
Pertanyaan: "Ada gasih skripsi bimbingan pak elon musk?"
[Konteks tidak berisi dosen bernama elon musk]
Jawaban: "Maaf, informasi tersebut tidak ditemukan dalam basis data pencarian saat ini. Tidak ada skripsi bimbingan dosen terkait dalam dokumen yang ditemukan."

PERINGATAN: Memasukkan nama, angka, atau judul apa pun yang tidak ada di konteks adalah KESALAHAN FATAL. Pahami maksud pertanyaan user (baik baku maupun gaul), tapi pastikan jawabannya 100% bersumber dari konteks. JIKA TIDAK ADA JAWABAN DI DALAM KONTEKS, ANDA WAJIB MENJAWAB: "Informasi tidak ditemukan." JANGAN MENGARANG!"""

# Fungsi untuk Response Streaming LLM
def generate_answer_stream(query: str, context_str: str, model: str):
    if not context_str.strip() or len(context_str.strip()) < 30:
        yield ("Maaf, tidak ditemukan dokumen yang relevan dalam basis data PTA UTM.\n\n"
               "**Saran:**\n"
               "- Gunakan kata kunci topik yang lebih spesifik\n"
               "- Sebutkan nama mahasiswa atau dosen secara lengkap\n"
               "- Coba: *'Carikan skripsi tentang [topik]'* atau "
               "*'Apa topik skripsi milik [nama mahasiswa]?'*")
        return

    n = len([p for p in context_str.split('[Dokumen ') if p.strip()])
    user_msg = (
        f"Pertanyaan: \"{query}\"\n\n"
        f"KONTEKS DOKUMEN ({n} dokumen - HANYA GUNAKAN DATA INI):\n"
        f"{context_str}\n\n"
        f"INSTRUKSI KETAT: Jawablah pertanyaan \"{query}\" HANYA menggunakan referensi dokumen di atas. "
        f"Sebutkan semua {n} dokumen tersebut jika relevan. "
        f"JANGAN PERNAH menambahkan nama, judul, jumlah, atau info apa pun yang tidak tertera di dalam teks KONTEKS DOKUMEN."
    )
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 1024
            },
            stream=True,
            timeout=300
        )

        if response.status_code != 200:
            try: detail = response.json().get('error', {}).get('message', response.text[:200])
            except: detail = response.text[:200]
            yield f"⚠️ Groq API error ({response.status_code}): {detail}"
            return

        ada_isi = False
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        chunk = data['choices'][0]['delta'].get('content', '')
                        if chunk:
                            ada_isi = True
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        if not ada_isi:
            yield "⚠️ Tidak ada respons. Coba kirim ulang."

    except requests.exceptions.ConnectionError:
        yield "⚠️ Tidak bisa terhubung ke API Groq. Periksa koneksi internet."
    except requests.exceptions.Timeout:
        yield "⚠️ Timeout dari server Groq. Coba kirim ulang."
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# Halaman Utama Aplikasi
if "messages"       not in st.session_state: st.session_state.messages       = []
if "search_history" not in st.session_state: st.session_state.search_history = []

with st.sidebar:
    cl, cj = st.columns([1,3])
    with cl:
        try: st.image("./Logo UTM Terbaru.JPEG", width=48)
        except: st.write("🎓")
    with cj:
        st.markdown("<div style='display:flex;align-items:center;height:48px;'>"
                    "<span style='font-size:17px;font-weight:600;'>Portal Tugas Akhir TIF</span>"
                    "</div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("📝 Pencarian baru", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    st.markdown("<p style='color:#888;font-size:13px;margin:14px 0 4px'>Riwayat</p>",
                unsafe_allow_html=True)
    if st.session_state.search_history:
        for i, h in enumerate(reversed(st.session_state.search_history[-20:])):
            lbl = h[:38]+('...' if len(h)>38 else '')
            if st.button(lbl, key=f"h{i}", use_container_width=True):
                st.session_state['rerun_query'] = h; st.rerun()
    else:
        st.markdown("<p style='color:#888;font-size:13px;'>Belum ada riwayat pencarian.</p>",
                    unsafe_allow_html=True)
    st.markdown("---")
    ca,cb,cc = st.columns([1,3,1])
    with ca:
        st.markdown("<div style='width:32px;height:32px;border-radius:50%;"
                    "background:#1f77b4;color:white;display:flex;"
                    "align-items:center;justify-content:center;'>🎓</div>",
                    unsafe_allow_html=True)
    with cb:
        st.markdown(f"<div style='font-size:12px;line-height:1.2;'>"
                    f"<b>{MODEL_LLM}</b><br>"
                    f"<span style='color:#888;'>GraphRAG · RRF</span></div>",
                    unsafe_allow_html=True)
    with cc:
        with st.popover("⚙️"):
            st.markdown("**Panel Sistem**")
            st.write(f"LLM: `{MODEL_LLM}`")
            st.write(f"Top-K Retrieval: `{TOP_K_RETRIEVAL}`")
            st.write(f"Top-K LLM: `{TOP_K_KONTEKS_LLM}`")
            st.markdown("---")
            if st.button("🔄 Sambungkan Ulang DB", use_container_width=True):
                init_neo4j_driver.clear()
                st.success("Koneksi disegarkan.")
            if st.button("🗑️ Bersihkan Obrolan", use_container_width=True):
                st.session_state.messages = []; st.rerun()
            if st.button("🧹 Hapus Histori", use_container_width=True):
                st.session_state.search_history = []; st.rerun()

st.title("🎓 Portal Pencarian Tugas Akhir Cerdas")
st.markdown(
    "Sistem Temu Kembali Informasi Akademik Menggunakan *Knowledge Graph Retrieval-Augmented Generation* (GraphRAG)\n\n"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.session_state.pop('rerun_query', None) or \
        st.chat_input("Masukkan pertanyaan Anda...")

if query:
    if not st.session_state.search_history or st.session_state.search_history[-1] != query:
        st.session_state.search_history.append(query)
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            driver.verify_connectivity()
        except Exception:
            st.toast("Menyambungkan ulang database...", icon="🔄")
            init_neo4j_driver.clear()
            driver = init_neo4j_driver()

        # Langkah 1: Coba jawab pakai query database langsung (tanpa AI)
        jawaban = query_agregat_neo4j(query, driver)

        if jawaban:
            st.markdown(jawaban)
            full_response = jawaban
            ranked_full   = []
        else:
            # Langkah 2: Kalau nggak nemu, baru cari pakai pencarian pintar + AI
            with st.status("🔍 Melacak dokumen...", expanded=True) as status:
                prep    = preprocess_query(query, embed_model)
                sem_r   = semantic_search(prep['query_embedding'], driver, TOP_K_RETRIEVAL)
                graph_r = graph_traversal(prep['query_folded'], prep['query_raw_folded'],
                                          driver, TOP_K_RETRIEVAL)
                ranked_full = reciprocal_rank_fusion(sem_r, graph_r, k=60, top_k=TOP_K_RETRIEVAL)

                n_ke_llm   = min(len(ranked_full), TOP_K_KONTEKS_LLM)
                context_str = assemble_context(ranked_full[:n_ke_llm])
                if len(ranked_full) > 0:
                    status.update(
                        label=f"✅ Dokumen ditemukan, menyiapkan jawaban...",
                        state="complete", expanded=False
                    )
                else:
                    status.update(
                        label="❌ Tidak ditemukan dokumen yang relevan dengan pertanyaan ini.",
                        state="error", expanded=False
                    )

            ph = st.empty()
            full_response = ""
            for chunk in generate_answer_stream(query, context_str, MODEL_LLM):
                full_response += chunk
                ph.markdown(full_response + "▌")
            ph.markdown(full_response)

            if ranked_full:
                st.markdown("---")
                st.markdown("### 📚 Dokumen Referensi")
                for i, doc in enumerate(ranked_full, 1):
                    jl = doc.get('judul','Tanpa Judul')[:80]
                    if len(doc.get('judul','')) > 80: jl += '...'
                    with st.expander(f"[{i}] {jl}", expanded=False):
                        c1,c2 = st.columns(2)
                        with c1:
                            st.markdown("**ID**"); st.code(doc.get('doc_id','-'))
                            st.markdown("**Penulis**"); st.write(doc.get('penulis','-') or '-')
                            st.markdown("**Kategori**"); st.write(doc.get('kategori','-') or '-')
                        with c2:
                            st.markdown("**Pembimbing**")
                            pemb = doc.get('pembimbing',[])
                            if isinstance(pemb,list):
                                for p in pemb:
                                    if p: st.write(f"• {p}")
                            else: st.write(pemb or '-')
                            st.markdown("**Skor RRF**"); st.code(f"{doc.get('rrf_score',0):.6f}")
                            st.markdown("**Kata Kunci**")
                            kk = doc.get('kata_kunci',[])
                            st.write(', '.join(k for k in kk if k) if isinstance(kk,list) else (kk or '-'))
                        if doc.get('abstrak'):
                            st.markdown("**Abstrak**")
                            st.caption(str(doc.get('abstrak',''))[:400]+'...')

        st.session_state.messages.append({"role":"assistant","content":full_response})