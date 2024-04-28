import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware  # Tambahkan ini

# OCR Stater
import easyocr
import cv2
import numpy as np
import time

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Direktori tempat Anda menyimpan gambar-gambar
IMAGE_DIRECTORY = "image"

# Menggunakan StaticFiles untuk menyajikan file statis dari folder image
app = FastAPI()
app.mount("/image", StaticFiles(directory=IMAGE_DIRECTORY), name="image")

# Set CORS policies (disable for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_random_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    random_name = str(uuid.uuid4())
    return f"{random_name}.{ext}"

def generate_random_folder():
    random_dir = os.path.join('image', str(uuid.uuid4()))
    return random_dir


# Fungsi untuk memuat dan mengoptimalkan gambar
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Misalnya, mengubah gambar menjadi skala abu-abu untuk mengurangi ukuran dan kompleksitas
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bisa juga dilakukan proses prapemrosesan lainnya, seperti pengurangan noise atau peningkatan kontras
    return gray_image

# Convert NumPy int64 to regular Python integers
def convert_np_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

# Fungsi untuk melakukan OCR pada gambar yang telah diolah
def perform_ocr(image):
    # Membuat objek Reader dengan dukungan GPU
    reader = easyocr.Reader(['en'], gpu=True)
    # Membaca teks dari gambar
    start_time = time.time()
    results = reader.readtext(image)
    end_time = time.time()
    # Menghitung waktu yang dibutuhkan
    processing_time_ms = (end_time - start_time) * 1000
    return results, processing_time_ms

def get_name(hasil):
    try:
        index_nama = hasil[int(hasil.index("Nama")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_gender(hasil):
    try:
        index_nama = hasil[int(hasil.index("Jenis Kelamin")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_blood(hasil):
    try:
        index_nama = hasil[int(hasil.index("Gol. Darah")) + 1]
        return index_nama
    except ValueError:
        return "null"


def get_address(hasil):
    try:
        index_nama = hasil[int(hasil.index("Alamat")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_rtrw(hasil):
    try:
        index_nama = hasil[int(hasil.index("RTIRW")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_desa(hasil):
    try:
        index_nama = hasil[int(hasil.index("KellDesa")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_kecamatan(hasil):
    try:
        index_nama = hasil[int(hasil.index("Kecamatan")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_religion(hasil):
    try:
        index_nama = hasil[int(hasil.index("Agama")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_kawin(hasil):
    try:
        index_nama = hasil[int(hasil.index("Status Perkawinan")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_work(hasil):
    try:
        index_nama = hasil[int(hasil.index("Pekerjaan")) + 1]
        return index_nama
    except ValueError:
        return "null"

def get_warga(hasil):
    try:
        index_nama = hasil[int(hasil.index("Kewarganegaraan")) + 1]
        return index_nama
    except ValueError:
        return "null"

@app.get("/", status_code=200)
async def root():
    return {
        "success": True,
        "message": "System Has Been Deployment",
        "copyright": 'Created By Hai Ojan Development'
    }

@app.post("/uploadfile/")
async def uploaded_file(
    image: Annotated[UploadFile, File()],
    token: Annotated[str, Form()],
    request: Request
):
    valid_results = 0
    hasil = []

    if token != "fd87e9ff-0215-429c-82ae-60606b10ada5":
        return {
            "status": False,
            "message": "Token Invalid, Try Again"
        }
    
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="File format not supported. Only JPG, JPEG, and PNG files are allowed.")
    
    # Generate random filename and construct path
    random_filename = generate_random_filename(image.filename)
    upload_dir = generate_random_folder()
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, random_filename)

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Get base URL from request
    base_url = str(request.base_url)
    
    # Construct URL
    url = f"{base_url}{file_path}"

    image_cv2 = cv2.imread(file_path)

    # Memuat dan mengoptimalkan gambar
    processed_image = load_and_preprocess_image(file_path)

    # Melakukan OCR pada gambar yang telah diolah
    results, processing_time_ms = perform_ocr(processed_image)

    # Memfilter hasil OCR yang hanya mengandung angka dan panjangnya 16 digit
    for result in results:
        text = result[1]
        hasil.append(text)

        # Menghapus spasi dan karakter non-angka
        digit_text = ''.join(filter(str.isdigit, text))
        # Memeriksa apakah panjangnya 16 digit
        if len(digit_text) == 16:
            valid_results = digit_text

    # Grayscale face detection
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles, the 'area_treshold' value was determined empirically
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area_treshold = 4000

    # Draw rectangles on the original image
    for c in cnts:
        if cv2.contourArea(c) > area_treshold:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cropped_face = image_cv2[y:y+h, x:x+w]


    # Path untuk direktori Crop
    crop_dir = os.path.join(upload_dir ,'profile')

    # Pastikan direktori Crop sudah ada atau buat jika belum ada
    os.makedirs(crop_dir, exist_ok=True)
    
    # Convert BGR to RGB
    increased_resolution_cropped_face = cv2.resize(cropped_face, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Simpan gambar crop
    cropped_image_path = os.path.join(crop_dir, 'result.jpg')

    # # Simpan gambar dengan menggunakan cv2.imwrite()
    cv2.imwrite(cropped_image_path, increased_resolution_cropped_face)

    profile = f"{base_url}{cropped_image_path}"
    
    return {
        "success": True,
        "message": "Success Get Data KTP",
        "data": {
            'province' : hasil[0],
            'city' : hasil[1],
            'nik': str(valid_results),
            'nama_lengkap': get_name(hasil),
            'jenis_kelamin' : get_gender(hasil),
            # 'golongan_darah' : get_blood(hasil),
            'alamat' : get_address(hasil),
            'rtrw' : get_rtrw(hasil),
            'desa' : get_desa(hasil),
            'kecamatan' : get_kecamatan(hasil),
            'agama' : get_religion(hasil),
            'status_kawin' : get_kawin(hasil),
            'pekerjaan' : get_work(hasil),
            'kewarganegaraan' : get_kawin(hasil),
            
            "image_type": image.content_type,
            "profile_image": profile,
            "preview_ktp": url,
        }
    }