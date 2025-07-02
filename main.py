import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import mediapipe as mp
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Tuple
from enum import Enum
import math
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Enhanced Virtual Accessory Try-On")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Accessory types
class AccessoryType(str, Enum):
    NECKLACE = "necklace"
    EARRING = "earring"
    CROWN = "crown"
    NOSE_ORNAMENT = "nose_ornament"
    BANGLE = "bangle"
    RING = "ring"

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def read_upload_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file and convert to OpenCV format"""
    contents = file.file.read()
    image = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format uploaded")
    return image

def remove_background_advanced(image: np.ndarray, method: str = "grabcut") -> np.ndarray:
    """
    Advanced background removal using multiple techniques
    
    Args:
        image: Input image in BGR format
        method: Background removal method ('grabcut', 'color_clustering', 'edge_based')
    
    Returns:
        Image with transparent background (BGRA format)
    """
    height, width = image.shape[:2]
    
    if method == "grabcut":
        # GrabCut algorithm for background removal
        mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle around the center (assuming accessory is centered)
        rect = (width//6, height//6, width*2//3, height*2//3)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Modify mask to get final foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Create BGRA image
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask2 * 255
        
    elif method == "color_clustering":
        # Color-based background removal using K-means clustering
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))
        
        # Identify background color (most common in corners)
        corner_colors = []
        corner_size = min(20, width//10, height//10)
        corners = [
            image[0:corner_size, 0:corner_size],  # top-left
            image[0:corner_size, -corner_size:],  # top-right
            image[-corner_size:, 0:corner_size],  # bottom-left
            image[-corner_size:, -corner_size:]   # bottom-right
        ]
        
        for corner in corners:
            corner_colors.extend(corner.reshape(-1, 3).tolist())
        
        # Find most common corner color
        corner_colors = np.array(corner_colors)
        kmeans_corner = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans_corner.fit(corner_colors)
        bg_color = kmeans_corner.cluster_centers_[0]
        
        # Create mask based on color similarity to background
        mask = np.zeros((height, width), dtype=np.uint8)
        color_diff = np.sqrt(np.sum((image - bg_color) ** 2, axis=2))
        threshold = np.mean(color_diff) * 0.7
        mask[color_diff > threshold] = 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Create BGRA image
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
        
    elif method == "edge_based":
        # Edge-based background removal
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contour (assuming it's the accessory)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Create BGRA image
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
    
    return result

def enhance_accessory_appearance(accessory_image: np.ndarray, lighting_factor: float = 1.2, contrast_factor: float = 1.1) -> np.ndarray:
    """
    Enhance the accessory appearance for more realistic look
    
    Args:
        accessory_image: Accessory image in BGRA format
        lighting_factor: Factor to adjust brightness
        contrast_factor: Factor to adjust contrast
    
    Returns:
        Enhanced accessory image
    """
    # Convert to PIL for better image processing
    pil_image = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(lighting_factor)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    # Slight sharpening for better detail
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)

def apply_realistic_shadow(base_image: np.ndarray, accessory_shape: np.ndarray, position: Tuple[int, int], shadow_intensity: float = 0.3) -> np.ndarray:
    """
    Apply realistic shadow effect under the accessory
    
    Args:
        base_image: Base image in BGR format
        accessory_shape: Binary mask of accessory shape
        position: Position where accessory is placed
        shadow_intensity: Intensity of shadow (0.0-1.0)
    
    Returns:
        Base image with shadow applied
    """
    height, width = base_image.shape[:2]
    shadow_offset_x, shadow_offset_y = 3, 5  # Shadow offset
    
    # Create shadow mask
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate shadow position
    shadow_x = position[0] + shadow_offset_x - accessory_shape.shape[1] // 2
    shadow_y = position[1] + shadow_offset_y - accessory_shape.shape[0] // 2
    
    # Ensure shadow is within image bounds
    if (shadow_x >= 0 and shadow_y >= 0 and 
        shadow_x + accessory_shape.shape[1] <= width and 
        shadow_y + accessory_shape.shape[0] <= height):
        
        shadow_mask[shadow_y:shadow_y + accessory_shape.shape[0], 
                    shadow_x:shadow_x + accessory_shape.shape[1]] = accessory_shape
        
        # Blur the shadow for realistic effect
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        
        # Apply shadow to base image
        shadow_factor = 1.0 - shadow_intensity
        for i in range(3):  # Apply to all BGR channels
            base_image[:, :, i] = base_image[:, :, i] * (1 - (shadow_mask / 255.0) * (1 - shadow_factor))
    
    return base_image

def detect_face_landmarks(image: np.ndarray) -> Optional[dict]:
    """Detect face landmarks using MediaPipe FaceMesh"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        height, width = image.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate key facial points with more precision
        left_ear = (int(landmarks[234].x * width)-5, int(landmarks[234].y * height)+16)
        right_ear = (int(landmarks[454].x * width)+5, int(landmarks[454].y * height)+16)
        forehead = (int(landmarks[10].x * width), int(landmarks[10].y * height))
        chin = (int(landmarks[152].x * width), int(landmarks[152].y * height))
        nose_tip = (int(landmarks[4].x * width), int(landmarks[4].y * height))
        nose_bridge = (int(landmarks[6].x * width), int(landmarks[6].y * height))
        
        # Additional points for better necklace positioning
        neck_base = (int(landmarks[152].x * width), int(landmarks[152].y * (height) + (landmarks[152].y - landmarks[10].y) * height * 0.15))
    
        neck_bottom = (int(landmarks[152].x * width),int(landmarks[152].y * height + (landmarks[152].y - landmarks[10].y) * height * 0.35))

        face_width = calculate_distance(left_ear, right_ear)
        face_height = calculate_distance(forehead, chin)
        
        return {
            'forehead': forehead,
            'nose_tip': nose_tip,
            'nose_bridge': nose_bridge,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'chin': chin,
            'neck_base': neck_bottom,
            'face_width': face_width,
            'face_height': face_height
        }

def detect_hand_landmarks(image: np.ndarray) -> Optional[dict]:
    """Detect hand landmarks using MediaPipe Hands"""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        
        height, width = image.shape[:2]
        hand_data = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            wrist = (int(landmarks[0].x * width), int(landmarks[0].y * height+50))
            
            index_tip = (int(landmarks[8].x * width), int(landmarks[8].y * height) )
            middle_tip = (int(landmarks[12].x * width), int(landmarks[12].y * height) )
            ring_tip = (int(landmarks[16].x * width), int(landmarks[16].y * height) )
            pinky_tip = (int(landmarks[20].x * width), int(landmarks[20].y * height))
            
            hand_width = calculate_distance(index_tip, pinky_tip)
            finger_length = calculate_distance(wrist, index_tip)
            index_tip2 = (int(landmarks[8].x * width), (int(landmarks[8].y * height)+int(finger_length/3.5)) )
            middle_tip2 = (int(landmarks[12].x * width), (int(landmarks[12].y * height)+int(finger_length/3.5)) )
            ring_tip2 = (int(landmarks[16].x * width), (int(landmarks[16].y * height)+int(finger_length/3.5)) )
            pinky_tip2 = (int(landmarks[20].x * width),( int(landmarks[20].y * height)+int(finger_length/3.5)))
            
            hand_data.append({
                'wrist': wrist,
                'finger_tips': [index_tip2, middle_tip2, ring_tip2, pinky_tip2],
                'hand_width': hand_width,
                'finger_length': finger_length
            })
        
        return hand_data[0] if hand_data else None

def calculate_perfect_scale(base_image: np.ndarray, accessory_img: np.ndarray, accessory_type: AccessoryType, landmarks: dict) -> float:
    """Calculate the perfect scale for the accessory based on facial/hand features"""
    base_height, base_width = base_image.shape[:2]
    accessory_height, accessory_width = accessory_img.shape[:2]
    
    if accessory_type == AccessoryType.NECKLACE:
        # Scale based on neck width (more realistic than face width)
        desired_width = landmarks['face_width'] * 0.83
        scale = desired_width / accessory_width
    
    elif accessory_type == AccessoryType.EARRING:
        # Scale based on ear size (more precise calculation)
        ear_size = landmarks['face_height'] * 0.08
        scale = ear_size / max(accessory_width, accessory_height)
    
    elif accessory_type == AccessoryType.CROWN:
        # Scale based on head circumference
        desired_width = landmarks['face_width'] * 1.1
        scale = desired_width / accessory_width
    
    elif accessory_type == AccessoryType.NOSE_ORNAMENT:
        # Scale based on nose bridge length
        desired_size = landmarks['face_height'] * 0.04
        scale = desired_size / max(accessory_width, accessory_height)
    
    elif accessory_type == AccessoryType.BANGLE:
        # Scale based on wrist circumference
        desired_size = landmarks['hand_width'] * 0.87
        scale = desired_size / max(accessory_width, accessory_height)
    
    elif accessory_type == AccessoryType.RING:
        # Scale based on finger diameter
        desired_size = landmarks['finger_length'] * 0.16
        scale = desired_size / max(accessory_width, accessory_height)
    
    # Ensure scale is reasonable
    return max(0.08, min(3.0, scale))



def preprocess_person_image(image: np.ndarray, accessory_type: str = None) -> np.ndarray:
    """
    Preprocess the person's image with different aspect ratios based on content type
    - Face/earring/nose/ring images: 3:4 ratio (297x394)
    - All other cases: return original image without processing
    
    Args:
        image: Input image in BGR format
        accessory_type: Type of accessory ('earring', 'nose', 'ring') if known
    
    Returns:
        Resized and cropped image in 3:4 ratio for face/accessory cases, 
        or original image for all other cases
    """
    # Only process if it's a face/accessory case
    if accessory_type in ['earring', 'nose', 'ring']:
        height, width = image.shape[:2]
        
        # Try to detect face landmarks
        face_landmarks = detect_face_landmarks(image)
        
        # Face/accessory processing - use 3:4 ratio (297x394)
        target_width, target_height = 297, 394
        target_aspect = target_width / target_height
        
        if face_landmarks:
            # Use face landmarks to determine ROI
            forehead = face_landmarks['forehead']
            chin = face_landmarks['chin']
            left_ear = face_landmarks['left_ear']
            right_ear = face_landmarks['right_ear']
            
            # Calculate face bounding box with some padding
            face_top = max(0, forehead[1] - (chin[1] - forehead[1]))  # 1 face height above forehead
            face_bottom = min(height, chin[1] + (chin[1] - forehead[1]) * 0.5)  # Half face height below chin
            face_left = max(0, left_ear[0] - (right_ear[0] - left_ear[0]) * 0.3)
            face_right = min(width, right_ear[0] + (right_ear[0] - left_ear[0]) * 0.3)
            
            # Calculate aspect ratio of face region
            face_width = face_right - face_left
            face_height = face_bottom - face_top
            face_aspect = face_width / face_height
            
            # Determine crop dimensions based on target aspect ratio (3:4 = 0.75)
            if face_aspect > target_aspect:
                # Face is wider than target - crop width
                crop_width = int(face_height * target_aspect)
                crop_left = max(0, (face_left + face_right) // 2 - crop_width // 2)
                crop_right = min(width, crop_left + crop_width)
                crop_top = face_top
                crop_bottom = face_bottom
            else:
                # Face is taller than target - crop height
                crop_height = int(face_width / target_aspect)
                crop_top = max(0, (face_top + face_bottom) // 2 - crop_height // 2)
                crop_bottom = min(height, crop_top + crop_height)
                crop_left = face_left
                crop_right = face_right
                
            # Perform the crop
            cropped = image[int(crop_top):int(crop_bottom), int(crop_left):int(crop_right)]
        else:
            # If no face detected but accessory type requires face ratio, use center crop
            original_aspect = width / height
            
            if original_aspect > target_aspect:
                # Original is wider - crop width
                new_width = int(height * target_aspect)
                left = (width - new_width) // 2
                right = left + new_width
                cropped = image[:, left:right]
            else:
                # Original is taller - crop height
                new_height = int(width / target_aspect)
                top = (height - new_height) // 2
                bottom = top + new_height
                cropped = image[top:bottom, :]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized
    else:
        # For all other cases, return original image
        return image
    
def overlay_accessory_realistic(
    base_image: np.ndarray,
    accessory_image: np.ndarray,
    position: tuple,
    scale: float,
    rotation: float = 0,
    blend_alpha: float = 0.95,
    add_shadow: bool = True,
    enhance_lighting: bool = True
) -> np.ndarray:
    """
    Enhanced overlay with realistic effects including shadows and lighting
    """
    # Remove background from accessory if it has one
    if accessory_image.shape[2] == 3:
        accessory_image = remove_background_advanced(accessory_image, method="color_clustering")
    
    # Enhance accessory appearance
    if enhance_lighting:
        accessory_image = enhance_accessory_appearance(accessory_image)
    
    # Convert base image to RGBA for easier processing
    base_pil = Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    
    # Scale accessory with high-quality resampling
    new_size = (int(accessory_pil.width * scale), int(accessory_pil.height * scale))
    accessory_pil = accessory_pil.resize(new_size, Image.LANCZOS)
    
    # Rotate accessory if needed
    if rotation != 0:
        accessory_pil = accessory_pil.rotate(rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    
    # Adjust alpha for blending
    r, g, b, a = accessory_pil.split()
    a = a.point(lambda p: int(p * blend_alpha))
    accessory_pil = Image.merge("RGBA", (r, g, b, a))
    
    # Calculate paste position (centered)
    x = position[0] - accessory_pil.width // 2
    y = position[1] - accessory_pil.height // 2
    
    # Apply shadow effect to base image first
    if add_shadow:
        # Create shadow mask from accessory alpha channel
        shadow_mask = np.array(a)
        base_image_with_shadow = apply_realistic_shadow(
            base_image.copy(), 
            shadow_mask, 
            position, 
            shadow_intensity=0.2
        )
        base_pil = Image.fromarray(cv2.cvtColor(base_image_with_shadow, cv2.COLOR_BGR2RGB)).convert("RGBA")
    
    # Create a copy of the base image to modify
    combined = base_pil.copy()
    
    # Paste using alpha compositing for smooth blending
    try:
        combined.alpha_composite(accessory_pil, (x, y))
    except ValueError:
        # Handle case where accessory extends beyond image bounds
        # Crop accessory to fit within image
        img_width, img_height = combined.size
        acc_width, acc_height = accessory_pil.size
        
        # Calculate crop coordinates
        crop_left = max(0, -x)
        crop_top = max(0, -y)
        crop_right = min(acc_width, img_width - x)
        crop_bottom = min(acc_height, img_height - y)
        
        if crop_right > crop_left and crop_bottom > crop_top:
            cropped_accessory = accessory_pil.crop((crop_left, crop_top, crop_right, crop_bottom))
            paste_x = max(0, x)
            paste_y = max(0, y)
            combined.alpha_composite(cropped_accessory, (paste_x, paste_y))
    
    # Convert back to OpenCV format (BGR)
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGR)

# ... (keep all the existing imports and helper functions above) ...

@app.post("/try-on/necklace")
async def try_necklace(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True
):
    """Endpoint specifically for necklace try-on"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image)
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_face_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.NECKLACE, landmarks)
        position = landmarks['neck_base']
        
        result_image = overlay_accessory_realistic(
            image, accessory_img, position, scale,
            add_shadow=add_shadow, enhance_lighting=enhance_lighting
        )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/try-on/earring")
async def try_earring(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True,
    side: str = "both"  # 'left', 'right', or 'both'
):
    """Endpoint specifically for earring try-on with side selection"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image, accessory_type="earring") 
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_face_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.EARRING, landmarks)
        result_image = image.copy()  # Start with original image
        
        # Process left ear if requested
        if side in ["left", "both"]:
            position = landmarks['left_ear']
            rotation = -5  # Slight rotation for natural look
            result_image = overlay_accessory_realistic(
                result_image, accessory_img, position, scale, rotation=rotation,
                add_shadow=add_shadow, enhance_lighting=enhance_lighting
            )
        
        # Process right ear if requested
        if side in ["right", "both"]:
            position = landmarks['right_ear']
            rotation = 5  # Slight rotation for natural look
            # Flip the accessory image for right ear (mirror effect)
            flipped_accessory = cv2.flip(accessory_img, 1)
            result_image = overlay_accessory_realistic(
                result_image, flipped_accessory, position, scale, rotation=rotation,
                add_shadow=add_shadow, enhance_lighting=enhance_lighting
            )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/try-on/crown")
async def try_crown(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True
):
    """Endpoint specifically for crown try-on"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image)
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_face_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.CROWN, landmarks)
        forehead_y = landmarks['forehead'][1]
        chin_y = landmarks['chin'][1]
        face_vertical_span = chin_y - forehead_y
        position = (
            landmarks['forehead'][0],
            forehead_y - int(face_vertical_span * 0.25)  # 25% above forehead
        )
        
        result_image = overlay_accessory_realistic(
            image, accessory_img, position, scale,
            add_shadow=add_shadow, enhance_lighting=enhance_lighting
        )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/try-on/nose-ornament")
async def try_nose_ornament(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True,
    side: str = "left"  # 'left' or 'right'
):
    """Endpoint specifically for nose ornament try-on with side adjustment"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image, accessory_type="nose")
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_face_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.NOSE_ORNAMENT, landmarks)
        
        # Get nose tip position
        nose_tip = landmarks['nose_tip']
        width, height = image.shape[1], image.shape[0]
        
        # Calculate offset based on face width (5% of face width)
        offset = int(landmarks['face_width'] * 0.13)
        
        # Adjust position based on selected side
        if side == "left":
            position = (nose_tip[0] - offset, nose_tip[1])
        else:  # right side
            position = (nose_tip[0] + offset, nose_tip[1])
        
        result_image = overlay_accessory_realistic(
            image, accessory_img, position, scale,
            add_shadow=add_shadow, enhance_lighting=enhance_lighting
        )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/try-on/bangle")
async def try_bangle(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True
):
    """Endpoint specifically for bangle try-on"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image)
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_hand_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No hands detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.BANGLE, landmarks)
        position = landmarks['wrist']
        
        result_image = overlay_accessory_realistic(
            image, accessory_img, position, scale,
            add_shadow=add_shadow, enhance_lighting=enhance_lighting
        )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/try-on/ring")
async def try_ring(
    file: UploadFile = File(...),
    accessory_image_upload: UploadFile = File(...),
    background_removal_method: str = "color_clustering",
    add_shadow: bool = True,
    enhance_lighting: bool = True,
    finger: str = "index"  # 'index', 'middle', 'ring', 'pinky'
):
    """Endpoint specifically for ring try-on"""
    try:
        image = read_upload_file(file)
        # Preprocess person image to 3:4 aspect ratio
        image = preprocess_person_image(image, accessory_type="ring")
        accessory_img = read_upload_file(accessory_image_upload)
        
        landmarks = detect_hand_landmarks(image)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No hands detected in the image")
        
        scale = calculate_perfect_scale(image, accessory_img, AccessoryType.RING, landmarks)
        
        # Determine finger position
        finger_index = {"index": 0, "middle": 1, "ring": 2, "pinky": 3}.get(finger, 0)
        position = landmarks['finger_tips'][finger_index]
        
        result_image = overlay_accessory_realistic(
            image, accessory_img, position, scale,
            add_shadow=add_shadow, enhance_lighting=enhance_lighting
        )
        
        _, encoded_image = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Keep the remove-background endpoint as is
@app.post("/remove-background")
async def remove_background_endpoint(
    file: UploadFile = File(...),
    method: str = "color_clustering"
):
    """Endpoint to remove background from accessory images"""
    try:
        image = read_upload_file(file)
        result = remove_background_advanced(image, method=method)
        
        _, encoded_image = cv2.imencode(".png", result)
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/remove-background")
async def remove_background_endpoint(
    file: UploadFile = File(...),
    method: str = "color_clustering"
):
    """
    Endpoint to remove background from accessory images
    """
    try:
        image = read_upload_file(file)
        result = remove_background_advanced(image, method=method)
        
        # Convert BGRA to PNG format for transparency
        _, encoded_image = cv2.imencode(".png", result)
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
