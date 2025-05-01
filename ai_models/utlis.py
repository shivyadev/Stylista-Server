import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import re
from django.conf import settings
import io
import boto3
from uuid import uuid4
import mimetypes
import requests
import tempfile


def predict_category(image_bytes, category_model, category_mapping):
    try:
        # Ensure the image is a PIL Image object
        # if isinstance(image_data, (bytes, bytearray)):
        #     img = Image.open(io.BytesIO(image_data)).convert("RGB")
        # elif isinstance(image_data, Image.Image):
        #     img = image_data.convert("RGB")
        # else:
        #     raise ValueError("Unsupported image format. Provide a PIL Image or image bytes.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # img_tensor = transform(img).unsqueeze(0).to(device)

        # feature_extractors = [
        #     settings.MODEL_REGISTRY["resnet50"],
        #     settings.MODEL_REGISTRY["efficientnet"]
        # ]

        # for extractor in feature_extractors:
        #     extractor.fc = torch.nn.Identity()
        #     extractor = extractor.to(device)
        #     extractor.eval()

        # combined_feature = []
        # with torch.no_grad():
        #     for extractor in feature_extractors:
        #         feature = extractor(img_tensor).float().cpu().numpy().flatten()
        #         combined_feature.extend(feature)

        print("Here")

        # # Save a copy for each request
        # resnet_image_data = image_data
        # efficientnet_image_data = image_data

        resnet_files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
        efficientnet_files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}

        efficient_response = requests.post(
            "http://127.0.0.1:8001/embed", 
            files=efficientnet_files,
        )

        resnet_response = requests.post(
            "http://127.0.0.1:8000/embed", 
            files=resnet_files
        )

        resnet_feature = resnet_response.json()["embeddings"]
        efficient_feature = efficient_response.json()["embeddings"]

        print(len(resnet_feature), len(efficient_feature))

        combined_feature = np.concatenate([resnet_feature, efficient_feature])

        print("combined_feature", combined_feature)

        feature_tensor = torch.tensor(combined_feature, dtype=torch.float32).unsqueeze(0).to(device)

        category_model.eval()
        with torch.no_grad():
            category_id = category_model(feature_tensor).argmax(dim=1).item()

        inv_category_mapping = {v: k for k, v in category_mapping.items()}
        predicted_category = inv_category_mapping.get(category_id, "Unknown")

        return predicted_category, combined_feature

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None


def extract_cloth_colors_with_segmentation(image_bytes, num_colors=5):

    if isinstance(image_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif isinstance(image_bytes, Image.Image):
        img = image_bytes.convert("RGB")
    else:
        raise ValueError("Unsupported image format. Provide a PIL Image or image bytes.")

    img_np = np.array(img)
    
    clothing_classes = [1, 27, 28, 32, 33, 38, 44, 46, 78, 79]
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    
    maskrcnn_files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}

    maskrcnn_response = requests.post(
        "http://127.0.0.1:8002/detect", 
        files=maskrcnn_files,
    )
    
    scores = np.array(maskrcnn_response.json()["scores"])
    masks = np.array(maskrcnn_response.json()["masks"])
    labels = np.array(maskrcnn_response.json()["labels"])
    
    high_confidence_idxs = np.where(scores > 0.7)[0]
    
    for idx in high_confidence_idxs:
        if labels[idx] in clothing_classes:
            binary_mask = masks[idx, 0] > 0.5
            mask = np.logical_or(mask, binary_mask)
    
    cloth_only = img_np.copy()
    cloth_only[~mask] = [0, 0, 0]
    
    non_zero_pixels = cloth_only[np.any(cloth_only != [0, 0, 0], axis=-1)]
    
    if len(non_zero_pixels) == 0:
        return [], cloth_only
    
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(non_zero_pixels)
    
    colors = kmeans.cluster_centers_
    
    colors = colors.astype(int)
    
    labels = kmeans.labels_
    count = Counter(labels)
    
    total = sum(count.values())
    color_percentages = [(colors[i], count[i]/total*100) for i in count.keys()]
    
    color_percentages.sort(key=lambda x: x[1], reverse=True)
    
    return color_percentages, cloth_only

def find_closest_color(csv_path, rgb_input, column_name='rgb'):

    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert RGB string format "(r,g,b)" to numeric tuples
    def extract_rgb(rgb_value):
        # Handle non-string values
        if pd.isna(rgb_value):
            return None
            
        # Convert to string if not already
        rgb_str = str(rgb_value)
        
        # Extract numbers from the string using regex
        matches = re.findall(r'\d+', rgb_str)
        if len(matches) >= 3:
            return (int(matches[0]), int(matches[1]), int(matches[2]))
        return None
    
    # Apply the extraction function to the RGB column
    df['rgb_tuple'] = df[column_name].apply(extract_rgb)
    
    # Calculate Euclidean distance between input RGB and each RGB in the dataframe
    def calculate_distance(rgb_tuple):
        if rgb_tuple is None:
            return float('inf')  # Return infinity for invalid RGB values
        r1, g1, b1 = rgb_input
        r2, g2, b2 = rgb_tuple
        return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    
    # Apply distance calculation
    df['color_distance'] = df['rgb_tuple'].apply(calculate_distance)
    
    # Sort by distance (closest first)
    result = df.sort_values('color_distance')
    
    return result

def extract_compatible_clothes(dataset, matches, cloth_type, gender, usage):

    dataset = pd.read_csv(dataset)
    
    filtered_comp = matches.loc[(matches["Gender"] == gender) & (matches["Usage"] == usage)]
    filtered_df = dataset.loc[(dataset["gender"] == gender) & (dataset["usage"] == usage)]
    
    columns = [cols for cols in filtered_comp.columns if (cloth_type not in cols) and ("RGB" not in cols)]
    columns = columns[:-5]

    clothes = []
        
    for _, outfit in filtered_comp.iterrows():
        cloth_id = []    
        notfound = False
        
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):    
                article_type = outfit[columns[i]]
                article_color = outfit[columns[i+1]]
    
                if article_type != article_type:
                    continue
                
                sample = filtered_df.loc[(filtered_df["articleType"] == article_type) & (filtered_df["baseColour"] == article_color)]
    
                if not sample.empty:
                    cloth_id.append(sample.sample(1).id.values[0])
                else:
                    continue

        if len(cloth_id) != 0:  
            clothes.append(cloth_id)
        
    return clothes

def upload_image(image):
    imagename = f"{uuid4()}_{image.name}"

    content_type, _ = mimetypes.guess_type(image.name)
    if content_type is None:
        content_type = "application/octet-stream"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME,
    )

    try:
        with image.open("rb") as file_obj:
            s3.upload_fileobj(file_obj, settings.AWS_STORAGE_BUCKET_NAME, imagename, ExtraArgs={"ContentType": content_type})

        file_url = f"{settings.AWS_S3_CUSTOM_DOMAIN}/{imagename}"
        return file_url

    except Exception as e:
        raise Exception("Could not upload the file", e)


def download_file_from_url(url, as_numpy=False, map_location="cpu"):
    response = requests.get(url)
    response.raise_for_status()

    if as_numpy:
        return np.load(io.BytesIO(response.content), allow_pickle=True).item()
    else:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            return torch.load(temp_file.name, map_location=map_location)
        
        
def load_npy_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True).item()