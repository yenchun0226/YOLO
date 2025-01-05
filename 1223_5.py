import torch
import cv2
import matplotlib.pyplot as plt
import os
import glob

# 載入 YOLOv5 模型
def load_yolov5_model():
    print("Loading YOLOv5 model...")
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 載入圖像
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 視覺化檢測結果
def visualize_detection(results, original_image):
    print("Visualizing detection results...")
    results.render()  # 在檢測結果上繪製邊框

    # 顯示渲染後的圖像
    rendered_image = results.ims[0]  # 獲取渲染後的圖像
    cv2.imshow("Detection Results", rendered_image)
    cv2.waitKey(0)  # 按任意鍵關閉窗口
    cv2.destroyAllWindows()

# 過濾低置信度的檢測框
def filter_detections(results, confidence_threshold):
    df = results.pandas().xyxy[0]  # 轉換為 pandas DataFrame
    filtered_df = df[df['confidence'] >= confidence_threshold]
    return filtered_df

# 主程序
if __name__ == "__main__":
    try:
        # Step 1: 載入 YOLOv5 模型
        model = load_yolov5_model()

        # Step 2: 設置圖片資料夾路徑
        image_folder = r"D:\images\input"  # 替換為包含圖片的資料夾路徑

        # Step 3: 獲取所有圖片路徑
        image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

        print(f"Found {len(image_paths)} images to process.")

        # Step 4: 批量處理圖片
        for image_path in image_paths:
            try:
                # 載入圖片
                original_image, image_rgb = load_image(image_path)

                # 運行檢測
                print(f"Running object detection on {image_path}...")
                results = model(image_rgb)  # 不使用 'conf' 參數

                # 過濾檢測結果
                confidence_threshold = 0.25  # 設置置信度閾值
                filtered_results = filter_detections(results, confidence_threshold)

                # 列印檢測結果
                detection_count = len(filtered_results)
                print(f"Number of detections in {os.path.basename(image_path)}: {detection_count}")

                if detection_count > 0:
                    print("Detection Results:")
                    print(filtered_results)  # 列印過濾後的檢測結果

                    # 視覺化檢測結果
                    visualize_detection(results, original_image)
                else:
                    print(f"No objects detected in {os.path.basename(image_path)}.")
            except Exception as e:
                print(f"An error occurred while processing {image_path}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

