# import cv2
# import os

# def images_to_video(image_folder, output_path, fps=30):
#     """
#     将指定文件夹中的图像按文件名排序后写入 mp4 视频
    
#     :param image_folder: 存放图像的文件夹路径
#     :param output_path: 输出视频文件的路径（含 .mp4 后缀）
#     :param fps: 视频帧率（frames per second）
#     """
#     # 1. 获取所有图像文件并按文件名排序（假设所有图像都是 .png 或 .jpg）
#     images = [img for img in os.listdir(image_folder) 
#               if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     images.sort()

#     if not images:
#         print("No images found in the folder.")
#         return

#     # 2. 读取第一张图像以获取视频尺寸
#     first_image_path = os.path.join(image_folder, images[0])
#     frame = cv2.imread(first_image_path)
#     if frame is None:
#         print(f"Failed to read the first image: {first_image_path}")
#         return
    
#     height, width, _ = frame.shape
    
#     # 3. 创建 VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 *'XVID'
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # 4. 按顺序写入每帧
#     for img_name in images:
#         img_path = os.path.join(image_folder, img_name)
#         frame = cv2.imread(img_path)
#         if frame is not None:
#             out.write(frame)
#         else:
#             print(f"Warning: Failed to read {img_path}, skipping this frame.")

#     # 5. 释放资源
#     out.release()
#     print(f"Video saved to {output_path}")


# if __name__ == "__main__":
#     folder_path = "/cpfs01/user/xiaozeqi/diffusionforcing/paper_material/demo_quality/rain_compare_gen"  # 替换为实际图像文件夹路径
#     output_video = "/cpfs01/user/xiaozeqi/diffusionforcing/paper_material/demo_quality/rain_compare_gen.mp4"               # 输出文件名
#     fps = 10                                  # 设置帧率

#     images_to_video(folder_path, output_video, fps)

import cv2
import os
import subprocess

def images_to_video(image_folder, output_path, fps=30):
    """
    将指定文件夹中的图像按文件名排序后写入 mp4 视频（H.264 编码，适用于 HTML 播放）
    
    :param image_folder: 存放图像的文件夹路径
    :param output_path: 输出视频文件的路径（含 .mp4 后缀）
    :param fps: 视频帧率（frames per second）
    """
    # 1. 获取所有图像文件并按文件名排序（假设所有图像都是 .png 或 .jpg）
    images = [img for img in os.listdir(image_folder) 
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if not images:
        print("No images found in the folder.")
        return

    # 2. 读取第一张图像以获取视频尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the first image: {first_image_path}")
        return
    
    height, width, _ = frame.shape
    
    # 3. 创建 VideoWriter（使用 H.264 编码）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # H.264 编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 按顺序写入每帧
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Failed to read {img_path}, skipping this frame.")

    # 5. 释放资源
    out.release()
    print(f"Video saved to {output_path} (H.264 format)")

    output_mp4 = output_path[:-3] + 'mp4'
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", output_mp4
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"MP4 video saved: {output_mp4}")

        # 6. Optionally, remove the intermediate AVI file
        os.remove(output_path)
    except Exception as e:
        print(f"Error during MP4 conversion: {e}")
        
if __name__ == "__main__":
    folder_path = "/cpfs01/user/xiaozeqi/diffusion-forcing-transformer/paper_material/compare/wo_mem_gen"  # 替换为实际图像文件夹路径
    output_video = "/cpfs01/user/xiaozeqi/diffusion-forcing-transformer/paper_material/compare/wo_mem_gen.avi"  # 输出文件名
    fps = 10  # 设置帧率

    images_to_video(folder_path, output_video, fps)
