#AI 3D Scanner
import keyboard
import time
print("------------------------")
print("AI 3D Scanner")
print("[ Press Space To Start ]")
print("------------------------")
print("")
keyboard.wait("space")
keyboard.send("backspace")
time.sleep(0.3)
print("Loading Libaries...")
from PIL import Image
import torch
import glob
import os
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
print("Setting AI model...")
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
name = input("Enter image name: ")
name = str(name+".jpg")
quality=99
while quality>10:
    quality = int(input("Quality ( Max 10 ): "))
print("Importing Image...")
for k in glob.glob('Images/'+name):
    image = Image.open(k)
    kn, kext = os.path.splitext(k)
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)
print("Predicting Depth...")
inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))
import numpy as np
import open3d as o3d
print("Creating Pointcloud...")
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype('uint8')
image = np.array(image)
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd = pcd.select_by_index(ind)
print("Creating Mesh...")
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=quality, n_threads=1)[0]
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))
o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)
print("[ Succsess ]")
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
while True:
    continue
