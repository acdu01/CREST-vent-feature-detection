import pycolmap
model_path = "6_colmap/1_colmap_output/0"
reconstruction = pycolmap.Reconstruction(model_path)

print(reconstruction.summary())
# iterate:
for image_id, image in reconstruction.images.items():
    print(image_id, image)

for point_id, point in reconstruction.points3D.items():
    print(point_id, point)

for cam_id, cam in reconstruction.cameras.items():
    print(cam_id, cam)
