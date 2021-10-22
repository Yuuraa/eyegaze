# Yeong-oo seonbae
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import Process
def get_new_size(height, width, minimum_size=300):
    if min(height, width) <= minimum_size:
        return height, width
    if height < width:
        resize_ratio = minimum_size / height
        new_height = minimum_size
        new_width = int(width * resize_ratio)
    else:
        resize_ratio = minimum_size / width
        new_height = int(height * resize_ratio)
        new_width = minimum_size
    return new_height, new_width
def resize_image(image_root, segmentation_map_root,
                 resized_image_root, resized_segmentation_map_root, cls, minimum_size):
    image_list = sorted(os.listdir(os.path.join(image_root, cls)))
    os.makedirs(os.path.join(resized_image_root, cls), exist_ok=True)
    os.makedirs(os.path.join(resized_segmentation_map_root, cls), exist_ok=True)
    for image_name in image_list:
        try:
            segmentation_map_name = image_name.split('.')[0] + '.png'
            image_path = os.path.join(image_root, cls, image_name)
            segmentation_map_path = os.path.join(segmentation_map_root, cls, segmentation_map_name)
            resized_image_path = os.path.join(resized_image_root, cls, image_name)
            resized_segmentation_map_path = os.path.join(resized_segmentation_map_root, cls, segmentation_map_name)
            original_image = Image.open(image_path)
            original_segmentation_map = Image.open(segmentation_map_path)
            assert original_image.size == original_segmentation_map.size
            width, height = original_image.size
            new_height, new_width = get_new_size(height, width, minimum_size)
            resized_image = transforms.Resize(size=minimum_size)(original_image)
            resized_segmentation_map = transforms.Resize(size=minimum_size,
                                                         interpolation=transforms.InterpolationMode.NEAREST)(
                original_segmentation_map)
            assert (new_width, new_height) == resized_image.size
            assert original_image.size == original_segmentation_map.size
            assert len(set(np.unique(resized_segmentation_map)) - {0}) == 1
            resized_image.save(resized_image_path)
            resized_segmentation_map.save(resized_segmentation_map_path)
        except:
            print(cls, image_name)
def main():
    image_root = '/mnt/disk1/nyw/disk1/preprocessed_cubox/segmentation/first_dataset/original_size/images_v2'
    segmentation_map_root = '/mnt/disk1/nyw/disk1/preprocessed_cubox/segmentation/first_dataset/original_size/seg_map'
    resized_image_root = '/mnt/disk1/nyw/disk1/preprocessed_cubox/segmentation/first_dataset/resize/300/images'
    resized_segmentation_map_root = '/mnt/disk1/nyw/disk1/preprocessed_cubox/segmentation/first_dataset/resize/300/seg_map'
    minimum_size = 300
    proc_list = []
    maximum_proc = 12
    class_list = sorted(os.listdir(image_root))
    for cls in tqdm(class_list):
        if len(proc_list) >= maximum_proc:
            cur_proc = proc_list.pop(0)
            cur_proc.join()
        proc = Process(target=resize_image, args=(
            image_root, segmentation_map_root, resized_image_root, resized_segmentation_map_root, cls, minimum_size))
        proc.start()
        proc_list.append(proc)
    while len(proc_list) != 0:
        cur_proc = proc_list.pop(0)
        cur_proc.join()
if __name__ == "__main__":
    main()