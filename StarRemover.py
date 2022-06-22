import os
import sys

import matplotlib.pyplot
import numpy as np
import skimage.draw
import tkinter as tk
import math

from PIL import Image, ImageDraw, ImageTk
from skimage.restoration import inpaint
from skimage.transform import rescale
from skimage.io._plugins.pil_plugin import ndarray_to_pil
from scipy import ndimage
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename
from collections import defaultdict
from threading import *
# Root directory of the project
from datetime import datetime

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

class StarConfig(Config):
    """Configuration for Star dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Star"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Star

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    # Set anchor values accordingly to the small object size
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1000

    # Weight decay regularization test for .001 .0005?
    WEIGHT_DECAY = 0.0001

def percentile_changed(val=0):
    canvas.imageList = []
    img = draw_markers(img_path)
    img = img.resize((img_width, img_height), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(img)
    created_image_id = canvas.create_image(450, 300, image=image)
    canvas.imageList.append(image)


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


def slice_image(image):
    height = image.shape[0]
    height_copied = 0
    width = image.shape[1]
    width_copied = 0
    x2 = 0
    y2 = 0
    global img_width_in_slices
    img_width_in_slices = 0
    image_slices = []
    if height > 512 or width > 512:
        while height_copied < height:
            if height_copied == 0:
                y1 = height_copied
            else:
                y1 = height_copied# - 24
            if y2 + 512 > height:
                y2 = height
            else:
                y2 = y1 + 512
            img_width_in_slices = 0
            while width_copied < width:
                if width_copied == 0:
                    x1 = width_copied
                else:
                    x1 = width_copied# - 24
                if x2 + 512 > width:
                    x2 = width
                else:
                    x2 = x1 + 512
                img_slice = image[y1:y2, x1:x2]  # img[y1:y2, x1:x2]
                image_slices.append(img_slice)
                height_copied = y2
                width_copied = x2
                img_width_in_slices += 1
            width_copied = 0
            x2 = 0

    return image_slices


def detect_slices(slices):
    detection_outcomes = []
    for image_slice in slices:
        detection_outcomes.append(model.detect([rgba2rgb(image_slice)])[0])
    return detection_outcomes


def detect():
    global r
    global large_image
    image = skimage.io.imread(img_path)    # Read image
    if image.shape[0] > 512 or image.shape[1] > 512:
        large_image = True
        global image_slices
        image_slices = slice_image(image)   # Slice img
        detection_outcomes = detect_slices(image_slices)    # Detection on slices
        r = detection_outcomes
    else:
        large_image = False
        r = model.detect([image])[0]  # Detect stars
    sort_bboxes()


def sort_bboxes():
    box_sizes.clear()
    global boxes_list
    global large_image
    if large_image:
        boxes_list = []
        for i in range(len(r)):
            for item in r[i]['rois']:
                y1, x1, y2, x2 = item
                y1 += 512 * int(i%len(r)/img_width_in_slices)
                x1 += 512 * (i % img_width_in_slices)
                y2 += 512 * int(i%len(r)/img_width_in_slices)
                x2 += 512 * (i % img_width_in_slices)
                boxes_list.append((y1, x1, y2, x2))
    else:
        boxes_list = r['rois']

    for i in range(len(boxes_list)):  # Get and sort bounding boxes of detected objects by size
        y1, x1, y2, x2 = boxes_list[i]
        size = abs((y1 - y2) * (x1 - x2))
        box_sizes.append((i, size))
    box_sizes.sort(key=lambda x: x[1])




def calculate_percentiles():  # Calculates percentile of size for each bounding box
    global percentiles
    percentiles = [None]*len(box_sizes)
    for i in range(len(box_sizes)):
        index, size = box_sizes[i]
        percentile = (i / len(box_sizes)) * 100
        percentiles[index] = percentile


def draw_markers(filename):
    with Image.open(filename) as img:
        draw = ImageDraw.Draw(img)
        for i in range(len(box_sizes)):
            index, size = box_sizes[i]
            initial_y1, initial_x1, initial_y2, initial_x2 = boxes_list[index]

            if output_star_size.get() < 100:    # If size changes calculate new coordinates
                size_multiplicator = output_star_size.get() / 100
                width = (initial_x2-initial_x1)*size_multiplicator
                height = (initial_y2-initial_y1)*size_multiplicator
                x1 = initial_x1 + (initial_x2-initial_x1)/2-width/2
                y1 = initial_y1 + (initial_y2-initial_y1)/2-height/2
                x2 = initial_x1 + (initial_x2-initial_x1)/2+width/2
                y2 = initial_y1 + (initial_y2-initial_y1)/2+height/2
            else:
                y1, x1, y2, x2 = initial_y1, initial_x1, initial_y2, initial_x2

            #   Draw markers
            if (percentiles[index] < percentage_of_smallest_stars.get() or 100 - percentiles[index] <= percentage_of_largest_stars.get()) and output_star_size.get() != 0:
                if box_markers_check_value.get() == 1: draw.rectangle([(x1, y1), (x2, y2)], outline='yellow')
                if circle_markers_check_value.get() == 1: draw.ellipse((x1, y1, x2, y2), outline='yellow')
            elif (percentiles[index] < percentage_of_smallest_stars.get() or 100 - percentiles[index] <= percentage_of_largest_stars.get()) and output_star_size.get() == 0:
                draw.line([(initial_x1, initial_y1), (initial_x2, initial_y2)], fill="red", width=2)
                draw.line([(initial_x2, initial_y1), (initial_x1, initial_y2)], fill="red", width=2)
            else:
                if box_markers_check_value.get() == 1: draw.rectangle([(initial_x1, initial_y1), (initial_x2, initial_y2)], outline='red')
                if circle_markers_check_value.get() == 1: draw.ellipse((initial_x1, initial_y1, initial_x2, initial_y2), outline='red')
        return img


def open_file():
    path = askopenfilename(title="Select A File",
                               filetype=(("jpg, bmp, png, tiff files", "*.jpg *.bmp *.png *.tiff"),
                                         ("jpg files", "*.jpg"), ("bmp files", "*.bmp"), ("png files", "*.png"),
                                         ("tiff files", "*.tiff"), ("all files", "*.*")))
    if path:
        global img_path
        img_path = path
        canvas.imageList = []
        t1 = Thread(target=detect())
        t1.start()
        calculate_percentiles()
        img = draw_markers(img_path)
        image = img.resize((img_width, img_height), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        created_image_id = canvas.create_image(450, 300, image=image)
        canvas.imageList.append(image)

        # Enable controls
        output_star_size['state'] = "normal"
        percentage_of_largest_stars['state'] = "normal"
        percentage_of_smallest_stars['state'] = "normal"
        process_button['state'] = "normal"

def task():
    m = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs/")
    m.load_weights(weights_path, by_name=True)
    loading_window.destroy()
    return m


# def create_defect_mask_bbox(image):
#     mask = np.zeros(image.shape[:-1], dtype=bool)
#
#     for i in range(len(r['rois'])):
#         if percentiles[i] < percentage_of_smallest_stars.get() or 100 - percentiles[i] <= percentage_of_largest_stars.get():
#             y1, x1, y2, x2 = r['rois'][i]
#             height = y2-y1
#             width = x2-x1
#             mask[y1-(int(height*0.2)):y2+(int(height*0.2)), x1-(int(width*0.2)):x2+(int(width*0.2))] = 1
#     #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
#     print(mask.shape)
#     return mask

def get_masks_of_chosen_stars(masks, bboxes, starting_index=0):
    a = masks
    length = starting_index+len(bboxes)
    removed = 0

    # if get_bboxes == 1:
    #     bboxes = r['rois']

    for i in range(starting_index, length - removed):
        if percentiles[i] >= percentage_of_smallest_stars.get() and 100 - percentiles[i] > percentage_of_largest_stars.get():
            a = np.delete(a, i - removed - starting_index, 2)
            bboxes = np.delete(bboxes, i - removed - starting_index, 0)
            removed += 1

    return a, bboxes


def create_defect_mask(masks_of_chosen_stars, keep_dimensions=False):
    mask = (np.sum(masks_of_chosen_stars, -1, keepdims=keep_dimensions) >= 1)   # sum masks into one mask

    if keep_dimensions:
        mask = ndimage.binary_dilation(mask, None, 1)
    else:
        mask = ndimage.binary_dilation(mask, None, dilatation_iterations.get())
    return mask


def img_frombytes(data):  # allows for viewing defect mask as an img (for debugging)
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def inpaint_image(img, mask, mask2d, bboxes=None):
    inpainted_img = inpaint.inpaint_biharmonic(img, mask2d, multichannel=True)  # Remove stars by inpainting
    inpainted_img = (inpainted_img * 255).round().astype(np.uint8)  # convert to RGB
    if output_star_size_value.get() != 0:  # If expected size is not 0 then adjust star size and paste onto image
        for i in range(len(bboxes)):
            y1, x1, y2, x2 = bboxes[i]
            star = img[y1:y2, x1:x2]
            size_multiplicator = output_star_size_value.get() / 100
            star = rescale(star, (size_multiplicator, size_multiplicator), multichannel=True, preserve_range=False)
            small_mask = rescale(mask[y1:y2, x1:x2], (size_multiplicator, size_multiplicator), multichannel=True, preserve_range=False)
            new_y1 = int(y2 - (y2 - y1) / 2 - star.shape[0] / 2)
            new_y2 = int(y2 - (y2 - y1) / 2 + star.shape[0] / 2)
            new_x1 = int(x2 - (x2 - x1) / 2 - star.shape[1] / 2)
            new_x2 = int(x2 - (x2 - x1) / 2 + star.shape[1] / 2)
            star_bg = inpainted_img[new_y1:new_y2, new_x1:new_x2]
            star = (star * 255).round().astype(np.uint8)  # convert to RGB
            star = np.where(small_mask, star, star_bg)  # Copy background to non star pixels so only the star changes size
            inpainted_img[new_y1:new_y2, new_x1:new_x2] = star
    return inpainted_img


def process_img():  # return inpainted image
    if output_star_size_value.get() != 100:
        img = skimage.io.imread(img_path)
        if large_image:
            starting_index = 0
            inpainted_img = img
            for i in range(len(image_slices)):
                masks_of_chosen_stars, bboxes_of_chosen_stars = get_masks_of_chosen_stars(r[i]['masks'], r[i]['rois'], starting_index)
                mask2d = create_defect_mask(masks_of_chosen_stars)  # Get mask without color channels (2D mask required for inpainting)
                mask = create_defect_mask(masks_of_chosen_stars, True)  # Get mask with color channels included (np.where requires same dimensions)
                inpainted_slice = inpaint_image(image_slices[i], mask, mask2d, bboxes_of_chosen_stars)
                starting_index += len(r[i]['rois'])

                #copy onto img
                y1 = 512 * int(i % len(r) / img_width_in_slices)
                x1 = 512 * (i % img_width_in_slices)
                y2 = 512 + 512 * int(i % len(r) / img_width_in_slices)
                x2 = 512 + 512 * (i % img_width_in_slices)
                if y2 > img.shape[0]:
                    y2 = img.shape[0]
                if x2 > img.shape[1]:
                    x2 = img.shape[1]
                inpainted_img[y1:y2, x1:x2] = inpainted_slice
        else:
            masks_of_chosen_stars, bboxes_of_chosen_stars = get_masks_of_chosen_stars(r['masks'], r['rois'], 0)
            mask2d = create_defect_mask(masks_of_chosen_stars)  # Get mask without color channels (2D mask required for inpainting)
            mask = create_defect_mask(masks_of_chosen_stars, True)  # Get mask with color channels included (np.where requires same dimensions)
            inpainted_img = inpaint_image(img, mask, mask2d, bboxes_of_chosen_stars)

        global processed_img
        processed_img = inpainted_img   # Save outcome in a global variable (for file saving)
        display_processed_img(inpainted_img)
        outcome_ui()
        return inpainted_img


def initial_ui():
    save_button.grid_forget()
    back_button.grid_forget()
    compare_button.grid_forget()

    output_star_size.grid(column=18, row=1)
    dilatation_iterations.grid(column=18, row=2)
    sliders_label.grid(column=17, row=2, columnspan=3)
    percentage_of_largest_stars.grid(column=18, row=3)
    percentage_of_smallest_stars.grid(column=18, row=4)
    box_markers_check.grid(column=18, row=5, sticky='w')
    circle_markers_check.grid(column=18, row=6, sticky='w')
    process_button.grid(column=18, row=9)
    load_image_button.grid(column=18, row=11)
    save_button.grid_forget()


def outcome_ui():
    save_button.grid(column=18, row=10)
    save_button["command"] = save_processed_img
    back_button.grid(column=18, row=11)
    compare_button.grid(column=18, row=9)

    output_star_size.grid_forget()
    dilatation_iterations.grid_forget()
    sliders_label.grid_forget()
    percentage_of_largest_stars.grid_forget()
    percentage_of_smallest_stars.grid_forget()
    box_markers_check.grid_forget()
    circle_markers_check.grid_forget()
    process_button.grid_forget()
    load_image_button.grid_forget()
    save_button.grid()

    root.update()


def save_processed_img():
    filename = asksaveasfilename(defaultextension=".jpg", filetypes=[('jpg', '*.jpg'), ('bmp', '*.bmp'), ('png', '*.png'), ('tiff', '*.tiff')])
    plt.imsave(filename, processed_img)


def display_processed_img(img):
    img = skimage.transform.resize(img, (img_height, img_width), anti_aliasing=True)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageTk.PhotoImage(img)
    canvas.create_image(450, 300, image=img)
    canvas.imageList.append(img)


def back_button_event():
    percentile_changed()
    initial_ui()


def compare_button_event():
    # Create new window with size as canvas in old window
    original_image_window = tk.Toplevel()
    original_image_window.geometry("900x600")
    original_image_window.resizable(0, 0)
    original_image_window.title("StarRemover Input")

    original_img_canvas = tk.Canvas(original_image_window, width=900, height=600, bg="#cfcfcf", bd=0)
    original_img_canvas.grid(column=0, row=0, columnspan=17, rowspan=12)
    # draw original on new canvas
    original_img_canvas.imageList = []  # To get rid of garbage collector, empty list is added
    img = skimage.io.imread(img_path)
    img = skimage.transform.resize(img, (img_height, img_width), anti_aliasing=True)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageTk.PhotoImage(img)
    original_img_canvas.create_image(450, 300, image=img)
    original_img_canvas.imageList.append(img)


class InferenceConfig(StarConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    global large_image
    large_image = False
    weights_path = "mask_rcnn_star.h5"

    # Loading window
    loading_window = tk.Tk()
    loading_window.geometry("200x70")
    loading_window.eval('tk::PlaceWindow . center')
    loading_window.resizable(0, 0)
    loading_window.title("Star Remover")

    loading_label = tk.Label(loading_window, text="Loading the model...")
    loading_label.grid(column=0, row=0,)

    config = InferenceConfig()
    model = None
    loading_window.update()
    model = task()
    loading_window.mainloop()

    # Main window
    box_sizes = []
    img_width = 900
    img_height = 600

    root = tk.Tk()
    root.geometry("1025x600")
    root.resizable(0, 0)
    root.title("StarRemover")

    canvas = tk.Canvas(root, width=900, height=600, bg="#cfcfcf", bd=0)
    canvas.grid(column=0, row=0, columnspan=17, rowspan=12)

    # Buttons
    # Load Image button
    load_image_text = tk.StringVar()
    load_image_button = tk.Button(root, command=open_file,
                              textvariable=load_image_text,
                              font="Ariel",
                              bg="black",
                              fg="white",
                              height=2, width=10)
    load_image_text.set("Load Image")

    # Save picture button
    save_button_text = tk.StringVar()
    save_button = tk.Button(root, textvariable=save_button_text, font="Ariel",
                            bg="black",
                            fg="white",
                            height=2, width=10)
    save_button_text.set("Save Image")

    # Process img button
    process_button_text = tk.StringVar()
    process_button = tk.Button(root, command=process_img, textvariable=process_button_text, font="Ariel",
                               bg="black",
                               fg="white",
                               height=2, width=10)
    process_button_text.set("Process")

    # Back button
    back_button_text = tk.StringVar()
    back_button = tk.Button(root, command=back_button_event, textvariable=back_button_text, font="Ariel",
                               bg="black",
                               fg="white",
                               height=2, width=10)
    back_button_text.set("Back")


    # Compare button
    compare_button_text = tk.StringVar()
    compare_button = tk.Button(root, command=compare_button_event, textvariable=compare_button_text, font="Ariel",
                            bg="black",
                            fg="white",
                            height=2, width=10)
    compare_button_text.set("Show Original")

    # Affected stars label
    sliders_label_text = tk.StringVar()
    sliders_label = tk.Label(root, textvariable=sliders_label_text)

    sliders_label_text.set("Affected stars")

    # Output star size slider
    output_star_size_value = tk.DoubleVar()
    output_star_size = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label='Output star size',
                                variable=output_star_size_value, command=percentile_changed)
    output_star_size_value.set(100)

    # Number of dilatation iterations slider
    dilatation_iterations_value = tk.DoubleVar()
    dilatation_iterations = tk.Scale(root, from_=1, to=20, orient=tk.HORIZONTAL, label='Dilatate mask', variable=dilatation_iterations_value)
    dilatation_iterations_value.set(5)

    # Percentage of largest/smallest stars sliders
    percentage_of_largest_stars = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label='Largest:',
                                           command=percentile_changed)
    percentage_of_smallest_stars = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, label='Smallest:',
                                            command=percentile_changed)

    # Box/Circle markers checkboxes
    box_markers_check_value = tk.IntVar()
    box_markers_check = tk.Checkbutton(root, text="Box", variable=box_markers_check_value, command=percentile_changed)

    circle_markers_check_value = tk.IntVar()
    circle_markers_check = tk.Checkbutton(root, text="Circle", variable=circle_markers_check_value, command=percentile_changed)
    circle_markers_check_value.set(1)

    # Initialise UI
    initial_ui()
    output_star_size['state'] = "disabled"
    percentage_of_largest_stars['state'] = "disabled"
    percentage_of_smallest_stars['state'] = "disabled"
    process_button['state'] = "disabled"

    root.mainloop()
