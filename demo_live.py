import torch
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

import matplotlib.pyplot as plt

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants

import tkinter as tk


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


def process_image(img, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = img[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def next_frame():
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)

    # Preprocess input image and generate predictions
    img, norm_img = process_image(frame, None, None, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1),
                           pose2rot=False)
        pred_vertices = pred_output.vertices

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                      2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                     dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    pred_joints = pred_output.joints[0].cpu().numpy()
    img = img.permute(1, 2, 0).cpu().numpy()

    if rgb_view:
        img_shape_255 = np.array(255 * img, dtype=np.uint8)
    else:
        img_shape_255 = 255 * np.ones_like(img, dtype=np.uint8)
    img_rgb = Image.fromarray(img_shape_255)
    imgtk = ImageTk.PhotoImage(image=img_rgb)
    display1.imgtk = imgtk
    display1.configure(image=imgtk)

    if skeleton_view:
        img_shape = renderer.render_skeleton(pred_joints, camera_translation, np.ones_like(img))

        """fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], color='r')
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], alpha=0.1)
        ax.scatter(pred_joints[16, 0], pred_joints[16, 1], pred_joints[16, 2], color='b')
        ax.scatter(pred_joints[15, 0], pred_joints[15, 1], pred_joints[15, 2], color='b')
        for i in range(len(pred_joints)):
            ax.text(pred_joints[i, 0], pred_joints[i, 1], pred_joints[i, 2], i, None)"""


    else:
        # Render parametric shape
        img_shape = renderer.render_mesh(pred_vertices, camera_translation, np.ones_like(img))
    # cv2.imshow('prediction', img_shape[:, :, ::-1])
    # cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:, :, ::-1])
    img_shape_255 = np.array(255 * img_shape, dtype=np.uint8)
    img_shape_i = Image.fromarray(img_shape_255)
    imgtk = ImageTk.PhotoImage(image=img_shape_i)
    display2.imgtk = imgtk
    display2.configure(image=imgtk)

    if side_view:
        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center

        # Render non-parametric shape
        if skeleton_view:
            img_shape_side = renderer.render_mesh(rot_vertices, camera_translation, np.ones_like(img))
        else:
            img_shape_side = renderer.render_mesh(rot_vertices, camera_translation, np.ones_like(img))
        img_shape_side_255 = np.array(255 * img_shape_side, dtype=np.uint8)
        img_side = Image.fromarray(img_shape_side_255)
        imgtk_side = ImageTk.PhotoImage(image=img_side)
        display3.imgtk = imgtk_side
        display3.configure(image=imgtk_side)
        # cv2.imshow('side view', img_shape_side[:, :, ::-1])

    window.after(1, next_frame)


def toggle_rgb():
    global rgb_view
    rgb_view = not rgb_view


def toggle_skeleton():
    global skeleton_view
    skeleton_view = not skeleton_view


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--outfile', type=str, default=None,
                        help='Filename of output images. If not set use input filename.')

    args = parser.parse_args()

    side_view = True
    rgb_view = True
    skeleton_view = True

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 5)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

    window = tk.Tk()
    window.wm_title("Posture Coach")
    window.config(background="#FFFFFF")
    imageFrame = tk.Frame(window, width=600, height=500)
    imageFrame.grid(row=0, column=0, padx=10, pady=2)

    display1 = tk.Label(imageFrame)
    display1.grid(row=0, column=0, padx=10, pady=2)  # Display 1
    display2 = tk.Label(imageFrame)
    display2.grid(row=0, column=1)  # Display 2
    display3 = tk.Label(imageFrame)
    display3.grid(row=0, column=2)  # Display 3

    buttonFrame = tk.Frame(window, width=600, height=200)
    buttonFrame.grid(row=1, column=0, padx=10, pady=2)
    rgb = tk.Button(buttonFrame, text="RGB", command=toggle_rgb)
    rgb.pack(side=tk.LEFT)
    side = tk.Button(buttonFrame, text="Side")
    side.pack(side=tk.LEFT)
    skeleton = tk.Button(buttonFrame, text="Skeleton", command=toggle_skeleton)
    skeleton.pack(side=tk.LEFT)
    mesh = tk.Button(buttonFrame, text="Mesh")
    mesh.pack(side=tk.LEFT)

    next_frame()
    window.mainloop()

    cap.release()
    cv2.destroyAllWindows()
