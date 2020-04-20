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

import polyscope as ps


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


def get_prediction(frame):
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
    pred_joints[:, 1] *= -1
    pred_vertices[:, 1] *= -1
    return pred_joints, pred_vertices


def next_frame(pred_joints, pred_vertices):
    ps_net.update_node_positions(pred_joints)
    ps_mesh.update_vertex_positions(pred_vertices)
    print(abs(pred_vertices[3508,0] - pred_vertices[3021,0]))
    if abs(pred_vertices[3508, 0] - pred_vertices[3021, 0]) > 0.02:
        ps_mesh.set_color([245/255., 105/255., 66/255.])
        ps_net.set_color([245 / 255., 105 / 255., 66 / 255.])
    else:
        ps_mesh.set_color([144/255., 245/255., 66/255.])
        ps_net.set_color([144 / 255., 245 / 255., 66 / 255.])

    ps.show()


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

    ret, first_frame = cap.read()
    cv2.imshow('frame', first_frame)
    # Preprocess input image and generate predictions
    first_joints, first_vertices = get_prediction(first_frame)
    edges = np.array([[2, 3], [3, 4], [5, 6], [6, 7], [9, 10], [10, 11], [12, 13], [13, 14], [14, 19], [14, 20],
                      [14, 21], [11, 22], [11, 23], [11, 24], [12, 8], [8, 9], [8, 41], [41, 40], [37, 43],
                      [40, 5], [40, 2], [40, 37]])
    ps.init()
    ps_net = ps.register_curve_network("my network", first_joints, edges)
    ps_mesh = ps.register_surface_mesh("my mesh", first_vertices, smpl.faces)
    ps.show()

    # Setup renderer for visualization
    # renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
    while True:
        _, current_frame = cap.read()
        cv2.imshow('frame', current_frame)
        current_joints, current_vertices = get_prediction(current_frame)
        next_frame(current_joints, current_vertices)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
