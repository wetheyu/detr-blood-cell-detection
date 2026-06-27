import argparse
import pathlib

import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image

from models import build_model
from predict import filter_boxes, plot_one_box, rescale_bboxes
from project_config import CLASSES, DEFAULT_COCO_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_TRAINED_CHECKPOINT, NUM_CLASSES


# Checkpoints created on Linux may contain PosixPath objects. This keeps loading
# them usable on Windows machines.
pathlib.PosixPath = pathlib.WindowsPath


def get_args_parser():
    parser = argparse.ArgumentParser('DETR Gradio demo', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_classes', default=NUM_CLASSES, type=int)

    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default=str(DEFAULT_COCO_PATH))
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=str(DEFAULT_TRAINED_CHECKPOINT))
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', default=True)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--server_name', default='127.0.0.1', type=str)
    parser.add_argument('--server_port', default=7860, type=int)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--no_inbrowser', dest='inbrowser', action='store_false')
    parser.set_defaults(inbrowser=True)
    return parser


def load_detector(args):
    device = torch.device(args.device)
    print(f"Loading model on {device} from {args.resume} ...", flush=True)
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("Model loaded.", flush=True)
    return model, device


def build_detector_fn(model, device):
    image_to_tensor = torchvision.transforms.ToTensor()

    def detect_image(image):
        image_tensor = image_to_tensor(image)
        image_tensor = torch.reshape(image_tensor, [-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]])
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            inference_result = model(image_tensor)

        probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        bboxes_scaled = rescale_bboxes(
            inference_result['pred_boxes'][0].cpu(),
            (image_tensor.shape[3], image_tensor.shape[2]),
        )
        scores, boxes = filter_boxes(probas, bboxes_scaled)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()

        image = np.array(image)
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            confidence = scores[i].max()
            text = f"{CLASSES[class_id]} {confidence:.3f}"
            plot_one_box(boxes[i], image, label=text)

        return Image.fromarray(np.uint8(image))

    return detect_image


def main():
    parser = argparse.ArgumentParser('DETR Gradio demo', parents=[get_args_parser()])
    args = parser.parse_args()
    model, device = load_detector(args)

    print(f"Starting Gradio at http://{args.server_name}:{args.server_port}", flush=True)
    gr.Interface(
        inputs=gr.Image(type="pil", label="输入图片"),
        outputs=gr.Image(type="pil", label="检测结果"),
        fn=build_detector_fn(model, device),
        title="基于 DETR 的血细胞检测系统",
        description="上传血细胞图片后，系统会输出检测框、类别和置信度。",
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == '__main__':
    main()
