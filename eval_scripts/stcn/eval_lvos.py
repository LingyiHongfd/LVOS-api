
import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from util.tensor_util import unpad
from inference_core_xmem import InferenceCore
from dataset.lvos_test_dataset import LVOSTestDataset
from progressbar import progressbar
from tqdm import tqdm
from dataset.mask_mapper import MaskMapper
import os



"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--lvos_path', default='../LVOS')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')

parser.add_argument('--output',default='./output/lvos')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

lvos_path = args.lvos_path

ckpt_step=args.model.split('_')[-1].split('.')[0]

out_path = args.output




def main():
    # Simple setup
    torch.set_num_threads(8)
    os.makedirs(out_path, exist_ok=True)
    
    palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0,
    128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0,
    191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24,
    24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30,
    30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36,
    37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43,
    43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49,
    49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55,
    56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62,
    62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68,
    68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74,
    75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81,
    81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87,
    87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93,
    94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99,
    100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104,
    105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109,
    110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114,
    115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119,
    120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124,
    125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129,
    130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134,
    135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139,
    140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144,
    145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149,
    150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154,
    155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159,
    160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164,
    165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169,
    170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174,
    175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179,
    180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184,
    185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189,
    190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194,
    195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199,
    200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204,
    205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209,
    210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214,
    215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219,
    220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224,
    225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229,
    230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234,
    235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239,
    240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244,
    245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249,
    250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254,
    255, 255, 255
]

    torch.autograd.set_grad_enabled(False)

    # Setup Dataset
    test_dataset = LVOSTestDataset(data_root=lvos_path, split=args.split)

    # Load our checkpoint
    prop_saved = torch.load(args.model)
    top_k = args.top
    prop_model = STCN().cuda().eval()
    prop_model.load_state_dict(prop_saved)
    prop_model.eval()

    meta_loader = test_dataset.get_datasets()

    # Load our checkpoint
    total_process_time = 0
    total_frames = 0

    # Start eval
    with torch.cuda.amp.autocast(enabled=True):
        for vid_reader in progressbar(meta_loader, max_value=len(test_dataset), redirect_stdout=False):
            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=6,pin_memory=True)
            vid_name = vid_reader.vid_name
            vid_length = len(loader)
            # no need to count usage for LT if the video is not that long anyway

            mapper = MaskMapper()
            processor = InferenceCore(prop_model, top_k=top_k, mem_every=args.mem_every)
            first_mask_loaded = False

            ti=0
            for data in loader:
                rgb = data['rgb'].cuda()[0]
                msk = data.get('mask')
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]
                """
                For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                Seems to be very similar in testing as my previous timing method 
                with two cuda sync + time.time() in STCN though 
                """
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                    else:
                        # no point to do anything without a mask
                        continue

                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = mapper.convert_mask(msk[0].numpy())
                    msk = torch.Tensor(msk).cuda()
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    s_all_labels = list(mapper.remappings.values())
                    processor.set_all_labels(s_all_labels)
                else:
                    labels = None

                # Run the model on this frame
                prob = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))

                if msk is not None:
                    processor.set_all_labels(s_all_labels)
                
                # Upsample to original size if needed
                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

                end.record()
                torch.cuda.synchronize()
                total_process_time = total_process_time + (start.elapsed_time(end) / 1000)
                total_frames = total_frames + 1

                # Probability mask -> index mask
                out_mask = torch.argmax(prob, dim=0)
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

                # Save the mask
                if info['save'][0]:
                    this_out_path = path.join(out_path, 'Annotations', vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    out_img.putpalette(palette)

                    out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))
                
                ti=ti+1

            del loader
            del mapper
            del processor
            del rgb 
            del msk 
            del info 
            del frame 
            del shape 
            del need_resize
            del prob
            del data
            del s_all_labels
            torch.cuda.empty_cache()

    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    print(f'FPS: {total_frames / total_process_time}')

if __name__ == "__main__":
    main()


