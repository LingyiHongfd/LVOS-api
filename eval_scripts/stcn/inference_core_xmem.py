import torch

from inference_memory_bank_long import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad
import torch.nn.functional as F



def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([
        torch.prod(1 - prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))
    prob = F.softmax(logits, dim=dim)

    if return_logits:
        return logits, prob
    else:
        return prob


class InferenceCore:
    def __init__(self, network: STCN, top_k=20, mem_every=5, ):

        self.all_labels = None
        self.seg_labels = None

        self.network = network
        self.mem_every = mem_every

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.top_k = top_k

        self.device = 'cuda'
        self.curr_ti = -1
        self.last_mem_ti = 0

        self.enabled_obj = []

        self.mem_banks = dict()

    def set_all_labels(self, all_labels):
        if self.seg_labels is None:
            self.seg_labels = all_labels
        else:
            self.seg_labels = self.all_labels
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))

        k16, qv16, qf16, qf8, qf4 = self.network.encode_key(image, )
        multi_scale_features = (qf16, qf8, qf4)

        # segment the current frame is needed
        if need_segment:
            
            out_mask = torch.cat([
                self.network.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
                for pi, oi in enumerate(self.seg_labels)], 0)
            pred_prob_with_bg = aggregate(out_mask, dim=0, ).transpose(0, 1)

            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if
                                               (i + 1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)
            # also create new hidden states

        # save as memory if needed
        if is_mem_frame:

            value = self.network.encode_value(image, qf16, pred_prob_with_bg[1:].unsqueeze(1),)

            # K, CK, _, H, W = k16.shape
            # _, CV, _, _, _ = value.shape

            for i, oi in enumerate(self.all_labels):
                if oi not in self.mem_banks:
                    self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k)
                self.mem_banks[oi].add_memory(k16.unsqueeze(2), value[i:i + 1])

            self.last_mem_ti = self.curr_ti

        self.prev_mask = pred_prob_with_bg
        self.prev_image = image

        return unpad(pred_prob_with_bg, self.pad)
