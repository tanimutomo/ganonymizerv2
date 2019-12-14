import torch

from .utils import labels


class MaskCreater:
    def __init__(self, opt):
        for k, v in opt.__dict__.items():
            setattr(self, k, v)

        self.target_ids = list()
        for label in labels:
            if ((label.category is 'vehicle' or label.category is 'human')
                    and label.trainId is not 19):

                self.target_ids.append(label.trainId)

    def create_mask(self, segmap):
        obj_mask = segmap.clone()
        for id_ in self.target_ids:
            obj_mask = torch.where(obj_mask==id_,
                                   torch.full_like(obj_mask, 255),
                                   obj_mask)
        return  torch.where(obj_mask==255,
                            torch.full_like(segmap, 1),
                            torch.full_like(segmap, 0)).to(torch.float32)
