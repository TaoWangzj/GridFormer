from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from .image_restoration_model import ImageCleanModel
from basicsr.models import lr_scheduler as lr_scheduler
import torch.nn.functional as F


@MODEL_REGISTRY.register()
class MIMOModel(ImageCleanModel):
    """Base IR model for General Image Restoration."""
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        ######## produce multi-images as inputs
        gt_images = [ 0 for _ in range(len(self.output))]

        gt_images[0] = self.gt
        for i in range(1,len(self.output)):
            gt_images[i] = F.interpolate(gt_images[i-1], scale_factor=0.5, mode='bilinear',recompute_scale_factor=True)

        gt_images.reverse()


        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for j in range(len(self.output)):
                l_pix += self.cri_pix(self.output[j], gt_images[j])
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep = 0.

            for i in range(len(self.output)):
                l_percep1, l_style1 = self.cri_perceptual(self.output[i], gt_images[i])
                l_percep += l_percep1

            l_total += l_percep
            loss_dict['l_percep'] = l_percep

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    