import torch
import numpy as np
import os, sys
from monai.inferers import sliding_window_inference
import tifffile as tif
import time

from skimage import morphology, measure
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BasePredictor import BasePredictor
from core.MEDIAR.utils import compute_masks

__all__ = ["Predictor"]


class Predictor(BasePredictor):
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        super(Predictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            make_submission,
            exp_name,
            algo_params,
        )
        self.hflip_tta = HorizontalFlip()
        self.vflip_tta = VerticalFlip()

    @torch.no_grad()
    def conduct_prediction(self):
        self.model.to(self.device)
        self.model.eval()
        total_time = 0
        total_times = []

        for img_name in self.img_names:
            img_data = self._get_img_data(img_name)
            img_data = img_data.to(self.device)

            start = time.time()

            results = self._inference(img_data)

            cell_class_pred_mask = self._post_process_cell_class(results[1].squeeze(0).cpu().numpy())
        
            cell_instance_pred_mask = self._post_process_cell_detect(results[0].squeeze(0).cpu().numpy())


            self.write_pred_mask(
                cell_instance_pred_mask, self.output_path, img_name, self.make_submission
            )

            self.write_pred_mask(
                cell_class_pred_mask, self.output_path, "class_"+img_name, submission=False
            )

            end = time.time()

            time_cost = end - start
            total_times.append(time_cost)
            total_time += time_cost
            print(
                f"Prediction finished: {img_name}; img size = {img_data.shape}; costing: {time_cost:.2f}s"
            )

        print(f"\n Total Time Cost: {total_time:.2f}s")

        if self.make_submission:
            fname = "%s.zip" % self.exp_name

            os.makedirs("./submissions", exist_ok=True)
            submission_path = os.path.join("./submissions", fname)

            with ZipFile(submission_path, "w") as zipObj2:
                pred_names = sorted(os.listdir(self.output_path))
                for pred_name in pred_names:
                    pred_path = os.path.join(self.output_path, pred_name)
                    zipObj2.write(pred_path)

            print("\n>>>>> Submission file is saved at: %s\n" % submission_path)

        return time_cost
    
    @torch.no_grad()
    def _inference(self, img_data):
        """Conduct model prediction"""

        img_data = img_data.to(self.device)
        img_base = img_data
        cell_detect_mask, class_segm_mask = self._window_inference(img_base)
        cell_detect_mask = cell_detect_mask.cpu().squeeze()
        class_segm_mask = class_segm_mask.cpu().squeeze()
        img_base.cpu()

        if not self.use_tta:
            # pred_mask = create_pred_mask(cell_detect_mask, class_segm_mask)
            return cell_detect_mask, class_segm_mask

        else:
            # HorizontalFlip TTA
            img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)
            outputs_hflip = self._window_inference(img_hflip)
            outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
            outputs_hflip = outputs_hflip.cpu().squeeze()
            img_hflip = img_hflip.cpu()

            # VertricalFlip TTA
            img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)
            outputs_vflip = self._window_inference(img_vflip)
            outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
            outputs_vflip = outputs_vflip.cpu().squeeze()
            img_vflip = img_vflip.cpu()

            # Merge Results
            pred_mask = torch.zeros_like(outputs_base)
            pred_mask[0] = (outputs_base[0] + outputs_hflip[0] - outputs_vflip[0]) / 3
            pred_mask[1] = (outputs_base[1] - outputs_hflip[1] + outputs_vflip[1]) / 3
            pred_mask[2] = (outputs_base[2] + outputs_hflip[2] + outputs_vflip[2]) / 3

        return pred_mask

    def _window_inference(self, img_data, aux=False):
        """Inference on RoI-sized window"""
        outputs = sliding_window_inference(
            img_data,
            roi_size=256,
            sw_batch_size=4,
            predictor=self.model if not aux else self.model_aux,
            padding_mode="constant",
            mode="gaussian",
            overlap=0.6,
        )

        return outputs

    def _post_process_cell_class(self, pred_mask):
        print(f"mask shape = {pred_mask.shape} ")

        pred_mask = torch.from_numpy(pred_mask)
        pred_mask = torch.softmax(pred_mask, dim=0)
        pred_mask = torch.argmax(pred_mask, dim=0)

        return pred_mask.numpy().astype('uint32')
    

    def _post_process_cell_detect(self, pred_mask):
        """Generate cell instance masks."""
        print(f"mask shape = {pred_mask.shape} ")
        dP, cellprob = pred_mask[:2], self._sigmoid(pred_mask[-1])
        H, W = pred_mask.shape[-2], pred_mask.shape[-1]

        if np.prod(H * W) < (5000 * 5000):
            pred_mask = compute_masks(
                dP,
                cellprob,
                use_gpu=True,
                flow_threshold=0.4,
                device=self.device,
                cellprob_threshold=0.5,
            )[0]

        else:
            print("\n[Whole Slide] Grid Prediction starting...")
            roi_size = 2000

            # Get patch grid by roi_size
            if H % roi_size != 0:
                n_H = H // roi_size + 1
                new_H = roi_size * n_H
            else:
                n_H = H // roi_size
                new_H = H

            if W % roi_size != 0:
                n_W = W // roi_size + 1
                new_W = roi_size * n_W
            else:
                n_W = W // roi_size
                new_W = W

            # Allocate values on the grid
            pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
            dP_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
            cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

            dP_pad[:, :H, :W], cellprob_pad[:H, :W] = dP, cellprob

            for i in range(n_H):
                for j in range(n_W):
                    print("Pred on Grid (%d, %d) processing..." % (i, j))
                    dP_roi = dP_pad[
                        :,
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                    cellprob_roi = cellprob_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]

                    pred_mask = compute_masks(
                        dP_roi,
                        cellprob_roi,
                        use_gpu=True,
                        flow_threshold=0.4,
                        device=self.device,
                        cellprob_threshold=0.5,
                    )[0]

                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ] = pred_mask

            pred_mask = pred_pad[:H, :W]

        return pred_mask

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


"""
Adapted from the following references:
[1] https://github.com/qubvel/ttach/blob/master/ttach/transforms.py

"""


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


class DualTransform:
    identity_param = None

    def __init__(
        self, name: str, params,
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError


class HorizontalFlip(DualTransform):
    """Flip images horizontally (left -> right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = hflip(mask)
        return mask


class VerticalFlip(DualTransform):
    """Flip images vertically (up -> down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = vflip(image)

        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = vflip(mask)

        return mask
