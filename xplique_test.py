import os
from pathlib import Path
import numpy as np

import kecam
from model_wrapper import ModelWrapper
from xplique.attributions import SobolAttributionMethod
from xplique.attributions import IntegratedGradients
from xplique.attributions import SmoothGrad, Saliency, DeconvNet, GradientInput, GuidedBackprop
from object_detector import BoxIouCalculator,ImageObjectDetectorScoreCalculator, YoloObjectFormater

import matplotlib.pyplot as plt
from xplique.plots.image import _normalize, _clip_percentile
from xplique_metrics import Deletion, Insertion, MuFidelity, AverageStability


class Explainer:
    methods = {
        "ig":IntegratedGradients,
        "saliency":Saliency,
        "deconvnet":DeconvNet,
        "gradient_input":GradientInput,
        "guided_backprop": GuidedBackprop,
        "sobol": SobolAttributionMethod,
        "smoothgrad": SmoothGrad
    }
    metrics = {
        "deletion": Deletion,
        "insertion": Insertion,
        "mufidelity": MuFidelity,
        "average_stability": AverageStability
    }
    def __init__(self, model, nb_classes=20):
        self.model = model
        self.model.decode_predictions.use_static_output = True
        self.score_calculator = ImageObjectDetectorScoreCalculator(
            YoloObjectFormater(), BoxIouCalculator())
        self.explainer_wrapper = None
        self.nb_classes = nb_classes
        self.last_params = None

    def apply(self, method_name, preds, img, params):
        self.last_params = params.copy()
        self.last_img = img
        wrapper = ModelWrapper(self.model, img.shape[1], img.shape[2], self.nb_classes)
        params["operator"] = self.score_calculator.tf_batched_score
        self.explainer_wrapper = self.methods[method_name](wrapper, **self.last_params)
        self.last_expl = self.explainer_wrapper.explain(img, wrapper.get_boxes(preds))
        self.last_params["method"] = method_name
        return self.last_expl

    def score(self, method, explanation, img, preds, params={}):
        wrapper = ModelWrapper(self.model, img.shape[1], explanation.shape[2], self.nb_classes)
        #operator_batched = operator_batching()
        if method != "average_stability":
            params["operator"] = self.score_calculator.tf_batched_score
            metric = self.metrics[method](wrapper, np.array(img), wrapper.get_boxes(preds), **params)
            return metric(explanation)

        metric = self.metrics[method](wrapper, np.array(img), wrapper.get_boxes(preds), **params)
        return metric(self.explainer_wrapper)

    def _get_experiment_id(self, exp_name, alpha, cmap, clip_percentile):
        id = exp_name
        for param_name, param_value in self.last_params.items():
            id += f"_{param_name}_{param_value}"
        id += f"_alpha_{alpha}_cmap_{cmap}_clip_percentile_{clip_percentile}"
        return id

    def visualize(self, exp_name="experiment", dest_folder="img", alpha=0.5, cmap="jet", clip_percentile=0.5, figsize=(10,10)):
        dest_folder = Path(dest_folder)
        os.makedirs(dest_folder, exist_ok=True)
        image = self.last_img[-1]
        expl = self.last_expl[-1]
        plt.figure(figsize=figsize)
        if image is not None:
            image = _normalize(image)
            plt.imshow(image)
        if expl.shape[-1] == 3:
            expl = np.mean(expl, -1)
        if clip_percentile:
            expl = _clip_percentile(expl, clip_percentile)
        if not np.isnan(np.max(np.min(expl))):
            expl = _normalize(expl)

        plt.imshow(expl, cmap=cmap, alpha=alpha)
        plt.axis('off')
        img_file = dest_folder / (self._get_experiment_id(exp_name, alpha, cmap, clip_percentile) + ".png")
        plt.savefig(img_file)


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    # mm = kecam.yolor.YOLOR_CSP()
    # model = "yolor_csp"
    model = "yolov7_tiny"
    mm = kecam.yolov7.YOLOV7_Tiny()
    imm_orig = kecam.test_images.cat()
    img_name = "cat"
    # imm_orig = kecam.test_images.dog_cat()
    # img_name = "cat_dog"
    print(imm_orig.shape)
    imm = mm.preprocess_input([imm_orig])
    preds = mm(imm)

    print(preds.shape)
    preds = mm.decode_predictions(preds)
    bboxs, lables, confidences = preds[0]
    kecam.coco.show_image_with_bboxes(imm_orig, bboxs, lables, confidences)

    nb_classes = 20
    explainer = Explainer(mm)
    """
    params_sobol = {
        "batch_size": 16,
        "grid_size": 16,
        "nb_design": 32
    }"""
    """
    ### Param saliency
    method = "saliency"
    params = {
        "batch_size": 16
    }
    """
    method = "smoothgrad"
    params = {
        "batch_size": 16,
        "nb_samples": 5,
        "noise": 0.069
    }
    explanation = explainer.apply(method, preds, imm, params)
    explainer.visualize(f"{model}_{img_name}", "img")
    plt.show()
    #print("Score :", explainer.score("average_stability", explanation, imm, preds, {"batch_size":16, "nb_samples":50}))
    print("Score :", explainer.score("deletion", explanation, imm, preds, {"batch_size":16, "steps":30,"max_percentage_perturbed":0.5}))
