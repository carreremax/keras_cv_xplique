import kecam
from kecam import visualizing
#mm = kecam.yolov7.YOLOV7_Tiny()
mm = kecam.yolor.YOLOR_CSP()
imm = kecam.test_images.dog_cat()
print(imm.shape)
preds = mm(mm.preprocess_input([imm]))
bboxs, lables, confidences = mm.decode_predictions(preds)[0]
kecam.coco.show_image_with_bboxes(imm, bboxs, lables, confidences)
superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, imm, layer_name="auto")

