from paddleocr import PaddleOCR,draw_ocr
# The path of detection and recognition model must contain model and params files
ocr = PaddleOCR(det_model_dir=None, det=False,
                rec_model_dir='/home/frinksserver/Deepak/OCR/models/paddleocr_train/inference/rec_ppocrv3_skh_jun27', 
                rec_char_dict_path= "/home/frinksserver/Deepak/OCR/models/paddleocr_train/ppocr/utils/en_dict.txt", 
                cls_model_dir=None, use_angle_cls=False)

"""
Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir='/home/frinksserver/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det=True, det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960, det_limit_type='max', det_model_dir='/home/frinksserver/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer', det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_id=0, gpu_mem=500, help='==SUPPRESS==', image_dir=None, image_orientation=False, invert=False, ir_optim=True, kie_algorithm='LayoutXLM', label_list=['0', '180'], lang='ch', layout=True, layout_dict_path=None, layout_model_dir=None, layout_nms_threshold=0.5, layout_score_threshold=0.5, max_batch_size=10, max_text_length=25, merge_no_span_structure=True, min_subgraph_size=15, mode='structure', ocr=True, ocr_order_method=None, ocr_version='PP-OCRv4', output='./output', page_num=0, precision='fp32', process_id=0, re_model_dir=None, rec=True, rec_algorithm='SVTR_LCNet', rec_batch_num=6, rec_char_dict_path='/home/frinksserver/Deepak/OCR/models/paddleocr_train/ppocr/utils/en_dict.txt', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_model_dir='/home/frinksserver/Deepak/OCR/models/paddleocr_train/inference/rec_ppocrv3_skh_jun27', recovery=False, save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ser_model_dir=None, show_log=True, sr_batch_num=1, sr_image_shape='3, 32, 128', sr_model_dir=None, structure_version='PP-StructureV2', table=True, table_algorithm='TableAttn', table_char_dict_path=None, table_max_len=488, table_model_dir=None, total_process_num=1, type='ocr', use_angle_cls=False, use_dilation=False, use_gpu=False, use_mp=False, use_npu=False, use_onnx=False, use_pdf2docx_api=False, use_pdserving=False, use_space_char=True, use_tensorrt=False, use_visual_backbone=True, use_xpu=False, vis_font_path='./doc/fonts/simfang.ttf', warmup=False)

"""
img_path = '/home/frinksserver/Deepak/yolo_fasterrcnn_trocr_modular/results/img_out/ocr_skh_base_jun24_old_data/ocr_skh_base_jun24_rh_new/crops/00ba69bf-35cc-418c-b978-8e283ce0a2cc_0_actual.png_crop_0.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')