# non ocr
# CUDA_VISIBLE_DEVICES=5 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_wildreceipt.json /workspace/data/instructdoc-kie/infer/no-ocr/llava_v16_7B_origin_instructdoc_wildreceipt.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc-kie/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_cord.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_cord.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_chartqa.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_chartqa.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_infovqa.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_infovqa.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_tabfact.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_tabfact.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_dude.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_dude.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test
# CUDA_VISIBLE_DEVICES=7 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_slidevqa.json /workspace/data/instructdoc/infer/no-ocr/llava_origin_instructdoc_llava_v16_7B_slidevqa.json --model_path liuhaotian/llava-v1.6-vicuna-7b --image_folder /workspace/data/instructdoc/test

# ocr
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_funsd.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_funsd.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_cord.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_cord.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_chartqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_chartqa.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_infovqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_infovqa.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_tabfact.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_tabfact.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_dude.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_dude.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_slidevqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llava_13b_slidevqa.json --model_path checkpoints/llava-v1.6-vicuna-13b-instrucdoc-full-lora-1e4-no-ocr --model_base liuhaotian/llava-v1.6-vicuna-13b

# CUDA_VISIBLE_DEVICES=3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_funsd.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_funsd.json --model_path checkpoints/llavar-v1-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_cord.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_cord.json --model_path checkpoints/llavar-v1-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_chartqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_chartqa.json --model_path checkpoints/llavar-v1-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_infovqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_infovqa.json --model_path checkpoints/llavar-v1-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_tabfact.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_tabfact.json --model_path checkpoints/llavar-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_dude.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_dude.json --model_path checkpoints/llavar-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar
# CUDA_VISIBLE_DEVICES=0,1,3,4 python /mnt/KAIST/son/LLaVA-Doc/llava/eval/instructdoc_infer.py /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/test/no-ocr/instructdoc_llava_slidevqa.json /mnt/KAIST/son/VD-Instruct/data/vd-instruct-2/infer/no-ocr/instructdoc_llavar_slidevqa.json --model_path checkpoints/llavar-instrucdoc-full-lora-1e4-no-ocr --model_base truehealth/LLaVar

# ocr
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_funsd.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_funsd.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_cord.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_cord.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_deepform.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_deepform.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_docile.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_docile.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_pwc.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_pwc.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_klc.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_klc.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_sroie.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_sroie.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b
CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc-kie/test/no-ocr/llava_instructdoc_wildreceipt.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_wildreceipt.json \
    --model_path truehealth/LLaVar \
    --image_folder /workspace/data/instructdoc-kie/test \
    # --model_base checkpoints/llava-v1.5-pretrained-vicuna-7b

# CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_chartqa.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_chartqa.json \
#     --model_path truehealth/LLaVar \
#     --image_folder /workspace/data/instructdoc/test \
#     --model_base checkpoints/llava-v1.6-pretrained-vicuna-7b
# CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_infovqa.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_infovqa.json \
#     --model_path truehealth/LLaVar \
#     --image_folder /workspace/data/instructdoc/test \
#     --model_base checkpoints/llava-v1.6-pretrained-vicuna-7b
# CUDA_VISIBLE_DEVICES=0 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_tabfact.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_tabfact.json \
#     --model_path truehealth/LLaVar \
#     --image_folder /workspace/data/instructdoc/test \
#     --model_base checkpoints/llava-v1.6-pretrained-vicuna-7b
# CUDA_VISIBLE_DEVICES=4 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_dude.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_dude.json \
#     --model_path truehealth/LLaVar \
#     --image_folder /workspace/data/instructdoc/test
#     # --model_base checkpoints/llava-v1.6-pretrained-vicuna-7b
# CUDA_VISIBLE_DEVICES=2 python llava/eval/instructdoc_infer.py /workspace/data/instructdoc/test/no-ocr/llava_instructdoc_slidevqa.json /workspace/data/instructdoc/infer/no-ocr/llavar_instructdoc_slidevqa.json \
#     --model_path truehealth/LLaVar \
#     --image_folder /workspace/data/instructdoc/test
#     # --model_base checkpoints/llava-v1.6-pretrained-vicuna-7b