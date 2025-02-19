from transformers import Qwen2_5_VLForConditionalGeneration
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

print('preparing model')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

print('cuda memory:')
torch.cuda.memory_allocated()

print('preparing processor')
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

print('processing inputs')
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print('moving model and tensors to gpu')
inputs = inputs.to("cuda")
model.visual = model.visual.to('cuda')

print('getting image embeddings')
pixel_values = inputs['pixel_values']
image_grid_thw = inputs['image_grid_thw']
pixel_values = pixel_values.type(model.visual.dtype)
image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

print('image embeddings:')
print(image_embeds.shape)