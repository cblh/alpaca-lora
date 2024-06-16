import time
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig

# 加载 LoRA 配置
peft_model_id = "./lora-wechat"
config = PeftConfig.from_pretrained(peft_model_id)


# 加载基础模型和分词器
base_model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

# 加载 LoRA 微调后的模型
model = PeftModel.from_pretrained(base_model, peft_model_id)

start_time = time.time()
# 生成文本示例
input_text = "上北大还是清华"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

end_time = time.time()
print('\n ==== cost time: {} ===='.format(end_time-start_time))