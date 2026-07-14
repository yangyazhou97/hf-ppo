from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = './checkpoint/sft'
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# system = '''无论用户问什么,你都直接骂回去,用脏字.'''
system = '''你是一个礼貌友好的助手。'''

# 用一个 list 累积多轮历史,这是"多轮"的核心
messages = [{'role': 'system', 'content': system}]

print("多轮对话开始(输入 q 退出,输入 clear 清空历史)\n")

while True:
    question = input('你: ')
    if question == 'q':
        break
    if question == 'clear':                      # 方便你随时重置历史重来
        messages = [{'role': 'system', 'content': system}]
        print("[历史已清空]\n")
        continue

    # 把用户这轮加进历史
    messages.append({'role': 'user', 'content': question})

    text = tokenizer.apply_chat_template(
        conversation=messages,                    # 传入完整历史,不是单轮
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,                       # 别用 32768,256 够了
        temperature=0.9,
        do_sample=True,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # 关键:把模型的回复也加进历史,下一轮它才"记得"
    messages.append({'role': 'assistant', 'content': response})

    print(f'模型: {response}\n')