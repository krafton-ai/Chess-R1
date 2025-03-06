SYSTEM_PROMPT = {
    "Qwen/Qwen2-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/QwQ-32B-Preview": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-72B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-32B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-1.5B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-Math-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "You are a helpful and harmless assistant. You are DeepSeek R1 developed by DeepSeek. You should think step-by-step.",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "You are a helpful and harmless assistant. You are DeepSeek R1 developed by DeepSeek. You should think step-by-step.",
    "NovaSky-AI/Sky-T1-32B-Preview": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:",
    "bespokelabs/Bespoke-Stratos-7B": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:",
    "bespokelabs/Bespoke-Stratos-32B": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:",
    "mlfoundations-dev/Bespoke-Stratos-17k": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:",
    "openai/o1-mini": "Question: {input}\nAnswer: ",
    "openai/o1-preview": "Question: {input}\nAnswer: ",
    "openai/gpt-4o-mini": "User: {input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:",
    "meta-llama/Llama-3.2-1B-Instruct":  "You are a helpful and harmless assistant. You are Llama developed by Meta. You should think step-by-step.",
    "meta-llama/Meta-Llama-3-8B-Instruct": "You are a helpful and harmless assistant. You are Llama developed by Meta. You should think step-by-step.",
}

MODEL_TO_NAME = {
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen/QwQ-32B-Preview": "QwQ-32B-Preview",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B-Instruct",
    "PRIME-RL/Eurus-2-7B-PRIME": "Eurus-2-7B-PRIME",
    "NovaSky-AI/Sky-T1-32B-Preview": "Sky-T1-32B-Preview",
    "bespokelabs/Stratos-R1-checkpoint-100": "Stratos-R1-checkpoint-100",
    "bespokelabs/Stratos-R1-checkpoint-200": "Stratos-R1-checkpoint-200",
    "bespokelabs/Stratos-R1-Micro": "Stratos-R1-Micro",
    "bespokelabs/Bespoke-Stratos-7B": "Bespoke-Stratos-7B",
    "bespokelabs/Bespoke-Stratos-32B": "Bespoke-Stratos-32B",
    "bespokelabs/Bespoke-Stratos-32B-BACKUP": "Bespoke-Stratos-32B-BACKUP",
    "bespokelabs/Stratos-R1-MICRO-QWEN-checkpoint-200": "Stratos-R1-MICRO-QWEN-checkpoint-200",
    "bespokelabs/Stratos-R1-Micro-Llama": "Stratos-R1-Micro-Llama",
    "openai/o1-mini": "o1-mini",
    "openai/o1-preview": "o1-preview",  
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "ryanmarten/Sky-T1-32B-Preview-5k-1-epoch": "Sky-T1-32B-Preview",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    "mlfoundations-dev/Bespoke-Stratos-17k": "Bespoke-Stratos-17k",
    "mlfoundations-dev/Bespoke-Stratos-17k-v2": "Bespoke-Stratos-17k-v2",
    "mlfoundations-dev/DCFT-Stratos-Verified-114k-7B-4gpus": "DCFT-Stratos-Verified-114k-7B-4gpus",
    "mlfoundations-dev/Bespoke-Stratos-35k-32b": "DCFT-Bespoke-Stratos-35k-32b",
    "mlfoundations-dev/DCFT-Stratos-Verified-114k-32B-4gpus": "DCFT-Stratos-Verified-114k-32B-4gpus",
}
