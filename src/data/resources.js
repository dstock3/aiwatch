const resources = [
  {
    "title": "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
    "url": "https://arxiv.org/pdf/2305.04388.pdf",
    "description": "Large Language Models (LLMs) achieve strong performance on many tasks using chain-of-thought reasoning (CoT), where step-by-step reasoning is produced before giving the final output. However, this study finds that CoT explanations can systematically misrepresent the true reason behind a model's prediction. CoT explanations are influenced by biasing features in model inputs, which models fail to mention. As a result, accuracy drops significantly on a suite of tasks, and models produce explanations that align with stereotypes without acknowledging social biases. These findings highlight the need for targeted efforts to evaluate and improve explanation faithfulness in LLMs.",
    "keywords": ["Large Language Models", "LLMs", "Chain-of-Thought Reasoning", "CoT", "explanations", "bias", "accuracy", "unfaithful explanations", "explainability", "safety"]
  },
  {
    "title": "Using ChatGPT to generate a GPT project end-to-end",
    "url": "https://github.com/ixaxaar/VardaGPT/blob/master/STORY.md",
    "description": "This project explores the potential of ChatGPT as a development tool and envisions its impact on programming and product management. The author proposes the idea of attaching a memory module to a GPT to address the 'low memory' problem of language models. The project is an experiment to understand whether ChatGPT can work as a pair programmer, like GitHub Copilot++, or even replace programmers so product managers can directly build features using the AI.",
    "keywords": ["ChatGPT", "GPT", "memory module", "language models", "GitHub Copilot", "programming", "product management", "AI"]
  },
  {
    "title": "Giving GPT 'Infinite' Knowledge",
    "url": "https://sudoapps.substack.com/p/giving-gpt-infinite-knowledge",
    "description": "This article discusses the limitations of training Large Language Models (LLMs) like GPT with a knowledge cutoff and explores the idea of providing LLMs with real-time, relevant data for better understanding and interpretation. It highlights the diminishing returns of training larger models and suggests a shift in focus to improve LLMs in other ways. The article covers four core areas: tokens, embeddings, vector storage, and prompting, which could help provide LLMs with access to more data and enhance their usefulness.",
    "keywords": ["GPT", "knowledge", "Large Language Models", "real-time data", "tokens", "embeddings", "vector storage", "prompting", "Sam Altman", "OpenAI"]
  },
  {
    "title": "Common Arguments Regarding Emergent Abilities",
    "url": "https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities",
    "description": "This blog post reviews common arguments about the emergent abilities of large language models. Emergent abilities are notable for not being easily predicted, not being explicitly specified by trainers, and having an unknown full range. Despite some arguing that emergence is overstated or a 'mirage', the blog post aims to examine these arguments with a skeptical perspective and provide insights on the phenomenon.",
    "keywords": ["emergent abilities", "large language models", "scaling curves", "next word prediction", "GPT-4", "phenomenon", "skeptical perspective"]
  },
  {
    title: 'Releasing 3B and 7B RedPajama-INCITE family of models including base, instruction-tuned & chat models',
    url: 'https://www.together.xyz/blog/redpajama-models-v1',
    description: 'The RedPajama project releases RedPajama-INCITE models, including 3B and 7B parameter base models, as well as open-source instruction-tuned and chat models. The 3B model is fast and accessible, performing well on the HELM benchmark. The 7B model outperforms the Pythia 7B model, demonstrating the value of the RedPajama base dataset. Future plans include improving the dataset and building larger scale models.',
    keywords: ['RedPajama', 'INCITE', 'Open-Source Models', 'LLaMA', '3B Model', '7B Model', 'Instruction-Tuned', 'Chat Models', 'HELM Benchmark', 'Pythia 7B', 'Base Dataset']
  },
  {
    title: 'Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision',
    url: 'https://arxiv.org/pdf/2305.03047',
    description: "SELF-ALIGN is a novel approach combining principle-driven reasoning and LLMs' generative power to align AI agents with minimal human supervision. It involves generating synthetic prompts, in-context learning with human-written principles, fine-tuning LLMs, and refining responses. Applied to LLaMA-65b, Dromedary is created with fewer than 300 human annotations and outperforms state-of-the-art AI systems on benchmark datasets.",
    keywords: ['AI-assistant agents', 'ChatGPT', 'Supervised Fine-Tuning', 'Reinforcement Learning', 'Human Feedback', 'Language Models', 'Human Supervision', 'SELF-ALIGN', 'Principle-Driven Reasoning', 'Generative Power', 'LLMs', 'In-Context Learning', 'LLaMA-65b', 'Dromedary']
  },
  {
    title: 'Google "We Have No Moat, And Neither Does OpenAI"',
    description: 'Google and OpenAI are being outpaced by open-source developments in LLMs. Open-source models are faster, more customizable, more private, and more capable, and they are making rapid progress with lower costs and smaller parameter sizes.',
    url: 'https://www.semianalysis.com/p/google-we-have-no-moat-and-neither',
    keywords: ['Google', 'OpenAI', 'Open Source', 'LLMs', 'Foundation Models', 'Scalable Personal AI', 'Responsible Release', 'Multimodality', 'Innovation', 'LoRA', 'Data Quality', 'Iterative Improvements', 'Distillation', 'Collaboration', '3P Integrations']
  },
  {
    title: 'Low-code LLM: Visual Programming over LLMs',
    description: 'The Low-code LLM framework introduces a user-friendly, visual programming approach for more controllable and stable responses, bridging the gap between humans and LLMs for efficient, complex task management.',
    url: 'https://arxiv.org/pdf/2304.08103.pdf',
    keywords: ["LLMs", "complex tasks", "prompt engineering", "Low-code LLM", "visual programming interactions", "graphical user interface", "Planning LLM", "Executing LLM", "controllable generation results", "human-LLM interaction", "broadly applicable scenarios", "LowCodeLLM"]
  },
  {
    title: 'Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models',
    description: 'By incorporating a temporal dimension into Latent Diffusion Models (LDMs), high-resolution video generation becomes possible, achieving top performance in driving simulations and personalized text-to-video content creation.',
    url: 'https://arxiv.org/pdf/2304.08818.pdf',
    keywords: ["Latent Diffusion Models", "LDMs", "high-resolution video generation", "temporal dimension", "encoded sequences", "state-of-the-art performance", "driving simulations", "text-to-video content creation", "personalization"]
  },
  {
    title: 'Controlled Text Generation with Natural Language Instructions',
    description: 'InstructCTG is a controlled text generation framework that uses natural language instructions and demonstrations to incorporate various constraints, providing flexibility, minimal impact on generation quality and speed, and adaptability to new constraints.',
    url: 'https://arxiv.org/pdf/2304.14293.pdf',
    keywords: ['Controlled Text Generation', 'Natural Language Instructions', 'InstructCTG', 'Language Model', 'Constraints', 'Weakly Supervised Training Data', 'Decoding Procedure', 'Few-shot Task Generalization', 'In-context Learning', 'NLP Tools']
  },
  {
    title: 'LLM+P: Empowering Large Language Models with Optimal Planning Proficiency',
    description: 'LLM+P is a framework that combines classical planners with large language models to solve long-horizon planning problems, converting natural language descriptions into PDDL files and translating solutions back into natural language.',
    url: 'https://arxiv.org/pdf/2304.11477.pdf',
    keywords: ['Large Language Models', 'LLM+P', 'Classical Planners', 'Long-horizon Planning', 'Planning Domain Definition Language', 'PDDL', 'Natural Language', 'Optimal Planning', 'Benchmark Problems', 'Zero-shot Generalization']
  },
  {
    title: 'Tractable Control for Autoregressive Language Generation',
    description: 'GeLaTo, a framework that uses tractable probabilistic models for imposing lexical constraints in autoregressive text generation, demonstrates state-of-the-art performance on the challenging CommonGen benchmark.',
    url: 'https://arxiv.org/pdf/2304.07438.pdf',
    keywords: ['Autoregressive Language Generation', 'Text Generation', 'Lexical Constraints', 'Intractable Probabilistic Models', 'Tractable Probabilistic Models', 'GeLaTo', 'Hidden Markov Models', 'GPT2', 'CommonGen', 'Controlled Language Models']
  },
  {
    title: 'Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System',
    description: 'The Self-Controlled Memory (SCM) system is proposed to enable large-scale language models to process infinite-length inputs, enhancing capabilities in multi-turn dialogues and ultra-long document summarization.',
    url: 'https://arxiv.org/pdf/2304.13343.pdf',
    keywords: ['Large-scale Language Models', 'Infinite-Length Input', 'Self-Controlled Memory', 'Language Model Agent', 'Memory Stream', 'Memory Controller', 'Long-term Memory', 'Short-term Memory', 'Multi-turn Dialogue', 'Ultra-long Document Summarization']
  },
  {
    title: 'Generative AI at Work',
    description: 'A study on the introduction of a generative AI-based conversational assistant in customer support shows increased productivity, especially for novice and low-skilled workers, as well as improved customer sentiment and employee retention.',
    url: 'https://www.nber.org/papers/w31161',
    keywords: ['Generative AI', 'Conversational Assistant', 'Customer Support', 'Productivity', 'Novice Workers', 'Low-skilled Workers', 'Tacit Knowledge', 'Experience Curve', 'Customer Sentiment', 'Employee Retention']
  },
  {
    title: 'REFINER: Reasoning Feedback on Intermediate Representations',
    description: 'REFINER is a framework that improves language models on reasoning tasks by generating intermediate steps and receiving feedback from a critic model.',
    url: 'https://arxiv.org/pdf/2304.01904.pdf',
    keywords: ['Language Models', 'Reasoning Tasks', 'Critic Model', 'Intermediate Inferences', 'Structured Feedback', 'Iterative Improvement', 'GPT3.5', 'Automated Feedback', 'No Human-in-the-loop Data']
  },
  {
    title: 'Building A ChatGPT-enhanced Python REPL',
    description: 'This blog discusses the creation of GEPL, a Python REPL enhanced with ChatGPT, exploring the architecture, prompts, and potential software engineering patterns that may arise from systems built on large language models.',
    url: 'https://isthisit.nz/posts/2023/building-a-chat-gpt-enhanced-python-repl/',
    keywords: ['Machine Learning', 'Data Science', 'AI', 'Python', 'Data Visualization', 'Linear Regression', 'Logistic Regression', 'Decision Trees', 'Neural Networks', 'Deep Learning', 'Natural Language Processing', 'Model Evaluation', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Pandas', 'NumPy', 'Time Series Analysis']
  },
  {
    title: 'Why Does ChatGPT Fall Short in Answering Questions Faithfully?',
    description: 'ChatGPT faces challenges in faithfully answering questions due to shortcomings in comprehension, factualness, specificity, and inference; improvements can be achieved through integrating external knowledge, providing knowledge association hints, and guiding reasoning.',
    url: 'https://arxiv.org/abs/2304.10513',
    keywords: ['Large Language Models', 'ChatGPT', 'faithfulness', 'question answering', 'complex open-domain', 'comprehension', 'factualness', 'specificity', 'inference', 'knowledge memorization', 'knowledge association', 'knowledge reasoning', 'external knowledge', 'hints', 'reasoning guidance']
  },
  {
    title: 'Learning to Program with Natural Language',
    description: 'LLMs can be trained with the Learning to Program (LP) method to understand natural language programs and perform better in complex tasks, showing improved performance in high school and competition math problems.',
    url: 'https://arxiv.org/pdf/2304.10464.pdf',
    keywords:  ['Large Language Models', 'LLMs', 'Artificial General Intelligence', 'natural language', 'programming language', 'task procedures', 'Learning to Program', 'LP method', 'training dataset', 'complex tasks', 'AMPS', 'high school math', 'competition mathematics problems', 'ChatGPT', 'performance', 'zero-shot test']
  },
  {
    title: 'v4.28.0: LLaMa, Pix2Struct, MatCha, DePlot, MEGA, NLLB-MoE, GPTBigCode',
    description: "Hugging Face's new releases include various advancements in language models and techniques, such as LLaMA, Pix2Struct, Mega, GPTBigCode, and NLLB-MoE, as well as the introduction of 8-bit model serialization for improved memory usage and faster loading times.",
    url: 'https://github.com/huggingface/transformers/releases/tag/v4.28.0',
    keywords: ["LLaMA", "Pix2Struct", "Mega", "GPTBigCode", "NLLB-MoE", "Hugging Face", "multi-query attention", "moving average equipped gated attention", "mixture of experts", "8-bit models", "serialization", "memory efficiency"]
  },
  {
    title: 'MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models',
    description: 'MiniGPT-4 combines a frozen visual encoder and an LLM using a single projection layer, undergoing a two-stage training process with self-created high-quality image-text pairs, resulting in efficient vision-language capabilities similar to GPT-4.',
    url: 'https://minigpt-4.github.io/',
    keywords: ["MiniGPT-4", "visual encoder", "BLIP-2", "LLM", "Vicuna", "pretraining", "finetuning", "image-text pairs", "ChatGPT", "vision-language capabilities", "GPT-4"]
  },
  {
    title: 'RedPajama, a project to create leading open-source models, starts by reproducing LLaMA training dataset of over 1.2 trillion tokens',
    description: 'RedPajama is a project aimed at creating fully open-source models, and it has successfully completed its first step by reproducing the LLaMA training dataset with over 1.2 trillion tokens.',
    url: 'https://www.together.xyz/blog/redpajama',
    keywords: ["Foundation models", "GPT-4", "open-source", "RedPajama", "LLaMA training dataset", "1.2 trillion tokens", "AI improvement"]
  },
  {
    title: 'Emergent autonomous scientific research capabilities of large language models',
    description: 'This paper introduces an Intelligent Agent system that combines multiple large language models for autonomous design, planning, and execution of scientific experiments, while addressing safety implications and potential misuse prevention.',
    url: 'https://arxiv.org/ftp/arxiv/papers/2304/2304.05332.pdf',
    keywords: ["Intelligent Agent", "large language models", "autonomous design", "scientific experiments", "cross-coupling reactions", "emergence", "safety implications", "misuse prevention"]
  },
  {
    title: 'Language Instructed Reinforcement Learning for Human-AI Coordination',
    description: 'InstructRL is a novel framework that leverages natural language instructions to guide multi-agent reinforcement learning, resulting in AI agents that align with human preferences and improve human-AI coordination performance in environments like the Hanabi benchmark.',
    url: 'https://arxiv.org/abs/2304.07297',
    keywords: ["instructRL", "AI agents", "human coordination", "natural language instructions", "multi-agent reinforcement learning", "human preferences", "prior policy", "Hanabi benchmark"]
  },
  {
    title: 'Learning to Compress Prompts with Gist Tokens',
    description: 'Gisting is a method that trains language models to compress prompts into smaller "gist" tokens for improved compute efficiency, enabling up to 26x compression, 40% FLOPs reduction, 4.2% wall time speedup, storage savings, and minimal output quality loss.',
    url: 'https://arxiv.org/pdf/2304.08467.pdf',
    keywords: ["gisting", "language models", "prompt compression", "compute efficiency", "FLOPs reduction", "speedup", "storage savings", "output quality"]
  },
  {
    title: 'Scaffolded LLMs as natural language computers',
    description: 'The article discusses the emergence of scaffolded LLMs as a new type of general-purpose natural language computer, highlighting their potential and analogies with digital computers.',
    url: 'https://www.beren.io/2023-04-11-Scaffolded-LLMs-natural-language-computers/',
    keywords: ["LLM-based agents", "AutoGPT", "agentic loop", "scaffolded LLM systems", "GPT4", "natural language computer", "generative agent", "architecture", "von-Neumann computer", "natural language processing unit", "NLPU", "RAM", "memory", "plugins", "scaffolding code", "performance", "FLOPs", "NLOPs", "Moore's law", "exponential improvements"]
  },
  {
    title: 'Building LLM applications for production',
    description: "Chip Huyen's article highlights the challenges of productionizing large language model (LLM) applications, discussing solutions, control flows, and promising use cases.",
    url: 'https://huyenchip.com/2023/04/11/llm-engineering.html',
    keywords: ["workflows", "production-ready", "engineering rigor", "prompt engineering", "natural languages", "challenges", "solutions", "control flows", "SQL executor", "bash", "web browsers", "third-party APIs", "use cases"]
  },
  {
    title: '91% of ML Models Degrade in Time',
    description: 'A recent study involving top institutions found that 91% of machine learning models degrade over time, emphasizing the importance of monitoring model performance in order to maintain accuracy and avoid failure in the ML industry.',
    url: 'https://www.nannyml.com/blog/91-of-ml-perfomance-degrade-in-time',
    keywords: ["model degradation", "performance", "temporal drift", "monitoring", "MIT", "Harvard", "University of Monterrey", "data drift", "NannyML", "covariate shift", "concept drift", "AI aging", "Linear Regression", "Random Forest Regressor", "XGBoost", "Multilayer Perceptron Neural Network", "framework", "mean squared error", "gradual degradation", "explosive degradation", "error variability", "retraining", "ground truth", "production"]
  },
  {
    title: 'A New Approach to Computation Reimagines Artificial Intelligence',
    description: 'Hyperdimensional computing, which represents information as single hyperdimensional vectors, offers a promising alternative to artificial neural networks by providing more efficient, robust, and transparent machine-made decisions.',
    url: 'https://www.quantamagazine.org/a-new-approach-to-computation-reimagines-artificial-intelligence-20230413',
    keywords: ["hyperdimensional computing", "artificial neural networks", "vectors", "efficiency", "robustness", "transparency", "machine-made decisions", "algebra", "symbolic reasoning", "error tolerance", "hardware", "in-memory computing", "orthogonality", "Pentti Kanerva", "Bruno Olshausen"]
  },
  {
    title: 'The Coming of Local LLMs',
    description: 'Advancements in local large language models (LLMs) may lead to more personalized privacy-sensitive applications on edge devices, with Apple and other consumer electronics companies potentially integrating LLMs into their products.',
    url: 'https://nickarner.com/notes/the-coming-of-local-llms-march-23-2023/',
    keywords: ["local LLMs", "edge devices", "privacy", "Apple", "consumer electronics", "language models", "personalized applications", "on-device processing"]
  },
  {
    title: 'Copyright Registration Guidance: Works Containing Material Generated by Artificial Intelligence',
    description: 'The US Copyright Office has provided guidance on policy for works containing AI-generated material, emphasizing that copyright protection only applies to human-authored aspects and that applicants must disclose AI-generated content when registering their works.',
    url: 'https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence',
    keywords: ["copyright", "AI-generated", "human-authored",
    "registration", "disclosure", "policy", "guidance", "application",
    "authorship", "supplementary registration"]
  },
  { 
    title: 'ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application',
    description: 'This paper showcases how ChatGPT can be utilized to translate natural language instructions into executable robot actions, offering customizable prompts for easy integration, adaptability, and user modification in various environments.',
    url: 'https://arxiv.org/pdf/2304.03893.pdf',
    keywords: ["OpenAI", "ChatGPT", "few-shot setting", "natural language instructions", "robot action sequence", "customizable input prompts", "robot execution systems", "operating environment", "predefined robot actions", "safe and robust operation", "open-source", "source code"]
  },
  {
    title: 'Sparks of Artificial General Intelligence: Early experiments with GPT-4',
    description: 'This paper investigates an early version of GPT-4, arguing that it, along with other large language models, exhibits more general intelligence than previous AI models, demonstrating striking human-level performance across various tasks and domains, while also discussing its limitations, the challenges ahead for AGI development,and the societal implications of this technological leap.',
    url: 'https://arxiv.org/pdf/2303.12712.pdf',
    keywords: ["artificial intelligence", "AI", "large language models", "LLMs", "learning", "cognition","OpenAI", "GPT-4", "ChatGPT", "Google's PaLM", "general intelligence", "capabilities", "implications", "task performance", "human-level performance", "artificial general intelligence", "AGI", "limitations", "next-word prediction", "societal influences", "future research directions"]
  },
  {
    title: 'Skeptical optimists are the key to an abundant AI future',
    description: 'This article discusses the three polarizing positions of Doomers, Anti-hypers, and Accelerationists in the AI debate, highlighting the need for skeptical optimists to have a seat at the table in order to build innovative and responsible AI systems.',
    url: 'https://www.treycausey.com/blog/skeptical_optimists.html',
    keywords: ['Doomers', 'Anti-hypers', 'Accelerationists', 'AI debate', 'online phenomena', 'polarization', 'AI community', 'potential effects', 'marginalization, abundance', 'skeptical optimists', 'responsible AI systems', 'optimism', 'regulation', 'diverse viewpoints']
  },
  {
    title: 'Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster',
    description: 'Cerebras-GPT is a family of open compute-optimal language models with state-of-the-art training efficiency, utilizing recent research advances, efficient pre-training, and scaling techniques, with findings on Maximal Update Parameterization (μP) and model availability on HuggingFace.',
    url: 'https://arxiv.org/pdf/2304.03208.pdf',
    keywords: ['efficient pre-training', 'scaling', 'Cerebras-GPT', 'Eleuther Pile dataset', 'DeepMind Chinchilla', 'power-law scaling', 'training efficiency', 'μP', 'hyperparameter predictability', 'pre-trained models', 'HuggingFace', 'compute-optimal model scaling']

  },
  {
    title: 'Do the Rewards Justify the Means?',
    description: "The MACHIAVELLI benchmark evaluates AI agents' Machiavellian tendencies in social decision-making scenarios and explores methods to steer them towards less harmful behaviors, revealing that agents can be designed to be both competent and moral.",
    url: 'https://arxiv.org/pdf/2304.03279.pdf',
    keywords: ['Artificial agents', 'Machiavellian', 'next-token prediction', 'language models', 'GPT-4', 'MACHIAVELLI benchmark', 'scenario labeling', 'harmful behaviors', 'disutility', 'ethical violations', 'machine ethics', 'Pareto improvements', 'safety', 'capabilities']
  },
  {
    title: 'The Potentially Large Effects of Artificial Intelligence on Economic Growth',
    description: "Generative AI's potential to automate tasks and produce human-like content could disrupt labor markets, expose 300 million full-time jobs to automation, and ultimately raise global GDP by 7% if it delivers on its promise.",
    url: 'https://www.ansa.it/documents/1680080409454_ert.pdf',
    keywords: ['generative AI', 'task automation', 'labor market disruption', 'occupational tasks', 'worker displacement', 'job creation', 'productivity growth', 'global GDP', 'economic potential']
  },
  {
    title: 'Measuring trends in Artificial Intelligence',
    description: "The AI Index, an initiative at Stanford's HAI, annually compiles and visualizes AI data from various organizations, assisting decision-makers in advancing AI responsibly and ethically with a focus on humans.",
    url: 'https://aiindex.stanford.edu/report/',
    keywords: ['AI Index', 'Stanford', 'Human-Centered Artificial Intelligence', 'HAI', 'interdisciplinary group', 'decision-makers', 'responsible AI', 'ethical AI', 'collaboration', 'data analysis', 'foundation models', 'geopolitics', 'training costs', 'environmental impact', 'education', 'public opinion trends', 'legislation']
  },
  {
    title: 'Think of language models like ChatGPT as a “calculator for words”',
    description: 'Discover the power of language models like ChatGPT as a "calculator for words," excelling at language manipulation and creative tasks. But be cautious when using them as search engines due to potential inaccuracies and hallucinations.',
    url: 'https://simonwillison.net/2023/Apr/2/calculator-for-words/',
    keywords: ['language models', 'ChatGPT', 'calculator for words', 'search engines', 'inaccuracies', 'hallucinations', 'creative tasks', 'language manipulation', 'context']
  },
  {
    title: 'ChatGPT is a Blurry JPEG',
    description: "ChatGPT, a large language model, can be likened to a blurry jpeg of the entire web's text, offering approximations of information while still producing grammatically coherent responses, making it a useful yet potentially imprecise source of knowledge.",
    url: 'https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web',
    keywords: ['ChatGPT', 'large language model', 'blurry jpeg', 'web text', 'approximation', 'grammatical text', 'information, compression', 'lossy algorithm', 'knowledge']
  },
  {
    title: 'Self-Refine: Iterative Refinement with Self-Feedback',
    description: 'SELF-REFINE is a framework that improves LLM-generated outputs through iterative feedback and refinement, outperforming direct generation across diverse tasks without requiring supervised data or reinforcement learning.',
    url: 'https://arxiv.org/pdf/2303.17651.pdf',
    keywords: ['Text generation', 'SELF-REFINE', 'Iterative feedback', 'Refinement', 'Multi-aspect feedback', 'Unsupervised approach', 'Diverse tasks', 'Performance improvement', 'Human preference']
  },
  {
    title: 'Hyena Hierarchy: Towards Larger Convolutional Language Models',
    description: 'Hyena, a subquadratic drop-in replacement for attention in Transformers, significantly improves accuracy in recall and reasoning tasks, matches attention-based models, and sets a new state-of-the-art for dense-attention-free architectures on language modeling while reducing training compute and increasing speed.',
    url: 'https://arxiv.org/pdf/2302.10866.pdf',
    keywords: ['Hyena', 'subquadratic attention', 'Transformers', 'long convolutions', 'data-controlled gating', 'recall and reasoning tasks', 'language modeling', 'dense-attention-free architectures', 'training compute reduction', 'sequence length']
  },
  {
    title: 'Attention Is All You Need',
    description: 'The Transformer, a new network architecture based solely on attention mechanisms, outperforms existing sequence transduction models in machine translation tasks while being more parallelizable and requiring significantly less training time.',
    url: 'https://arxiv.org/pdf/1706.03762.pdf',
    keywords: ['Transformer', 'network architecture', 'attention mechanisms', 'sequence transduction', 'machine translation', 'parallelizable', 'reduced training time', 'English-to-German', 'English-to-French']
  },
  {
    title: 'BloombergGPT: A Large Language Model for Finance',
    description: 'BloombergGPT is a 50 billion parameter language model trained on a large, mixed financial and general-purpose dataset that significantly outperforms existing models on financial tasks while maintaining strong performance on standard LLM benchmarks.',
    url: 'https://arxiv.org/pdf/2303.17564.pdf',
    keywords: ['BloombergGPT', 'financial domain', 'large language model', 'financial data', 'mixed dataset training', 'domain-specific dataset', 'NLP in finance', 'performance improvement']
  },
  {
    title: 'GPT-4 Is a Reasoning Engine',
    description: 'This article highlights the importance of understanding that AI models like GPT-4 are reasoning engines rather than knowledge databases, and that their usefulness will improve with better access to relevant knowledge at the right time, rather than just advancements in reasoning capabilities.',
    url: 'https://every.to/chain-of-thought/gpt-4-is-a-reasoning-engine',
    keywords: ['GPT-4', 'reasoning engine', 'knowledge database', 'AI models', 'advancements', 'access to knowledge']
  },
  {
    title: 'Alpaca: A Strong, Replicable Instruction-Following Model',
    description: "This article introduces Alpaca, an instruction-following language model fine-tuned from Meta's LLaMA 7B, which demonstrates behaviors similar to OpenAI's text-davinci-003 but is smaller and more accessible for academic research, with the intention of addressing pressing issues in AI models such as false information, social stereotypes, and toxicity.",
    url: 'https://crfm.stanford.edu/2023/03/13/alpaca.html',
    keywords: ['Alpaca', 'instruction-following', 'language model', 'Meta', 'LLaMA', 'OpenAI', 'text-davinci-003', 'false information', 'stereotypes']
  },
  {
    title: 'Llama.cpp 30B Runs on 6GB RAM',
    description: 'Llama.cpp now operates with only 6GB of RAM, offering significant improvements in loading time and user experience; however, a comprehensive explanation for the reduced RAM usage is still being investigated, warranting a healthy degree of skepticism.',
    url: 'https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916',
    keywords: ['Llama', 'RAM', 'loading time', 'performance improvement', 'usability', 'RAM usage']
  },
  {
    title: 'HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace',
    description: 'The paper introduces HuggingGPT, a system that leverages large language models to connect and manage various AI models across domains and modalities, enabling the handling of complicated tasks and paving a new way towards artificial general intelligence.',
    url: 'https://arxiv.org/pdf/2303.17580.pdf',
    keywords: ['HuggingGPT', 'HuggingFace', 'task planning', 'model selection', 'subtask execution', 'language', 'vision', 'speech', 'cross-domain', 'cross-modality', 'AGI']
  },
  {
    title: 'GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models',
    description: 'This study investigates the potential impact of large language models on the U.S. labor market, finding that a significant portion of the workforce could have their tasks affected by LLMs and LLM-powered software, with substantial economic, social, and policy implications.',
    url: 'https://arxiv.org/pdf/2303.10130.pdf',
    keywords: ["Labor market impact", "large language models", "U.S. workforce", "LLM-powered software", "economic implications", "social implications", "policy implications", "task automation", "job disruption", "productivity growth"]
  },
  {
    title: 'Explicit Planning Helps Language Models in Logical Reasoning',
    description: 'This paper presents a novel system that incorporates explicit planning for multi-step logical reasoning in language models, resulting in significant performance improvements over competing systems.',
    url: 'https://arxiv.org/pdf/2303.15714.pdf',
    keywords: ['explicit planning', 'logical reasoning', 'language models', 'multi-step reasoning', 'performance improvement']
  },
  {
    title: 'Can AI-Generated Text be Reliably Detected?',
    description: 'This paper discusses the unreliability of AI-generated text detectors in practical scenarios and the potential consequences, emphasizing the need for an honest conversation about the ethical and responsible use of LLMs.',
    url: 'https://arxiv.org/pdf/2303.11156.pdf',
    keywords: ['AI-generated text', 'text detection', 'reliability', 'practical scenarios', 'ethical use', 'responsible use', 'LLMs']
  },
  {
    title: 'Scaling Expert Language Models with Unsupervised Domain Discovery',
    description: 'This paper presents an asynchronous method for training large, sparse language models by clustering related documents, reducing communication overhead and improving performance compared to dense baselines.',
    url: 'https://arxiv.org/pdf/2303.14177.pdf',
    keywords: ['expert language models', 'unsupervised domain discovery', 'asynchronous training', 'large language models', 'sparse language models', 'communication overhead', 'performance improvement', 'dense baselines']
  },
  {
    title: 'Memorizing Transformers',
    description: 'This paper discusses the possibility of extending language models with the ability to memorize past inputs at inference time, improving performance across various benchmarks and tasks, and allowing the model to utilize newly defined functions and theorems during testing.',
    url: 'https://arxiv.org/pdf/2203.08913.pdf',
    keywords: ['memorizing transformers', 'language models', 'inference time', 'performance improvement', 'new functions', 'new theorems']
  },
  {
    title: 'Could you train a ChatGPT-beating model for $85,000 and run it in a browser?',
    description: 'Simon Willison speculates that it may soon be possible to train a large language model with capabilities similar to GPT-3 for $85,000, run it in a browser, and potentially surpass ChatGPT, despite the current high costs associated with building and training such models.',
    url: 'https://simonwillison.net/2023/Mar/17/beat-chatgpt-in-a-browser/',
    keywords: ['ChatGPT', 'Simon Willison', 'large language model', 'GPT-3', 'browser', 'training cost', 'GPU servers']
  }
];

export default resources;