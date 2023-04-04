import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, FormControl, Button } from 'react-bootstrap';
import { Helmet } from 'react-helmet';
import '../../style/resources.css';

const Resources = () => {
  const resources = [
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

  const [searchKeyword, setSearchKeyword] = useState('');

  const handleKeywordClick = (keyword) => {
    setSearchKeyword(keyword);
  };

  const filteredResources = resources.filter((resource) => {
    if (searchKeyword === '') return true;
  
    const searchWords = searchKeyword.toLowerCase().split(' ');
    return searchWords.some((word) =>
      resource.keywords.some((keyword) => keyword.toLowerCase().includes(word))
    );
  });
  
  const handleSearchChange = (event) => {
    setSearchKeyword(event.target.value);
  };

  return (
    <>
      <Helmet>
        <title>Resources - AI Watch</title>
      </Helmet>
      <Container className="my-5 resources-container">
        <Row className="align-items-center mb-4">
          <Col xs={12} md={6}>
            <h1>Resources</h1>
          </Col>
          <Col xs={12} md={6}>
            <Form className="d-flex">
              <FormControl
                type="search"
                placeholder="Search by keyword"
                className="mr-2 search-input"
                value={searchKeyword}
                onChange={handleSearchChange}
              />
              <Button variant="outline-primary" className="search-button">
                Search
              </Button>
            </Form>
          </Col>
        </Row>
        <Row>
          {filteredResources.map((resource, index) => (
            <Col key={index} xs={12} md={4} className="mb-4">
              <Card className="resource-container gradient text-white">
                <Card.Body className="resource-card-body">
                  <Card.Title>
                    <a href={resource.url} target="_blank" rel="noopener noreferrer">{resource.title}</a>
                  </Card.Title>
                  <hr className="horizontal-rule" />
                  <Card.Text>{resource.description}</Card.Text>

                  {resource.keywords && (
                    <div className="mt-2">
                      {resource.keywords.map((keyword, idx) => (
                        <span
                          key={idx}
                          className="keyword-box"
                          onClick={() => handleKeywordClick(keyword)}
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      </Container>
    </>
  );
};

export default Resources;