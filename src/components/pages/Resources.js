import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { Helmet } from 'react-helmet';
import '../../style/resources.css';

const Resources = () => {
  const resources = [
    {
      title: 'HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace',
      description: 'The paper introduces HuggingGPT, a system that leverages large language models to connect and manage various AI models across domains and modalities, enabling the handling of complicated tasks and paving a new way towards artificial general intelligence.',
      url: 'https://arxiv.org/pdf/2303.17580.pdf'
    },
    {
      title: 'GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models',
      description: 'This study investigates the potential impact of large language models on the U.S. labor market, finding that a significant portion of the workforce could have their tasks affected by LLMs and LLM-powered software, with substantial economic, social, and policy implications.',
      url: 'https://arxiv.org/pdf/2303.10130.pdf'
    },
    {
      title: 'Explicit Planning Helps Language Models in Logical Reasoning',
      description: 'This paper presents a novel system that incorporates explicit planning for multi-step logical reasoning in language models, resulting in significant performance improvements over competing systems.',
      url: 'https://arxiv.org/pdf/2303.15714.pdf'
    },
    {
      title: 'Can AI-Generated Text be Reliably Detected?',
      description: 'This paper discusses the unreliability of AI-generated text detectors in practical scenarios and the potential consequences, emphasizing the need for an honest conversation about the ethical and responsible use of LLMs.',
      url: 'https://arxiv.org/pdf/2303.11156.pdf'
    },
    {
      title: 'Scaling Expert Language Models with Unsupervised Domain Discovery',
      description: 'This paper presents an asynchronous method for training large, sparse language models by clustering related documents, reducing communication overhead and improving performance compared to dense baselines.',
      url: 'https://arxiv.org/pdf/2303.14177.pdf'
    },
  ];

  return (
    <>
      <Helmet>
        <title>Resources - AI Watch</title>
      </Helmet>
      <Container className="my-5">
        <Row>
          <Col>
            <h1>AI Resources</h1>
          </Col>
        </Row>
        <Row>
          {resources.map((resource, index) => (
            <Col key={index} xs={12} md={4} className="mb-4">
              <Card className="resource-container gradient text-white">
                <Card.Body className="resource-card-body">
                  <Card.Title>{resource.title}</Card.Title>
                  <Card.Text>{resource.description}</Card.Text>
                  <div className="d-flex justify-content-end">
                    <Card.Link href={resource.url} target="_blank" rel="noopener noreferrer">
                      Visit Resource
                    </Card.Link>
                  </div>
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