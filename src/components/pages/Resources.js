import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { Helmet } from 'react-helmet';

const Resources = () => {
  const resources = [
    {
      title: 'Resource 1',
      description: 'This is a description of Resource 1.',
      url: 'https://example.com/resource1'
    },
    {
      title: 'Resource 2',
      description: 'This is a description of Resource 2.',
      url: 'https://example.com/resource2'
    },
    {
      title: 'Resource 3',
      description: 'This is a description of Resource 3.',
      url: 'https://example.com/resource3'
    }
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
              <Card className="bg-dark text-white">
                <Card.Body>
                  <Card.Title>{resource.title}</Card.Title>
                  <Card.Text>{resource.description}</Card.Text>
                  <Card.Link href={resource.url} target="_blank" rel="noopener noreferrer">
                    Visit Resource
                  </Card.Link>
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