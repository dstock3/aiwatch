import React from 'react';
import { Container, Row, Col, Card, ListGroup } from 'react-bootstrap';

const Blog = () => {
  return (
    <Container className="my-5">
      <Row>
        <Col md={8}>
        <Card className="mb-4">
            <Card.Img
              variant="top"
              src="https://via.placeholder.com/900x300"
              alt="Sample Blog Post"
            />
            <Card.Body>
              <Card.Title>Sample Blog Post Title</Card.Title>
              <Card.Subtitle className="mb-2 text-muted">
                by John Doe - March 1, 2023
              </Card.Subtitle>
              <Card.Text>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.
              </Card.Text>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <h4>Previous Blog Entries</h4>
          <ListGroup>
            <ListGroup.Item action href="#blog-entry-1">
              Blog Entry 1
            </ListGroup.Item>
            <ListGroup.Item action href="#blog-entry-2">
              Blog Entry 2
            </ListGroup.Item>
            <ListGroup.Item action href="#blog-entry-3">
              Blog Entry 3
            </ListGroup.Item>
            <ListGroup.Item action href="#blog-entry-4">
              Blog Entry 4
            </ListGroup.Item>
          </ListGroup>
        </Col>
      </Row>
    </Container>
  );
};

export default Blog;





