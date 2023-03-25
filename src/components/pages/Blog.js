import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';

const Blog = () => {
  return (
    <Container className="my-5">
      <Row>
        <Col xs={12} lg={{ span: 8, offset: 2 }}>
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
      </Row>
    </Container>
  );
};

export default Blog;