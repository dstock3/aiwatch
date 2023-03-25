import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';

const Videos = () => {
  return (
    <Container className="my-5">
      <Row>
        <Col xs={12} lg={{ span: 8, offset: 2 }}>
          <Card className="mb-4">
            <div className="embed-responsive embed-responsive-16by9">
              <iframe
                className="embed-responsive-item"
                title="Sample Video"
                src="https://www.youtube.com/embed/your-video-id"
                allowFullScreen
              ></iframe>
            </div>
            <Card.Body>
              <Card.Title>Sample Video Title</Card.Title>
              <Card.Text>
                This is a sample video description. You can replace this content with your
                own video details and customize the layout and style as needed. Lorem ipsum
                dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae
                vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper
                vel.
              </Card.Text>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Videos;