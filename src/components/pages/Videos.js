import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import '../../style/videos.css';
import { Helmet } from 'react-helmet';

const Videos = () => {
  return (
    <>
      <Helmet>
        <title>Videos - AI Watch</title>
      </Helmet>
      <Container className="my-5">
        <Row>
          <Col xs={12}>
            <Card className="mb-4 video-card">
              <div className="embed-responsive embed-responsive-16by9 video-container">
                <iframe
                  className="embed-responsive-item"
                  title="Sample Video"
                  src="https://www.youtube.com/embed/rrstrOrJxOc"
                  allowFullScreen
                ></iframe>
              </div>
              <Card.Body>
                <Card.Title>Sample Video Title</Card.Title>
                <Card.Text>
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.
                </Card.Text>
              </Card.Body>
            </Card>
            <Card className="mb-4 video-card">
              <div className="embed-responsive embed-responsive-16by9 video-container">
                <iframe
                  className="embed-responsive-item"
                  title="Sample Video"
                  src="https://www.youtube.com/embed/rrstrOrJxOc"
                  allowFullScreen
                ></iframe>
              </div>
              <Card.Body>
                <Card.Title>Sample Video Title</Card.Title>
                <Card.Text>
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.
                </Card.Text>
              </Card.Body>
            </Card>
            <Card className="mb-4 video-card">
              <div className="embed-responsive embed-responsive-16by9 video-container">
                <iframe
                  className="embed-responsive-item"
                  title="Sample Video"
                  src="https://www.youtube.com/embed/rrstrOrJxOc"
                  allowFullScreen
                ></iframe>
              </div>
              <Card.Body>
                <Card.Title>Sample Video Title</Card.Title>
                <Card.Text>
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.
                </Card.Text>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Videos;