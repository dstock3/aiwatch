import React from 'react';
import { Container, Row, Col, Form, Card, Button } from 'react-bootstrap';
import '../../style/home.css';

const Home = () => {
  const handleNewsletterSubmit = (event) => {
    event.preventDefault();
    // Handle newsletter submission logic here
  };

  return (
    <Container className="my-5">
      <Row>
        <Card className="home-card">
          <h1 className="text-center">Latest Video</h1>
          <div className="embed-responsive embed-responsive-16by9 home-container">
            <iframe
              className="embed-responsive-video"
              title="Latest Video"
              src="https://www.youtube.com/embed/your-video-id"
              allowFullScreen
            ></iframe>
          </div>
        </Card>
      </Row>
      <Row className="my-5">
        <Col className="text-center">
          <h2>Subscribe to our Newsletter</h2>
          <p>Get the latest AI news and updates straight to your inbox:</p>
          <Form onSubmit={handleNewsletterSubmit} inline className="d-flex justify-content-center">
            <Form.Group controlId="newsletterForm.Email" className="mb-3">
              <Form.Control type="email" placeholder="Enter your email" />
            </Form.Group>
            <Button variant="primary" type="submit" className="ms-2">
              Subscribe
            </Button>
          </Form>
        </Col>
      </Row>
    </Container>
  );
};

export default Home;
