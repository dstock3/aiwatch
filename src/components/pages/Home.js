import React from 'react';
import { Container, Row, Col, Form, Card, Button } from 'react-bootstrap';
import '../../style/home.css';
import { Helmet } from 'react-helmet';

const Home = () => {
  const handleNewsletterSubmit = (event) => {
    event.preventDefault();
    // Handle newsletter submission logic here
  };

  return (
    <>
      <Helmet>
        <title>AI Watch</title>
      </Helmet>
      
      <Container className="about-container my-5 gradient text-white" style={{ maxWidth: '55%' }}>
        <Row>
          <Col xs={12} md={{ span: 8, offset: 2 }}>
            <h1 className="mb-4">About AI Watch</h1>
            <p>
              AI Watch is a YouTube channel dedicated to providing the latest news, insights, and
              discussions about artificial intelligence. Our goal is to educate and inspire our
              viewers with high-quality content that explores the ever-evolving world of AI.
            </p>
            <p>
              If you're interested in learning more about AI or staying up-to-date with the latest
              developments, subscribe to AI Watch and join our growing community of AI enthusiasts.
            </p>
          </Col>
        </Row>
      </Container>
      
      <Container className="my-5">
        <Row>
          <Card className="home-card gradient text-white">
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
        <Row className="newsletter-container my-5 bg-dark text-white" style={{ maxWidth: '33%' }}>
          <Col className="text-center">
            <h2>Subscribe to our Newsletter</h2>
            <p>Get the latest AI news and updates straight to your inbox:</p>
            <Form onSubmit={handleNewsletterSubmit} inline className="d-flex justify-content-center">
              <Form.Group controlId="newsletterForm.Email">
                <Form.Control type="email" placeholder="Enter your email" />
              </Form.Group>
              <Button variant="primary" type="submit" className="ms-2">
                Subscribe
              </Button>
            </Form>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Home;
