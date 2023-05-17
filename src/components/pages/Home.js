import React from 'react';
import { Container, Row, Col, Card, Form, Button } from 'react-bootstrap';
import '../../style/home.css';
import { Helmet } from 'react-helmet';

const Home = () => {
  const handleNewsletterSubmit = async (event) => {
    event.preventDefault();
    const email = event.target.elements['newsletterForm.Email'].value;

    try {
      const response = await fetch('https://aiwatch-dstock3.vercel.app/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      alert('Successfully subscribed!');
      event.target.reset();
    } catch (error) {
      alert(`Failed to subscribe: ${error.message}`);
    }
  };

  return (
    <>
      <Helmet>
        <title>AI Watch</title>
      </Helmet>
      <Container fluid className="home-container">
        <Row className="home-row">
          <Col md={6} className="py-5 about-section">
            <Card className="h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h1 className="mb-4 text-center">About AI Watch</h1>
                <p>AI Watch is a news outlet...</p>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6} className="py-5 video-section">
            <Card className="h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h1 className="text-center">Latest Video</h1>
                <div className="embed-responsive embed-responsive-16by9 home-video">
                  <iframe
                    className="embed-responsive-item"
                    title="Latest Video"
                    src="https://www.youtube.com/embed/your-video-id"
                    allowFullScreen
                  ></iframe>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <Row className="home-row">
          <Col className="py-5 newsletter-section">
            <Card className="newsletter-card h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h2 className="text-center">Subscribe to our Newsletter</h2>
                <p className="text-center">Get the latest AI news...</p>
                <Form onSubmit={handleNewsletterSubmit} className="d-flex justify-content-center">
                  <Form.Group controlId="newsletterForm.Email">
                    <Form.Control type="email" placeholder="Enter your email" />
                  </Form.Group>
                  <Button variant="primary" type="submit" className="ms-2">
                    Subscribe
                  </Button>
                </Form>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Home;
