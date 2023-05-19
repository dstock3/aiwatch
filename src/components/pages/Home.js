import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert } from 'react-bootstrap';
import '../../style/home.css';
import { Helmet } from 'react-helmet';
import ReactPlayer from 'react-player/youtube';

const Home = () => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  
  const handleNewsletterSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
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

      setMessage('Successfully subscribed!');
      event.target.reset();
    } catch (error) {
      setMessage(`Failed to subscribe: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Helmet>
        <title>AI Watch</title>
      </Helmet>
      <Container fluid className="home-container d-flex flex-column align-items-center">
        <Row className="home-row w-100">
          <Col md={6} className="py-5 about-section">
            <Card className="h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h1 className="mb-4 text-center">About AI Watch</h1>
                <p>
                  AI Watch is a dedicated digital platform committed to exploring the 
                  advances, impacts, and the future of artificial intelligence. 
                  Our goal is to ensure that the public, from tech enthusiasts to AI 
                  professionals, are informed and engaged in the ongoing AI discourse.
                </p>
                <p>
                  We provide the latest news, insightful articles, video discussions, and 
                  interviews with leading AI researchers and innovators. Our team strives to 
                  present balanced, in-depth analysis to help our audience understand not only 
                  the latest AI technologies but also the policy, ethical, and societal 
                  implications they have.
                </p>
                <p>
                  At AI Watch, we believe in the transformative potential of artificial 
                  intelligence. Join us in exploring this exciting technology frontier 
                  that's shaping our future.
                </p>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6} className="py-5 video-section">
            <Card className="h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h1 className="text-center">Latest Video</h1>
                <div className="home-video">
                  <ReactPlayer
                    url='https://www.youtube.com/watch?v=your-video-id'
                    controls={true}
                    width='100%'
                    height='100%'
                  />
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <Row className="home-row w-100">
          <Col className="py-5 newsletter-section">
            <Card className="newsletter-card h-100 shadow-lg bg-dark text-white rounded">
              <Card.Body>
                <h2 className="text-center">Subscribe to our Newsletter</h2>
                <p className="text-center">Get the latest AI news...</p>
                {message && <Alert variant={loading ? "info" : "danger"}>{message}</Alert>}
                <Form onSubmit={handleNewsletterSubmit} className="d-flex justify-content-center">
                  <Form.Group controlId="newsletterForm.Email">
                    <Form.Control type="email" placeholder="Enter your email" />
                  </Form.Group>
                  <Button variant="primary" type="submit" className="ms-2" disabled={loading}>
                    {loading ? "Subscribing..." : "Subscribe"}
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
