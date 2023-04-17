import React from 'react';
import { Container, Row, Col, Form, Card, Button } from 'react-bootstrap';
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
      <Container className="about-container my-5" style={{ maxWidth: '55%' }}>
        // ...
      </Container>
      
      <Container className="my-5">
        // ...
        <Row className="newsletter-container my-5 pb-4" style={{ maxWidth: '33%' }}>
          <Col>
            <h2 className="text-center">Subscribe to our Newsletter</h2>
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