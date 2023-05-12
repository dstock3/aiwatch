import React, { useState } from 'react';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';
import { Helmet } from 'react-helmet';
import '../../style/contact.css';

const Contact = () => {
  const [formSubmitted, setFormSubmitted] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    try {
      const response = await fetch('https://aiwatch-dstock3.vercel.app/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      setFormSubmitted(true);
      event.target.reset();
    } catch (error) {
      alert(`Failed to submit the contact form: ${error.message}`);
    }
  };

  return (
    <>
      <Helmet>
        <title>Contact Us - AI Watch</title>
      </Helmet>
      <Container className="contact-container my-5" style={{ maxWidth: '50%' }}>
        <Row>
          <Col className="text-center">
            <h1>Contact Us</h1>
            <p>Feel free to reach out to us using the form below:</p>
          </Col>
        </Row>
        <Row>
          <Col md={{ span: 8, offset: 2 }}>
            <Form onSubmit={handleSubmit}>
              <Form.Group className="mb-3" controlId="contactForm.Name">
                <Form.Label>Name</Form.Label>
                <Form.Control type="text" placeholder="Enter your name" />
              </Form.Group>
              <Form.Group className="mb-3" controlId="contactForm.Email">
                <Form.Label>Email address</Form.Label>
                <Form.Control type="email" placeholder="Enter your email" />
              </Form.Group>
              <Form.Group className="mb-3" controlId="contactForm.Message">
                <Form.Label>Message</Form.Label>
                <Form.Control as="textarea" rows={3} placeholder="Enter your message" />
              </Form.Group>

              <div className="d-flex justify-content-end">
                <Button variant="primary" type="submit">
                  Submit
                </Button>
              </div>

            </Form>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Contact;