import React from 'react';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';
import { Helmet } from 'react-helmet';

const Contact = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    // Handle form submission logic here
  };

  return (
    <>
      <Helmet>
        <title>Contact Us - AI Watch</title>
      </Helmet>
      <Container className="my-5 bg-dark text-white">
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
              <Button variant="primary" type="submit">
                Submit
              </Button>
            </Form>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Contact;
