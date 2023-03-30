import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import { Helmet } from 'react-helmet';
import '../../style/about.css';

const About = () => {
  return (
    <>
      <Helmet>
        <title>About - AI Watch</title>
      </Helmet>

      <Container className="about-container my-5 bg-dark text-white" style={{ maxWidth: '55%' }}>
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
    </>
  );
};

export default About;


