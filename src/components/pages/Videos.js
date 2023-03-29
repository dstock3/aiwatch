import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import '../../style/videos.css';
import { Helmet } from 'react-helmet';

const Videos = () => {
  const videos = [
    {
      title: 'Sample Video Title 1',
      description:
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum.',
      embedUrl: 'https://www.youtube.com/embed/rrstrOrJxOc',
    },
    {
      title: 'Sample Video Title 2',
      description:
        'Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet.',
      embedUrl: 'https://www.youtube.com/embed/rrstrOrJxOc',
    },
    {
      title: 'Sample Video Title 3',
      description:
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum.',
      embedUrl: 'https://www.youtube.com/embed/rrstrOrJxOc',
    },
  ];

  return (
    <>
      <Helmet>
        <title>Videos - AI Watch</title>
      </Helmet>
      <Container className="my-5">
        <Row>
          <Col xs={12}>
            {videos.map((video, index) => (
              <Card key={index} className="mb-4 video-card bg-dark text-white">
                <div className="embed-responsive embed-responsive-16by9 video-container">
                  <iframe
                    className="embed-responsive-item"
                    title={video.title}
                    src={video.embedUrl}
                    allowFullScreen
                  ></iframe>
                </div>
                <Card.Body>
                  <Card.Title>{video.title}</Card.Title>
                  <Card.Text>{video.description}</Card.Text>
                </Card.Body>
              </Card>
            ))}
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default Videos;
