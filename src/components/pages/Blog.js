import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import BlogList from '../sections/BlogList';

const Blog = () => {
  const blogs = [
    { title: 'Blog Entry 1',
      date: 'March 1, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.'
    },
    { title: 'Blog Entry 2',
      date: 'March 3, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.'
    },
    { title: 'Blog Entry 3',
      date: 'March 5, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.'
    },
    { title: 'Blog Entry 4',
      date: 'March 9, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.'
    }
  ];

  return (
    <Container className="my-5">
      <Row>
        <Col md={8}>
        <Card className="mb-4">
            <Card.Img
              variant="top"
              src="https://via.placeholder.com/900x300"
              alt="Sample Blog Post"
            />
            <Card.Body>
              <Card.Title>Sample Blog Post Title</Card.Title>
              <Card.Subtitle className="mb-2 text-muted">
                by John Doe - March 1, 2023
              </Card.Subtitle>
              <Card.Text>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.
              </Card.Text>
            </Card.Body>
          </Card>
        </Col>
        <BlogList blogs={blogs} />
      </Row>
    </Container>
  );
};

export default Blog;





