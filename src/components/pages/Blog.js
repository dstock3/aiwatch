import React, { useState } from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import BlogList from '../sections/BlogList';

const Blog = () => {
  const blogs = [
    { title: 'Blog Entry 1',
      date: 'March 1, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.',
      img: 'https://via.placeholder.com/900x300'
    },
    { title: 'Blog Entry 2',
      date: 'March 3, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.',
      img: 'https://via.placeholder.com/900x300'
    },
    { title: 'Blog Entry 3',
      date: 'March 5, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.',
      img: 'https://via.placeholder.com/900x300'
    },
    { title: 'Blog Entry 4',
      date: 'March 9, 2023',
      text: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Duis elementum euismod quam, vitae condimentum mi ullamcorper vel.',
      img: 'https://via.placeholder.com/900x300'
    }
  ];

  const [selectedBlog, setSelectedBlog] = useState(blogs[0]);

  const handleBlogSelect = (blog) => {
    setSelectedBlog(blog);
  }

  return (
    <Container className="my-5">
      <Row>
        <Col md={8}>
          {selectedBlog && (
            <Card className="mb-4">
              <Card.Img
                variant="top"
                src={selectedBlog.img}
                alt={selectedBlog.title}
              />
              <Card.Body>
                <Card.Title>{selectedBlog.title}</Card.Title>
                <Card.Subtitle className="mb-2 text-muted">
                  {selectedBlog.date}
                </Card.Subtitle>
                <Card.Text>{selectedBlog.text}</Card.Text>
              </Card.Body>
            </Card>
          )}
        </Col>
        <BlogList blogs={blogs} onBlogSelect={handleBlogSelect} />
      </Row>
    </Container>
  );
};

export default Blog;





