import React, { useState } from 'react';
import { Container, Row, Button, Col, Card } from 'react-bootstrap';
import BlogList from '../sections/BlogList';
import { Helmet } from 'react-helmet';
import blogs from '../../data/blogs';
import { useParams } from 'react-router-dom';
import Update from '../modal/Updates';

const Blog = () => {
  const { blogId } = useParams();
  const [show, setShow] = useState(false);

  let blog
  if (blogId === undefined) {
    blog = blogs[0]
  } else {
    blog = blogs.find((blog) => blog.id === parseInt(blogId, 10));
  }
  
  if (!blog) {
    return (
      <>
        <Helmet>
          <title>404 - Blog Not Found - AI Watch</title>
        </Helmet>
        <Container className="my-5">
          <h1>Blog Not Found</h1>
          <p>The blog you are looking for does not exist or has been removed.</p>
        </Container>
      </>
    );
  } else {
    return (
      <>
        <Helmet>
          <title>{`${blog.title} - AI Watch`}</title>
          <meta
            key="description"
            name="description"
            content={`Read our latest blog post: ${blog.title}. ${blog.text.substring(0, 150)}...`}
          />
          <meta
            key="keywords"
            name="keywords"
            content={`AI, artificial intelligence, machine learning, ${blog.tags.join(', ')}`}
          />
        </Helmet>

        <Container className="my-5">
          <Row>
            <Col md={8}>
              {blog && (
                <Card className="mb-4">
                  <Card.Img
                    variant="top"
                    src={blog.img}
                    alt={blog.title}
                  />
                  <Card.Body>
                    <Card.Title>{blog.title}</Card.Title>
                    <Card.Subtitle className="mt-1 mb-2">
                      <i>{blog.date}</i>
                    </Card.Subtitle>
                    <Card.Text>{blog.text}</Card.Text>
                    <Button onClick={() => setShow(true)}>Want updates?</Button>
                  </Card.Body>
                </Card>
              )}
            </Col>
            <BlogList blogs={blogs} />
          </Row>
        </Container>
        <Update show={show} setShow={setShow} onHide={() => setShow(false)}/>
      </>
    );
  }
};

export default Blog;
