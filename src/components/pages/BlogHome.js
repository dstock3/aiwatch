import React from 'react';
import { Container, Row } from 'react-bootstrap';
import BlogList from '../sections/BlogList';
import { Helmet } from 'react-helmet';
import blogs from '../../data/blogs';

const BlogHome = () => {
  return (
    <>
      <Helmet>
        <title>Blog - AI Watch</title>
        <meta
          key="description"
          name="description"
          content="Explore our latest blog posts about AI and machine learning."
        />
        <meta
          key="keywords"
          name="keywords"
          content="AI, artificial intelligence, machine learning, blog"
        />
      </Helmet>

      <Container className="my-5">
        <Row>
          <h1 className="mb-4">Blog</h1>
          <BlogList blogs={blogs} />
        </Row>
      </Container>
    </>
  );
};

export default BlogHome;
